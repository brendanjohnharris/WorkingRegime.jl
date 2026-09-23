#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate "WRCircuit"
using Bootstrap
using WRCircuit
using JLD2
using LinearAlgebra
using Optim
using MoreMaps
using ForwardDiff
using Fathom
using CairoMakie
WRCircuit.@preamble
set_theme!(Fathom.fathom(:physics))

begin
    @info "Loading data"
    rawfile = datadir("demo_run.jld2")   # now holds only the last 5 s of raw traces + scalars
    fixed_params = load(rawfile, "fixed_params")
    epositions = load(rawfile, "epositions")
    ipositions = load(rawfile, "ipositions")
    N = load(rawfile, "N")              # grid side (√nE); the raw arrays are subsampled in time, not space
    mn = load(rawfile, "mean_V")        # mean membrane potential (mV), over the full trace
    nu = load(rawfile, "nu")            # E firing rate (Hz), over the full trace
    dx = fixed_params.dx
    spikes = load(rawfile, "E_spike")   # E spike raster (last 5 s)
    ispikes = load(rawfile, "I_spike")  # I spike raster (last 5 s)
    tmin = minimum(times(spikes)) .- step(spikes)
    tmax = maximum(times(spikes))
end

begin
    spike_times = map(eachslice(spikes, dims = Neuron)) do s
        sts = times(s)[findall(s)]
    end
end
begin  # * Spike raster (E + I)
    ispike_times = map(eachslice(ispikes, dims = Neuron)) do s
        sts = times(s)[findall(s)]
    end

    radius = 0.1 # mm; the project-wide patch radius, matching the spatial sweeps and Fig 1
    origin = [dx / 2, dx / 2]
    emask = map(epositions) do pos
        dp = abs.(pos .- origin)
        dp = min.(dp, dx .- dp)
        norm(dp) < radius
    end # scatter(positions.|> Point2f, color=mask) to check
    elocal_idxs = findall(emask)

    # ipositions = m.I.positions
    # ipositions = map(ipositions) do pos
    #     map(pos) do p
    #         p.tolist() |> convert2(Float32)
    #     end
    # end
    imask = map(ipositions) do pos
        dp = abs.(pos .- origin)
        dp = min.(dp, dx .- dp)
        norm(dp) < radius
    end # scatter(positions.|> Point2f, color=mask) to check
    ilocal_idxs = findall(imask)

    f = OnePanel()
    ax = Axis(f[1, 1]; ylabel = "Excitatory")
    hidexdecorations!(ax)
    hideydecorations!(ax; label = false)

    # A 500 ms window inside the saved last-5s slice (the old 9 s mark is no longer on disk).
    intrvl = (tmin + 2u"s") .. (tmin + 2.5u"s")

    for (i, s) in enumerate(spike_times[elocal_idxs])
        idxs = s .∈ [intrvl]
        scatter!(
            ax, ustripall(s[idxs] .- minimum(intrvl)), i * ones(sum(idxs)),
            color = qinghai,
            markersize = 3
        )
    end

    ax2 = Axis(f[2, 1]; xlabel = "Time (ms)", ylabel = "Inhibitory")
    hideydecorations!(ax2; label = false)
    for (i, s) in enumerate(spike_times[ilocal_idxs])
        idxs = s .∈ [intrvl]
        scatter!(
            ax2, ustripall(s[idxs] .- minimum(intrvl)), i * ones(sum(idxs)),
            color = bermejo,
            markersize = 3
        )
    end
    # hidedecorations!(ax2)
    linkxaxes!(ax, ax2)
    display(f)
    save(plotsdir("spike_example.pdf"), f)
end


begin # * Load precomputed statistics (computed in scripts/demo_run.jl)
    statsfile = datadir("demo_run_stats.jld2")
    fano = load(statsfile, "fano")
    mfano = load(statsfile, "mfano")
    spectra = load(statsfile, "spectra")
    mads = load(statsfile, "mads")
    spectrum_fit = load(statsfile, "spectrum_fit")
    spectrum_fits = load(statsfile, "spectrum_fits")
    mad_fit = load(statsfile, "mad_fit")
    mad_fits = load(statsfile, "mad_fits")
    αs = load(statsfile, "αs")
    βs = load(statsfile, "βs")
    μs = load(statsfile, "μs")
    σs = load(statsfile, "σs")
    V_hist = load(statsfile, "V_hist")     # precomputed membrane-potential density
    dI_hist = load(statsfile, "dI_hist")   # precomputed |ΔI| step-size density
end

# The variability exponent, shared with the sweep and experiment pipelines; see
# WRCircuit/src/Variability.jl. Fit to the neuron-MEDIAN curve: per-neuron curves (55 s of
# spikes) have no SNR for a free-knot fit, and their largest-rise median reads 0.78 against 0.28
# for the aggregated curve.

begin # * Fano exponent from the neuron-median curve, with a split-half reliability check
    @info "Fitting neuron-median Fano curve (unified estimator)"
    fano_median = dropdims(median(fano, dims = 2), dims = 2)
    fano_fit = variability_exponent(fano_median)
    fano_split = map((1:2:size(fano, 2), 2:2:size(fano, 2))) do js # odd/even neuron halves
        variability_exponent(dropdims(median(fano[:, js], dims = 2), dims = 2)).β
    end
end

begin # * Fano factor statistics
    open(plotsdir("critical_demo", "fano_statistics.txt"), "w") do f
        write(f, "unified (BIC-selected fit to neuron-median curve): $(fano_fit)\n")
        write(f, "split-half β (odd/even neurons): $(fano_split), Δ = $(abs(-(fano_split...)))\n")
    end
end


begin # * Membrane potential and input traces (last 5 s), for the trace/trajectory panels
    V = load(rawfile, "E_V")
    V = set(V, 𝑡 => convert2(u"s", times(V)))
    input = load(rawfile, "E_input")
    input = set(input, 𝑡 => convert2(u"s", times(input)))
    # LFP is not rebuilt here (its spectra/MAD are precomputed in the stats file); the stats-plot loops
    # below index the loaded NamedTuples by their keys directly (keys(spectra) = V/LFP/input).
end


if false
    f = SixPanel()
    gs = permutedims(subdivide(f, 3, 2), (2, 1))

    axs = map(enumerate(keys(spectra))) do (i, v)
        s = fit_spectra[v].s
        _s = fit_spectra[v]._s
        fitted_s = fit_spectra[v].fitted_s
        m = fit_spectra[v].m

        ax = Axis(
            gs[i, 1]; xscale = log10, yscale = log10, title = string(v),
            xlabel = "Frequency (Hz)", ylabel = "PSD"
        )
        lines!(ax, s; color = baikal, alpha = 0.4)
        # scatter!(ax, _s; color = baikal, markersize = 10)
        lines!(ax, fitted_s; color = bermejo, linestyle = :dash)
        text = m.params.components.β |> last
        text = "b = $(round(text, digits = 2))"
        text!(
            ax, 0.1, 0.1; text, fontsize = 16, space = :relative,
            align = (:left, :bottom)
        )
        return ax
    end
    # linkaxes!(axs...)

    axs = map(enumerate(keys(spectra))) do (i, v)
        s = fit_mads[v].s
        _s = s #fit_mads[v]._s
        fitted_s = fit_mads[v].fitted_s
        m = fit_mads[v].m

        ax = Axis(
            gs[i, 2]; xscale = log10, yscale = log10, title = string(v),
            xlabel = "Time lag (s)", ylabel = "MSD"
        )
        lines!(ax, s; color = baikal, alpha = 0.4)
        # scatter!(ax, _s; color = baikal, markersize = 10)
        lines!(ax, fitted_s; color = bermejo, linestyle = :dash)
        text = m.params.components.β |> first
        text = "a = $(round(text, digits = 2))"
        text!(
            ax, 0.1, 0.1; text, fontsize = 16, space = :relative,
            align = (:left, :bottom)
        )
        return ax
    end
    # linkaxes!(axs...)
    display(f)
end

begin # * Individual statistics
    # * Spectrum
    fs = [OnePanel() for _ in 1:3]
    axs = map(fs, keys(spectra)) do f, v
        s = spectra[v] |> ustripall
        s = median(s, dims = 2)
        s = dropdims(s, dims = 2)
        s = s[𝑓 = 1 .. 1000]
        # s = spectrum_fit[v].s
        _s = spectrum_fit[v]._s
        fitted_s = spectrum_fit[v].fitted_s
        m = spectrum_fit[v].m

        ax = Axis(
            f[1, 1]; xscale = log10, yscale = log10, title = string(v),
            xlabel = "Frequency (Hz)", ylabel = "PSD"
        )
        lines!(ax, s; color = baikal)
        # scatter!(ax, _s; color = baikal, markersize = 10)
        lines!(ax, fitted_s; color = bermejo, linestyle = :dash)
        text = m.params.components.β |> last
        text = "b = $(round(text, digits = 2))"
        text!(
            ax, 0.1, 0.1; text, fontsize = 16, space = :relative,
            align = (:left, :bottom)
        )
        wsave(plotsdir("critical_demo", "$(v)_spectrum.svg"), f)
    end

    # * MAD
    fs = [OnePanel() for _ in 1:3]
    axs = map(fs, keys(spectra)) do f, v
        s = mads[v] |> ustripall
        s = median(s, dims = 2)
        s = dropdims(s, dims = 2)
        # _s = mad_fit[v]._s
        fitted_s = mad_fit[v].fitted_s
        m = mad_fit[v].m

        ax = Axis(
            f[1, 1]; xscale = log10, yscale = log10, title = string(v),
            xlabel = "Time lag (s)"
        )
        lines!(ax, s; color = baikal)
        # scatter!(ax, _s; color = baikal, markersize = 10)
        lines!(ax, fitted_s; color = bermejo, linestyle = :dash)
        text = m.params.components.β |> first
        text = "a = $(round(text, digits = 2))"
        text!(
            ax, 0.1, 0.1; text, fontsize = 16, space = :relative,
            align = (:left, :bottom)
        )
        wsave(plotsdir("critical_demo", "$(v)_mad.svg"), f)
    end
end


begin # * Statistics
    open(plotsdir("critical_demo", "statistics.txt"), "w") do f
        for v in keys(spectra)
            println(f, "\n=== Variable: $(v) ===")
            println(f, "-- Spectrum fit --")
            m = map(spectrum_fits[v]) do x
                x.m.params.components.β |> last
            end
            stat = TimeseriesTools.bootstrapmedian(m)
            write(f, "$(stat)\n")

            println(f, "-- MAD fit --")
            m = map(mad_fits[v]) do x
                if x isa Number # ie nan
                    return x
                else
                    x.m.params.components.β |> first
                end
            end
            stat = TimeseriesTools.bootstrapmedian(m)
            write(f, "$(stat)\n")
        end
    end
end
begin # * Save pre-computed curves for combined plotting
    @info "Saving pre-computed circuit curves"

    # Input MAD
    input_mad = mads.input |> ustripall
    input_mad_median = median(input_mad, dims = 2)
    input_mad_median = dropdims(input_mad_median, dims = 2)
    input_mad_fit = mad_fit.input
    input_mad_exponent = input_mad_fit.m.params.components.β |> first
    input_mad_exponents = map(mad_fits.input) do x
        x.m.params.components.β |> first
    end

    # Input PSD
    input_psd = spectra.input |> ustripall
    input_psd_median = median(input_psd, dims = 2)
    input_psd_median = dropdims(input_psd_median, dims = 2)
    input_psd_median = input_psd_median[𝑓 = 1 .. 1000]
    input_psd_fit = spectrum_fit.input
    input_psd_exponent = input_psd_fit.m.params.components.β |> last
    input_psd_exponents = map(spectrum_fits.input) do x
        x.m.params.components.β |> last
    end

    # Fano factor (median across neurons; curve and fit computed above)
    fano_exponent = fano_fit.β

    circuit_curves = (;
        mad = (;
            t = collect(lookup(input_mad_median, 𝑡)),
            mu = collect(input_mad_median),
            fit_t = collect(lookup(input_mad_fit.s, 𝑡)),
            fit_vals = collect(input_mad_fit.fitted_s),
            exponent = input_mad_exponent,
            exponents = input_mad_exponents,
        ),
        psd = (;
            f = collect(lookup(input_psd_median, 𝑓)),
            mu = collect(input_psd_median),
            fit_f = collect(lookup(input_psd_fit.fitted_s, 𝑓)),
            fit_vals = collect(input_psd_fit.fitted_s),
            exponent = input_psd_exponent,
            exponents = input_psd_exponents,
        ),
        fano = (;
            t = collect(lookup(fano_median, 𝑡)),
            mu = collect(fano_median),
            exponent = fano_exponent,
            band = (fano_fit.lo, fano_fit.hi), # ms, the fit's measured scaling regime
            ncomponents = fano_fit.ncomponents, # 3 = a segment above the band, 2 = none
            censored = fano_fit.censored, # true = the band's top is the window end, not a knot
            split = fano_split, # odd/even neuron-half βs
        ),
    )

    mkpath(datadir("plots"))
    jldsave(datadir("circuit_curves.jld2"); circuit_curves)
    @info "Saved circuit curves to $(datadir("circuit_curves.jld2"))"
end

# The across-neuron distributions of αs/βs/μs/σs are drawn by scripts/FigS4_input_parameters.jl,
# which reads them straight out of demo_run_stats.jld2.

# Render a precomputed density histogram `h` (a `histcounts` ToolsArray: bin centre -> density) in the
# Fathom `ziggurat` style: filled translucent bars with a step outline over the top. On a log axis pass
# `logy = true` to drop the leftmost bin (its left edge is 0) and mask nonpositive heights.
function zigg!(ax, h; color = baikal, logy = false, dropfirst = logy)
    centers = collect(lookup(h, 1))
    pdf = collect(h)
    w = centers[2] - centers[1]
    edges = [centers .- w / 2; centers[end] + w / 2]
    e = dropfirst ? edges[2:end] : edges
    c = dropfirst ? centers[2:end] : centers
    p = dropfirst ? pdf[2:end] : pdf
    if dropfirst                       # renormalise to unit area over the shown bins (matches the old bins[2:end])
        s = sum(p) * w
        s > 0 && (p = p ./ s)
    end
    barplot!(ax, c, p; width = w, gap = 0, color = (color, 0.5), strokewidth = 0)
    ys = Float64.([p; last(p)])
    logy && (ys[ys .<= 0] .= NaN)
    return stairs!(ax, e, ys; step = :post, color = color)
end

begin # * Additional properties: image and distribution fit
    mf = TwoPanel(; size = (720, 324))
    myna = 27
    # Sample indices INTO the saved last-5s window (50 000 samples at dt = 0.1 ms). The old absolute
    # marks (18.3 s, 1.35-1.85 s) are no longer on disk, so pick a COM window and a trace window inside it.
    t = 3.0 * 10000 |> Int # Samples (3 s into the window)
    deltat = 0.117 * 10000 |> round |> Int # Samples
    shift = (-10, 5)
    input_ts = 5000:10000 # 0.5-1.0 s into the window

    g = mf[1, 1:2] = GridLayout()
    gg = g[1, 2] = GridLayout()
    hg = g[1, 1] = GridLayout()

    function track_com(field)
        # field is expected to be (time × x × y)
        # Assumes periodic boundary conditions (torus topology)
        nt, nx, ny = size(field)

        # Preallocate output vectors
        com_x = zeros(nt)
        com_y = zeros(nt)

        # Create coordinate grids (0-indexed for proper angular mapping)
        x_coords = 0:(nx - 1)
        y_coords = 0:(ny - 1)

        # Calculate center of mass for each time point
        for t in 1:nt
            slice = field[t, :, :]

            # Calculate total intensity (use absolute value to handle negative fields)
            weights = abs.(slice)
            total_weight = sum(weights)

            # Skip if total intensity is too small (avoid division by zero)
            if total_weight < 1.0e-10
                com_x[t] = nx / 2.0
                com_y[t] = ny / 2.0
                continue
            end

            # Convert to angles for periodic domain
            # θ = 2π * coordinate / domain_size
            θx = 2π .* x_coords' ./ nx
            θy = 2π .* y_coords ./ ny

            # Calculate weighted sum of unit vectors (circular mean)
            ξx = sum(weights .* cos.(θx)) / total_weight
            ζx = sum(weights .* sin.(θx)) / total_weight
            ξy = sum(weights .* cos.(θy)) / total_weight
            ζy = sum(weights .* sin.(θy)) / total_weight

            # Convert back to coordinates using atan
            θ_com_x = atan(ζx, ξx)
            θ_com_y = atan(ζy, ξy)

            # Map from [-π, π] back to [0, domain_size)
            # Add 1 to convert from 0-indexed to 1-indexed
            com_x[t] = mod(θ_com_x * nx / (2π), nx) + 1
            com_y[t] = mod(θ_com_y * ny / (2π), ny) + 1
        end

        return com_x, com_y
    end

    input_grid = reshape(input, (size(input, 1), N, N))

    # Shift only the window we use (cosmetic re-centring to dodge torus
    # wraparound), not the whole multi-GB trace.
    window = circshift(input_grid[(t - deltat):t, :, :], (0, shift...))
    frame = window[end, :, :] # last window frame == shifted input_grid[t]

    xs, ys = track_com(window)
    xs = xs[1:3:end]
    ys = ys[1:3:end]
    color = (0:deltat)[1:3:end] ./ 1000

    xx = range(0, dx, length = N)
    xs = dx .* xs ./ N
    ys = dx .* ys ./ N

    ax = Axis(
        hg[1, 1]; xlabel = "X (mm)", ylabel = "Y (mm)",
        limits = ((0, dx), (0, dx)), xticks = 0:0.25:0.5,
        yticks = 0:0.25:0.5, xtickformat = terseticks,
        ytickformat = terseticks
    )

    h = heatmap!(
        ax, xx, xx, frame';
        colormap = seethrough(reverse(sunrise))
    )
    lines!(ax, xs, ys; color = :white, linewidth = 3)
    p = lines!(
        ax, xs, ys; color,
        colormap = reverse(cgrad(:turbo)),
        linewidth = 2
    )
    Colorbar(hg[1, 2], h; label = "Input current (nA)")
    Colorbar(
        hg[0, 1], p; vertical = false, label = "Time (s)",
        tickformat = terseticks
    )

    rowgap!(hg, 1, Relative(0.06))
    colgap!(hg, 1, Relative(0.05))
    display(mf)

    # * Input distribution
    # * choose the neuron with the distribution closest to the average
    # mps = [mean(αs), mean(βs), mean(μs), mean(σs)]
    # dds = map(ds) do d
    #     [d.α, d.β, d.μ, d.σ]
    # end |> stack
    # dists = dds .- mps
    # idx = findmin(norm.(eachcol(dists)))[2]
    # ps = dds[:, idx]

    # ax = Axis(f[1, 3]; title = "Input distribution", xlabel = "Input (xxx)",
    #           ylabel = "Density", xscale = log10, yscale = log10)
    # bins = 0.1:0.1:5
    # is = input[:, idx] # Sample neuron
    # ziggurat!(ax, is; bins, normalization = :pdf,
    #           color = baikal)
    # S = Stable(ps...)
    # lines!(ax, bins, pdf.(S, bins); color = bermejo, linestyle = :dash)

    sf = TwoPanel()
    begin # * Add input fits to secondary figure
        v = :input

        s = spectrum_fit[v].s
        _s = spectrum_fit[v]._s
        fitted_s = spectrum_fit[v].fitted_s
        m = spectrum_fit[v].m
        ax = Axis(
            sf[1, 2]; xscale = log10, yscale = log10, title = "Input PSD",
            xlabel = "Frequency (Hz)",
            limits = ((1, 1000), nothing),
            yticks = WilkinsonTicks(3; k_max = 4) |> LogTicks
        )
        lines!(ax, decompose(s)...; color = baikal)
        # scatter!(ax, _s; color = baikal, markersize = 10)
        lines!(ax, fitted_s .* 0.7; color = bermejo, linestyle = :dash)
        text = m.params.components.β |> last
        text = "b = $(round(text, digits = 2))"
        text!(
            ax, 0.1, 0.1; text, fontsize = 16, space = :relative,
            align = (:left, :bottom)
        )

        s = mad_fit[v].s
        # _s = mad_fit[v]._s
        fitted_s = mad_fit[v].fitted_s
        m = mad_fit[v].m
        ax = Axis(
            sf[1, 1]; xscale = log10, yscale = log10, title = "Input MAD",
            xlabel = "Time lag (s)"
        )
        lines!(ax, s; color = baikal)
        # scatter!(ax, _s; color = baikal, markersize = 10)
        lines!(ax, fitted_s; color = bermejo, linestyle = :dash)
        text = m.params.components.β |> first
        text = "a = $(round(text, digits = 2))"
        text!(
            ax, 0.1, 0.1; text, fontsize = 16, space = :relative,
            align = (:left, :bottom)
        )
    end

    begin # * Fano plot
        ax = Axis(
            sf[1, 3]; xlabel = "Window size (s)",
            title = "Fano factor", xscale = log10, yscale = log10,
            yticks = WilkinsonTicks(3; k_max = 4) |> LogTicks
        )

        sfano = deepcopy(fano)
        sfano = set(sfano, 𝑡 => times(sfano) ./ 1000) #uconvert(u"s", times(sfano))

        muf = nansafe(median)(sfano, dims = 2) |> ustripall
        # muf = dropdims(muf, dims = Neuron)
        s = nansafe(std)(sfano, dims = 2) |> ustripall
        # s = dropdims(s, dims = Neuron)

        ma = fit(MAPPLE, muf; components = 3, peaks = 0)
        fit!(ma, muf)

        # * Plot each frequency break
        fstops = ma.params.components.log_f_stop |> collect .|> exp10
        vlines!(
            ax, fstops[1:(end - 1)]; color = :gray,
            linestyle = :dot
        )
        prepend!(fstops, 1 / 1000)
        fstops[end] = maximum(dims(sfano, 𝑡))
        fcenters = fstops[1:(end - 1)] .+ diff(fstops) ./ 2
        for (fcenter, β) in zip(fcenters, ma.params.components.β)
            mean_fano = muf[𝑡 = Near(fcenter)] .* 1.2
            text = "c = $(round(β, digits = 2))"
            text!(
                ax, fcenter .* 0.8, mean_fano; text,
                align = (:center, :bottom),
                fontsize = 12
            )
        end

        # m.params.transition_width = 0.0

        bandwidth!(ax, decompose(muf)...; bandwidth = collect(s), alpha = 0.4) # ! Bandwidth 1 sd wide
        lines!(ax, muf)
        # lines!.([ax], eachcol(fano)[1:500:end], linewidth=1, alpha=0.5, color=baikal)

        fitted_fano = predict(ma, muf)
        lines!(ax, fitted_fano; color = bermejo, linestyle = :dash)
    end

    begin # * Short trace
        axv1 = Axis(
            gg[1, 1]; title = "Membrane potential (mV)",
            yticks = WilkinsonTicks(3; k_max = 3), xlabel = "Time (s)"
        )
        hlines!(axv1, [-50]; color = bermejo)
        hlines!(axv1, [-70]; color = bermejo, linestyle = :dash)
        hlines!(axv1, [mn]; color = :gray, linestyle = :dash)   # mn: mean V over the full trace (loaded)
        y = V[input_ts, myna] |> ustripall
        ts = times(y) .- times(y)[1]
        lines!(axv1, ts, y, linewidth = 3)

        # nu: E firing rate over the full trace (loaded from the raw file)
        axislegend(
            axv1, [LineElement(color = :transparent, linestyle = nothing)],
            [L"\nu \approx %$(round(nu, digits=1)) \textrm{ Hz }"];
            position = :rb, framevisible = true, patchsize = (0.1, 0.1)
        )
    end
    begin # * Short trace
        vi = input

        axvi1 = Axis(
            gg[2, 1]; title = "Input current (nA)",
            xlabel = "Time (s)", yticks = WilkinsonTicks(3; k_max = 3),
            limits = (nothing, (-1, 3))
        )

        # hlines!(ax, [-50]; color = bermejo)
        # hlines!(ax, [-70]; color = bermejo, linestyle = :dash)
        # hlines!(ax, [mean(V)]; color = :gray, linestyle = :dash)
        y = vi[input_ts, myna] |> ustripall
        ts = times(y) .- times(y)[1]
        lines!(axvi1, ts, y, linewidth = 3)
    end
    begin # * Voltage distribution
        axv2 = Axis(
            gg[1, 2]; title = "Density", xticks = WilkinsonTicks(3; k_max = 3),
            xlabel = "V (mV)"
        )
        # hideydecorations!(axv2)
        # hidexdecorations!(axv2)

        # Precomputed membrane-potential density (over the full trace, saved in the stats file).
        # Drop the first bin: it is the Vr = -70 mV reset/refractory pile-up (the old plot's bins[2:end]).
        zigg!(axv2, V_hist; dropfirst = true)
        vlines!(axv2, [mn]; color = :gray, linestyle = :dash)
    end
    begin # * step size distribution
        axvi2 = Axis(
            gg[2, 2]; title = "Step sizes",
            xticks = LogTicks(WilkinsonTicks(3; k_max = 3)),
            yticks = LogTicks(WilkinsonTicks(3; k_max = 3)),
            yscale = log10, xscale = log10, xlabel = "|ΔI| (nA)"
        )
        # hideydecorations!(axvi2)
        # hidexdecorations!(axvi2)

        # Precomputed |ΔI| step-size density (over the full trace, saved in the stats file).
        zigg!(axvi2, dI_hist; logy = true)
        # hlines!(ax, [mean(V)]; color = :gray, linestyle = :dash)

        # rowsize!(mf.layout, 0, Relative(0.2))
    end

    # linkyaxes!(axv1, axv2)
    # linkyaxes!(axvi1, axvi2)

    colsize!(gg, 1, Relative(0.75))
    colsize!(g, 1, Relative(0.35))

    display(mf)
    wsave(plotsdir("critical_demo", "key_properties.png"), mf)
    wsave(plotsdir("critical_demo", "key_properties.pdf"), mf)
end

# # * Check against fooof
# function aperiodicfit(psd::PSDVector, freqrange = [1.0, 300.0]; max_n_peaks = 10,
#                       aperiodic_mode = "knee", peak_threshold = 0.5, mink = 0.01, kwargs...)
#     ffreqs = dims(psd, 𝑓) |> collect
#     freqrange = pylist([(freqrange[1]), (freqrange[2])])
#     spectrum = vec(collect(psd))
#     fm = PyFOOOF.FOOOF(; peak_width_limits = pylist([0.5, 50.0]), max_n_peaks,
#                        aperiodic_mode, peak_threshold, kwargs...)
#     fm.add_data(Py(ffreqs).to_numpy(), Py(spectrum).to_numpy(), freqrange)
#     fm.fit()
#     if aperiodic_mode == "fixed"
#         b, χ = [pyconvert(Float64, x) for x in fm.aperiodic_params_]
#         k = 0.0
#     else
#         b, k, χ = pyconvert.((Float64,), fm.aperiodic_params_)
#         k = max(k, mink)
#     end
#     # p = fm.plot(; plot_peaks = "shade", plt_log = true, file_name = "./ttttt.png",
#     # save_fig = true)
#     L = f -> 10.0 .^ (b - log10(k + (f)^χ))
#     return L, Dict(:b => b, :k => k, :χ => χ)
# end
