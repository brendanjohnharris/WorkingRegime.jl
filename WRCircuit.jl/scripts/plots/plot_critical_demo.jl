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
    x = load(datadir("critical_demo.jld2"), "x")
    fixed_params = load(datadir("critical_demo.jld2"), "fixed_params")
    epositions = load(datadir("critical_demo.jld2"), "epositions")
    ipositions = load(datadir("critical_demo.jld2"), "ipositions")
    dx = fixed_params.dx
    spikes = x[Population = At(:E), Var = At(:spike)]
    tmin = minimum(times(spikes)) .- step(spikes)
    tmax = maximum(times(spikes))
end

# begin # * Animate
#     @info "Animating rates"
#     rates = WRCircuit.compute_rates(spikes, 50u"ms")
#     WRCircuit.animate_rates(rates, dx; filename = "critical_demo.mp4")
# end

begin
    spike_times = map(eachslice(spikes, dims = Neuron)) do s
        sts = times(s)[findall(s)]
    end

    # epositions = m.E.positions
    # epositions = map(epositions) do pos
    #     map(pos) do p
    #         p.tolist() |> convert2(Float32)
    #     end
    # end
end
if :I ∈ lookup(x, Population)  # * Spike raster
    ispikes = x[Population = At(:I), Var = At(:spike)]
    ispike_times = map(eachslice(ispikes, dims = Neuron)) do s
        sts = times(s)[findall(s)]
    end

    radius = 0.15 # mm
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
    ax = Axis(f[1, 1]; ylabel = "Excitatory", yticks = [NaN])
    hidexdecorations!(ax)

    # intrvl = 5000u"ms" .. 5200u"ms"
    intrvl = 9000u"ms" .. 9500u"ms"

    for (i, s) in enumerate(spike_times[elocal_idxs])
        idxs = s .∈ [intrvl]
        scatter!(
            ax, ustripall(s[idxs] .- minimum(intrvl)), i * ones(sum(idxs)),
            color = qinghai,
            markersize = 3
        )
    end

    ax2 = Axis(f[2, 1]; xlabel = "Time (ms)", ylabel = "Inhibitory", yticks = [NaN])
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
    save(plotdir("spike_example.pdf"), f)
end

begin # * Fano factor
    @info "Calculating Fano factor"
    dt = step(spikes)
    τs = logrange(dt * 10 |> ustrip, dt * 1000 |> ustrip, length = 200) # ms
    fano = fano_factor(ustripall(spikes), τs)
    # fano = fano[𝑡 = 1..1000]

    mfano = map(Chart(ProgressLogger(), Threaded()), eachcol(fano)) do x
        ma = fit(MAPPLE, x; components = 3, peaks = 0)
        fit!(ma, x)
        return ma.params.components.β |> maximum
    end
    open(plotdir("critical_demo", "fano_statistics.txt"), "w") do f
        stat = TimeseriesTools.bootstrapmedian(mfano)
        write(f, "$(stat)\n")
    end
end

# begin # * Membrane potential
#     V = x[Population = At(:E), Var = At(:V)][1:10:end, :]
#     V = set(V, 𝑡 => uconvert.(u"s", times(V)))
#     V = rectify(V, dims = 𝑡)
#     lines(V[1:2000, 970]) |> display
# end
# begin # * Mean each V, zero out the 10ms around each spike
#     V = x[Population = At(:E), Var = At(:V)]
#     V = deepcopy(V[1:10:end, :])
#     V = set(V, 𝑡 => uconvert.(u"s", times(V)))
#     V = rectify(V, dims = 𝑡)
#     V̂ = V .- mean(V, dims = 𝑡)
#     map(eachslice(V̂, dims = Neuron), eachslice(spikes, dims = Neuron)) do v, s
#         sidxs = times(s[s])
#         ints = map(sidxs) do t
#             (t - 10u"ms") .. (t + 10u"ms")
#         end
#         for int in ints
#             v[𝑡 = int] .= 0.0
#         end
#     end
# end
# begin # * Average unit spectrum
#     _s = spectrum(V̂)
#     s = median(_s, dims = Neuron)
#     s = ustripall(dropdims(s, dims = Neuron))[𝑓 = 2 .. 100]
#     # s = _s[2:end, 2000] |> ustripall

#     ls = WRCircuit.log10spectrum(s)
#     params = fit_oneoneff(ls; n_peaks = 2, w = 10)
#     params = fit_oneoneff(ls, params)

#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Log frequency", ylabel
#               = "Log power", title = "MUA spectrum with fit")
#     lines!(ax, ls; color = :blue)
#     lines!(ax, lookup(ls, 1), oneoneff(lookup(ls, 1), params); color = bermejo)
#     display(f)
# end
# begin # * Mean membrane potential in a local patch
#     idxs = [1:10, 1:10]
#     N = lookup(V, Neuron) |> length |> sqrt |> Int
#     localV = reshape(V, (size(V, 1), N, N))
#     localV = localV[:, idxs...]
#     localV = mean(localV, dims = (2, 3))
#     localV = ToolsArray(vec(localV), dims(V, 𝑡))
#     lines(localV[1:5000]) |> display
#     hist(localV, bins = 200) |> display
# end

# begin # * Spectrum of local membrane potential mean
#     V = x[Population = At(:E), Var = At(:V)]
#     V = set(V, 𝑡 => convert2(u"s", times(V)))
#     N = lookup(V, Neuron) |> length |> sqrt |> Int
#     _V = reshape(V, (size(V, 1), N, N))

#     block_size = 10
#     num_row_blocks = div(size(_V, 2), block_size)
#     num_col_blocks = div(size(_V, 3), block_size)

#     idxs = [((i * block_size + 1):((i + 1) * block_size),
#              (j * block_size + 1):((j + 1) * block_size))
#             for i in 0:(num_row_blocks - 1), j in 0:(num_col_blocks - 1)]

#     LFP = map(idxs) do (i, j)
#         m = _V[:, i, j] # * Get local patch
#         m = mean(m, dims = (2, 3))
#         m = ToolsArray(vec(m), dims(V, 𝑡))
#     end
#     LFP = ToolsArray(LFP[:], Obs(1:length(LFP))) |> stack
#     sLFP = spectrum(LFP .- mean(LFP, dims = 𝑡), 0.5u"s")

#     s = mean(sLFP, dims = Obs) |> ustripall
#     s = dropdims(s, dims = Obs)[𝑓 = 0.5 .. 10000]
#     lines(s; axis = (; xscale = log10, yscale = log10)) |>
#     display
# end

# begin # * Mean spectrum fit
#     ls = logsample(s[3:end])
#     params = fit_oneoneff(ls; n_peaks = 2, w = 2)
#     params = fit_oneoneff(ls, params)

#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Frequency (Hz)", ylabel
#               = "Power", title = "MUA spectrum with fit")
#     lines!(ax, ls; color = :blue)
#     lines!(ax, lookup(ls, 1), oneoneff(lookup(ls, 1), params); color = bermejo),
#     display(f)
# end

# begin # * MUA spectrum
#     mdt = 2.0u"ms"
#     mua = groupby(spikes, 𝑡 => Base.Fix2(WRCircuit.group_dt, mdt))
#     mua = map(mua) do r
#         dropdims(sum(r, dims = 𝑡), dims = 𝑡) ./ uconvert(u"s", mdt)
#     end |> stack
#     mua = permutedims(mua, (𝑡, Neuron))
#     mua = rectify(mua, dims = 𝑡)
#     N = lookup(mua, Neuron) |> length |> sqrt |> Int
#     ts = dims(mua, 𝑡)
#     mua = reshape(mua, (size(mua, 1), N, N))

#     # Calculate number of complete blocks that fit in each dimension
#     block_size = 10
#     num_row_blocks = div(size(mua, 2), block_size)
#     num_col_blocks = div(size(mua, 3), block_size)

#     idxs = [((i * block_size + 1):((i + 1) * block_size),
#              (j * block_size + 1):((j + 1) * block_size))
#             for i in 0:(num_row_blocks - 1), j in 0:(num_col_blocks - 1)]

#     muas = map(idxs) do (i, j)
#         m = mua[:, i, j] # * Get local patch
#         m = mean(m, dims = (2, 3))
#         m = ToolsArray(vec(m), ts)
#         m = set(m, 𝑡 => uconvert.(u"s", times(m)))
#         m = rectify(m, dims = 𝑡)
#         return spectrum(m)
#     end

#     muas = mean(muas)
#     lines(ustripall(muas)[𝑓 = 1 .. 200]; axis = (; xscale = log10, yscale = log10)) |>
#     display
# end
# begin # * Plot inter-spike interval distributions
#     isis = map(spike_times) do sts
#         diff(sts)
#     end
#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Inter-spike interval (ms)",
#               ylabel = "Density", limits = ((0, 100), nothing))
#     hist!(ax, Iterators.flatten(isis) |> collect |> ustrip, normalization = :pdf,
#           bins = range(0.0, 1000, step = 5))
#     display(f)
# end

# begin # * Input trace
#     input = x[Population = At(:E), Var = At(:input)][:, 900]
#     input = set(input, 𝑡 => uconvert.(u"s", times(input)))
#     input = rectify(input, dims = 𝑡)
#     lines(input[1:9000]) |> display
# end
# begin # * Input distribution
#     input = x[Population = At(:E), Var = At(:input)]
#     input = log10.(input[input .> 0.1])
#     hist(input[:], bins = 100, axis = (; yscale = log10))
# end
# begin # * Input spectrum
#     input = x[Population = At(:E), Var = At(:input)]
#     input = set(input, 𝑡 => uconvert.(u"s", times(input)))
#     input = rectify(input, dims = 𝑡)
#     _s = spectrum(input .- mean(input, dims = 𝑡), 0.5)
#     s = mean(_s, dims = Neuron)
#     s = ustripall(dropdims(s, dims = Neuron))[𝑓 = 0.1 .. 200]
#     lines(s; axis = (; xscale = log10, yscale = log10)) |> display
# end
# begin # * MSD of inputs
#     input = x[Population = At(:E), Var = At(:input)]
#     msd = msdist(input, 1:100)
#     msd = mean(msd, dims = Neuron) |> ustripall
#     msd = dropdims(msd, dims = Neuron)
#     lines(msd, axis = (; xscale = log10, yscale = log10)) |> display
# end

# begin # * Increments of voltage
#     dV = diff(V, dims = 𝑡)
#     dV[dV .< 10] .= NaN
#     lines(dV[1:1000, 1])
# end

begin # * Plot the MSD and power spectrum, with fits, of the LFP, membrane potential, and input traces
    begin # * Membrane potential
        V = x[Population = At(:E), Var = At(:V)]
        V = set(V, 𝑡 => convert2(u"s", times(V)))
    end
    begin # * LFP
        N = lookup(V, Neuron) |> length |> sqrt |> Int
        _V = reshape(V, (size(V, 1), N, N))

        block_size = 10
        num_row_blocks = div(size(_V, 2), block_size)
        num_col_blocks = div(size(_V, 3), block_size)

        idxs = [
            (
                    (i * block_size + 1):((i + 1) * block_size),
                    (j * block_size + 1):((j + 1) * block_size),
                )
                for i in 0:(num_row_blocks - 1), j in 0:(num_col_blocks - 1)
        ]

        LFP = map(idxs) do (i, j)
            m = _V[:, i, j] # * Get local patch
            m = mean(m, dims = (2, 3))
            m = ToolsArray(vec(m), dims(V, 𝑡))
        end
        LFP = ToolsArray(LFP[:], Obs(1:length(LFP))) |> stack
    end
    begin # * Inputs
        input = x[Population = At(:E), Var = At(:input)]
        input = set(input, 𝑡 => convert2(u"s", times(V)))
    end
end

begin # * Fit distribution
    ds = map(Chart(Threaded()), eachslice(input[1:10:end, :], dims = Neuron)) do v
        fit(Stable, v)
    end
    αs = getfield.(ds, :α)
    βs = getfield.(ds, :β)
    μs = getfield.(ds, :μ)
    σs = getfield.(ds, :σ)
end

# * Fit mean spectrum/mad
function fit_spectrum(s; components, peaks, f_range)
    negdims = [i for i in 1:ndims(s) if i != dimnum(s, 𝑓)] |> Tuple
    original_s = deepcopy(s)
    original_s = ustripall(original_s)
    original_s = median(original_s, dims = negdims)
    original_s = dropdims(original_s, dims = negdims)

    s = s[𝑓 = f_range] |> ustripall
    s = median(s, dims = negdims)
    s = dropdims(s, dims = negdims)
    _s = logsample(s)
    m = fit(MAPPLE, _s; components, peaks)
    fit!(m, _s)
    fitted_s = predict(m, s)
    return (; m, s = original_s, fitted_s, _s)
end
function fit_mad(s; components, peaks, tau_range)
    negdims = [i for i in 1:ndims(s) if i != dimnum(s, 𝑡)] |> Tuple
    s = s[𝑡 = tau_range] |> ustripall
    s = median(s, dims = negdims)
    s = dropdims(s, dims = negdims)
    # _s = logsample(s)
    m = fit(MAPPLE, s; components, peaks)
    fit!(m, s)
    fitted_s = predict(m, s)
    return (; m, s, fitted_s)
end

# * Fit each spectrum individually
function fit_spectrums(s::AbstractVector; components, peaks, f_range)
    s = s[𝑓 = f_range] |> ustripall
    # s = mean(s, dims = negdims)
    # s = dropdims(s, dims = negdims)
    _s = logsample(s)
    m = fit(MAPPLE, _s; components, peaks)
    fit!(m, _s)
    fitted_s = predict(m, s)
    return (; m, s, fitted_s, _s)
end
function fit_spectrums(s::AbstractMatrix; kwargs...)
    return map(eachcol(s)) do v
        fit_spectrums(v; kwargs...)
    end
end
function fit_mads(s::AbstractVector; components, peaks, tau_range)
    s = s[𝑡 = tau_range] |> ustripall
    # s = mean(s, dims = negdims)
    # s = dropdims(s, dims = negdims)
    # _s = logsample(s)
    m = fit(MAPPLE, s; components, peaks)
    fit!(m, s)
    fitted_s = predict(m, s)
    return (; m, s, fitted_s)
end
function fit_mads(s::AbstractMatrix; kwargs...)
    return map(eachcol(s)) do v
        try
            fit_mads(v; kwargs...)
        catch
            return NaN
        end
    end
end

begin # * Calculate spectra and MAD
    vars = (; V = V[:, 1:10:end], LFP = LFP[:, 1:10:end], input = input[:, 1:10:end])

    @info "Calculating spectra"
    spectra = map(Chart(Threaded()), vars) do v
        spectrum(v .- mean(v, dims = 𝑡), 1.0u"Hz", padding = 5000)
    end
    @info "Calculating MADs"
    mads = map(Chart(Threaded()), vars) do v
        madev(v, round.(Int, logrange(10, 10000, length = 100) |> unique) .* step(v))
    end
end

begin # * Fits
    f_range = 10u"Hz" .. 1000u"Hz"
    tau_range = 0u"s" .. 1u"s"
    @info "Fitting spectra"
    spectrum_fit = map(Chart(Threaded(), ProgressLogger()), spectra) do s
        fit_spectrum(s; components = 1, peaks = 0, f_range)
    end
    spectrum_fits = map(Chart(Threaded(), ProgressLogger()), spectra) do s
        fit_spectrums(s; components = 1, peaks = 0, f_range)
    end
    @info "Fitting MADs"
    mad_fit = map(Chart(Threaded(), ProgressLogger()), mads) do m
        fit_mad(m; components = 2, peaks = 0, tau_range)
    end
    mad_fits = map(Chart(Threaded(), ProgressLogger()), mads) do m
        fit_mads(m; components = 2, peaks = 0, tau_range)
    end
end

if false
    f = SixPanel()
    gs = permutedims(subdivide(f, 3, 2), (2, 1))

    axs = map(enumerate(keys(vars))) do (i, v)
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

    axs = map(enumerate(keys(vars))) do (i, v)
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
    axs = map(fs, keys(vars)) do f, v
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
        wsave(plotdir("critical_demo", "$(v)_spectrum.pdf"), f)
    end

    # * MAD
    fs = [OnePanel() for _ in 1:3]
    axs = map(fs, keys(vars)) do f, v
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
        wsave(plotdir("critical_demo", "$(v)_mad.pdf"), f)
    end
end

begin # * Statistics
    open(plotdir("critical_demo", "statistics.txt"), "w") do f
        for v in keys(vars)
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

    # Fano factor (median across neurons)
    fano_median = median(fano, dims = 2)
    fano_median = dropdims(fano_median, dims = 2)
    fano_exponent = median(mfano)
    fano_exponents = collect(mfano)

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
            exponents = fano_exponents,
        ),
    )

    mkpath(datadir("plots"))
    jldsave(datadir("plots", "circuit_curves.jld2"); circuit_curves)
    @info "Saved circuit curves to $(datadir("plots", "circuit_curves.jld2"))"
end

begin # * Supplementary figure: distribution of input distribution parameters
    sf = FourPanel()
    gs = subdivide(sf, 2, 2)
    map(enumerate([(:α, αs), (:β, βs), (:μ, μs), (:σ, σs)])) do (i, (name, data))
        m = median(data)
        ax = Axis(
            gs[i]; title = "$(name): median=$(round(m, digits = 2))",
            xlabel = string(name),
            ylabel = "Density"
        )
        ziggurat!(
            ax, data; bins = 20, normalization = :pdf,
            color = baikal
        )
        vlines!(ax, [m]; color = bermejo, linestyle = :dash)
    end
    display(sf)
    wsave(plotdir("critical_demo", "input_distribution_parameters.pdf"), sf)
end

begin # * Additional properties: image and distribution fit
    mf = TwoPanel(; size = (720, 324))
    myna = 27
    t = 18.3 * 10000 |> Int # Samples
    deltat = 0.117 * 10000 |> round |> Int # Samples
    shift = (-10, 5)
    input_ts = 13500:18500 # 8000:13000

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

    input_grid = circshift(input_grid, (0, shift...)) # Avoid wraparounds

    xs, ys = track_com(input_grid[(t - deltat):t, :, :])
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
        ax, xx, xx, input_grid[t, :, :]';
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
        mn = mean(V[1:50:end, :])
        hlines!(axv1, [mn]; color = :gray, linestyle = :dash)
        y = V[input_ts, myna] |> ustripall
        ts = times(y) .- times(y)[1]
        lines!(axv1, ts, y, linewidth = 3)

        nu = sum(spikes) ./ size(spikes, 2) ./ uconvert(u"s", duration(spikes)) |> ustrip

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

        v = V[1:10:end]
        bins = -70:0.1:-50
        bins = bins[2:end]
        ziggurat!(
            axv2, v; bins, normalization = :pdf,
            color = baikal
        )
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

        vi = abs.(diff(input, dims = 1))

        bins = 0:0.1:4
        bins = bins[2:end]
        ziggurat!(
            axvi2, vi[1:10:end]; bins, normalization = :pdf,
            color = baikal
        )
        # hlines!(ax, [mean(V)]; color = :gray, linestyle = :dash)

        # rowsize!(mf.layout, 0, Relative(0.2))
    end

    # linkyaxes!(axv1, axv2)
    # linkyaxes!(axvi1, axvi2)

    colsize!(gg, 1, Relative(0.75))
    colsize!(g, 1, Relative(0.35))

    display(mf)
    wsave(plotdir("critical_demo", "key_properties.pdf"), mf)
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
