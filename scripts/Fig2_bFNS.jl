#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WorkingRegime"
using CairoMakie
using Fathom
using TimeseriesTools
using TimeseriesMakie
using DelimitedFiles
import StatsBase
import Fathom: crimson, cornflowerblue, cucumber, california # Foresight-era names, unexported by Fathom

set_theme!(Fathom.fathom())

begin # * Options
    inset_log = false  # true: log-log as inset over linear plot; false: log-log as full axis
    order_rows = true  # true: rows = {space, time}; false: columns = {space, time}
    NAME = "Fig2_bFNS"
    outdir = plotsdir(NAME)
    datafile = projectdir("WRTheory", "data", "bFNS_data.jld2")
    nyticks = 4
end

begin # * Load data (produced by WRTheory/scripts/bFNS_data.jl)
    data = wload(datafile)
    xs, Vs, Ṽs = data["xs"], data["Vs"], data["Ṽs"]
    xs_pdf, p1, p2 = data["xs_pdf"], data["p1"], data["p2"]
    xs_a1, xs_a2 = data["xs_a1"], data["xs_a2"]
    ts, R1, R2 = data["ts"], data["R1"], data["R2"]
    s_hi, s_lo = data["s_hi"], data["s_lo"]
    ts_windows, configs = data["ts_windows"], data["configs"]
    τs, mads, gmads = data["τs"], data["mads"], data["gmads"]
    τfit, mad_fit, a_exponent = data["τfit"], data["mad_fit"], data["a_exponent"]
    psd, gpsd = data["psd"], data["gpsd"]
    psd_fit_x, psd_fit_y = data["psd_fit_x"], data["psd_fit_y"]
    b_exponent = data["b_exponent"]
    ma, ms = data["ma"], data["ms"]
    prms = data["params"]
    α1, α2, β1, β2, β_hi, β_lo = prms.α1, prms.α2, prms.β1, prms.β2, prms.β_hi, prms.β_lo
end

begin # * Set up figure
    f = SixPanel()
    g_main = f[1:2, 1:2] = GridLayout()
    gs = [g_main[1, 1], g_main[1, 2], g_main[2, 1], g_main[2, 2]] # row-major
    # Panel indices: potential, step-size, relaxation, spectrum
    if order_rows
        # Row 1 = space (potential, step-size), Row 2 = time (relaxation, spectrum)
        gi_potential, gi_stepsize, gi_relax, gi_spectrum = 1, 2, 3, 4
    else
        # Col 1 = space (potential, step-size), Col 2 = time (relaxation, spectrum)
        gi_potential, gi_stepsize, gi_relax, gi_spectrum = 1, 3, 2, 4
    end
end

begin # * Panel 1 — Effective potential for unimodal density
    ax = Axis(
        gs[gi_potential], xlabel = "x", ylabel = "V(x)",
        limits = (extrema(xs), (-0.5, maximum(Vs))), title = "Potential function",
        yticks = WilkinsonTicks(nyticks)
    )
    lines!(ax, xs, Vs; color = :cornflowerblue, label = "Potential")
    lines!(
        ax, xs, Ṽs; color = :crimson,
        label = "Effective potential"
    )
    axislegend(ax; position = :ct, title = "α = $(prms.α)")
end

begin # * Panel 2 — Step-size distribution: α = 2 (Gaussian) vs α = 1.5 (heavy-tailed)
    if inset_log
        ax = Axis(
            gs[gi_stepsize], xlabel = "|x|", ylabel = "p(|x|)",
            title = "Step-size distribution",
            limits = ((0, 6), (0, nothing)),
            yticks = WilkinsonTicks(nyticks)
        )
        lines!(ax, xs_pdf, p1; color = :cornflowerblue, label = "α = $(α1)")
        lines!(ax, xs_pdf, p2; color = :crimson, label = "α = $(α2)")
        axislegend(ax; position = :lb)

        ax_inset = Axis( # log-log view of the pdf
            gs[gi_stepsize]; width = Relative(0.5), height = Relative(0.5),
            halign = 0.95, valign = 0.9,
            xscale = log10, yscale = log10,
            xlabelsize = 10, ylabelsize = 10,
            xticklabelsize = 8, yticklabelsize = 8,
            backgroundcolor = :white,
            limits = ((nothing, 25), (1.0e-4, 1.0e0)),
            yticks = LogTicks(WilkinsonTicks(nyticks - 1))
        )
        lines!(ax_inset, xs_pdf, p1; color = :cornflowerblue)
        lines!(ax_inset, xs_pdf, p2; color = :crimson)
        translate!(ax_inset.blockscene, 0, 0, 100)
    else
        bins = logrange(1, 40, length = 20)
        ax = Axis(
            gs[gi_stepsize]; xlabel = "|x|", ylabel = "p(|x|)",
            title = "Step-size distribution",
            xscale = log10, yscale = log10,
            limits = ((1, 25), (1.0e-4, nothing)),
            yticks = LogTicks(WilkinsonTicks(nyticks))
        )
        ziggurat!(
            ax, xs_a2; strokecolor = :crimson, label = "α = $(α2)", bins,
            normalization = :pdf, fillalpha = 0.3, color = brighten(crimson, 0.5)
        )
        ziggurat!(
            ax, xs_a1; strokecolor = :cornflowerblue, label = "α = $(α1)", bins,
            normalization = :pdf, fillalpha = 0.3,
            color = brighten(cornflowerblue, 0.5)
        )
        axislegend(ax; position = :rt)
    end
end

begin # * Panel 3 — Relaxation function: β = 1 (exponential) vs β < 1 (Mittag-Leffler)
    ts_log = ts[2:end] # skip t = 0 for the log axes
    R1_log = R1[2:end]
    R2_log = R2[2:end]

    if inset_log
        ax = Axis(
            gs[gi_relax], xlabel = "Time", ylabel = "R(t)",
            title = "Relaxation function", limits = ((0, 5), (0, 1.05))
        )
        lines!(ax, ts, R1; color = :cornflowerblue, label = "β = $(β1)")
        lines!(
            ax, ts, R2; color = :crimson,
            label = "β = $(β2)"
        )
        hlines!(ax, [0]; color = :gray, linewidth = 0.5)
        axislegend(ax; position = :lb)

        ax_inset = Axis( # log-log view
            gs[gi_relax]; width = Relative(0.5), height = Relative(0.5),
            halign = 0.95, valign = 0.9,
            xscale = log10, yscale = log10,
            xlabelsize = 10, ylabelsize = 10,
            xticklabelsize = 8, yticklabelsize = 8,
            backgroundcolor = :white,
            limits = ((minimum(ts_log), maximum(ts_log)), (1.0e-2, 1.0e0 + 0.5)),
            yticks = LogTicks(WilkinsonTicks(nyticks))
        )
        lines!(ax_inset, ts_log, R1_log; color = :cornflowerblue)
        lines!(ax_inset, ts_log, abs.(R2_log); color = :crimson)
        translate!(ax_inset.blockscene, 0, 0, 100)
    else
        ax = Axis(
            gs[gi_relax]; xlabel = "t", ylabel = "R(t)",
            title = "Relaxation function",
            xscale = log10, yscale = log10,
            limits = ((minimum(ts_log), maximum(ts_log)), (1.0e-2, 1.0e0 + 0.5)),
            yticks = LogTicks(WilkinsonTicks(nyticks))
        )
        lines!(ax, ts_log, R1_log; color = :cornflowerblue, label = "β = $(β1)")
        lines!(ax, ts_log, abs.(R2_log); color = :crimson, label = "β = $(β2)")
        axislegend(ax; position = :lb)
    end
end

begin # * Panel 4 — Unconfined power spectra: β = 1.0 vs β = 0.5
    ax = Axis(
        gs[gi_spectrum]; xlabel = "Frequency (Hz)", ylabel = "Power",
        title = "Unconfined spectrum", yticks = LogTicks(WilkinsonTicks(nyticks))
    )
    plotspectrum!(ax, s_hi; label = "β = $(β_hi)", color = :cornflowerblue)
    plotspectrum!(ax, s_lo; label = "β = $(β_lo)", color = :crimson)
    ax.limits = ((1.0e-1, 1.0e0), (nothing, nothing))
    axislegend(ax; position = :lb)
end

begin # * Right column — sample time series: effect of α, β, γ
    ts_colors = [:black, cornflowerblue, crimson, california]
    for (i, (s, c)) in enumerate(zip(ts_windows, configs))
        ax = Axis(
            f[1:2, 3][i, 1];
            title = c.label,
            titlesize = 11,
            titlealign = :right
        )
        lines!(ax, times(s), collect(s); color = ts_colors[i], linewidth = 2)
        hidedecorations!(ax)
        hidespines!(ax)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Row 3 — model summary: scaling of the flat vs unimodal samplers
# ──────────────────────────────────────────────────────────────────────────────
begin # * Panel e --- mean absolute deviation
    g_sum = f[3, 1:3] = GridLayout()
    ax = Axis(
        g_sum[1, 1]; xlabel = "Time lag (s)", ylabel = "MAD",
        xscale = log10, yscale = log10, title = "Anomalous\ndiffusion", # two lines: narrow panel
        limits = ((1.0e-4, 1.0), (0.02, 3))
    )
    lines!(ax, τs, mads; label = "Unconfined")
    lines!(ax, τs, gmads; color = cucumber, label = "Unimodal")
    lines!(
        ax, τfit, mad_fit;
        label = "a = $(round(a_exponent, digits = 2))", color = :crimson,
        linestyle = :dash
    )
    l = axislegend(ax; position = :lt)
    reverselegend!(l)
end

begin # * Panel f --- power spectral density
    ax = Axis(
        g_sum[1, 2]; title = "Power\nspectrum",
        xtickformat = x -> string.(round.(Int, x))
    )
    plotspectrum!(ax, psd; label = "Unconfined")
    plotspectrum!(ax, gpsd; color = cucumber, label = "Unimodal")
    lines!(
        ax, psd_fit_x, psd_fit_y;
        color = :red, linewidth = 2, label = "b = $(round(b_exponent, digits = 2))",
        linestyle = :dash
    )
    l = axislegend(ax; position = :lb)
    reverselegend!(l)
    ax.xlabel = "Frequency (Hz)"
    ax.limits = ((1, 1000), (2.0e-6, 1.0e-1))
end

begin # * Panels g, h --- exponent maps over (α, β) for the flat sampler
    function exponent_map!(gp, X; levels, colormap, title, clabel)
        ax = Axis(
            gp[1, 1]; xlabel = "α", ylabel = "β", title,
            xgridvisible = false, ygridvisible = false, backgroundcolor = :gray88,
            limits = ((1.2, 2.0), (0.2, 1.0)), xticks = [1.2, 1.6, 2.0]
        )
        p = contourf!(ax, X; levels, colormap, extendhigh = :auto, extendlow = :auto)
        contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash) # a = 1/2 boundary
        scatter!( # the working-regime point shared by the whole figure
            ax, [prms.α], [prms.β]; color = cucumber,
            markersize = 10, strokecolor = :white, strokewidth = 1
        )
        Colorbar(gp[1, 2], p; width = 8, ticks = WilkinsonTicks(3))
        Label(gp[1, 2, Top()], clabel, font = :regular)
        return ax
    end
    exponent_map!(
        g_sum[1, 3], ma; levels = range(0.25, 0.75, length = 10),
        colormap = darksunset, title = "Diffusion\nexponent", clabel = "a"
    )
    exponent_map!(
        g_sum[1, 4], ms; levels = range(-2.0, -1.0, length = 10),
        colormap = lightsunset, title = "Spectral\nexponent", clabel = "b"
    )
end

begin # * Save figure
    colsize!(f.layout, 3, Relative(0.2))
    addlabels!(
        [
            gs[1], gs[2], gs[3], gs[4],
            g_sum[1, 1], g_sum[1, 2], g_sum[1, 3], g_sum[1, 4],
        ], f
    )
    wsave(joinpath(outdir, "$NAME.pdf"), f)
    wsave(joinpath(outdir, "$NAME.svg"), f)
    wsave(joinpath(outdir, "$NAME.png"), f)
    f |> display
end

begin # * Save source data
    """
    Write one tsv per panel of `$NAME` into the figure's own directory, reusing the
    arrays handed to each plot call so the saved data is exactly what was drawn.
    """
    function save_source_data()
        mkpath(outdir)
        savedir(x) = joinpath(outdir, x)

        writedlm( # a: potential and effective potential over the plotted grid
            savedir("panelA_potential.tsv"),
            vcat(["x" "V" "V_effective"], hcat(collect(xs), Vs, Ṽs)), '\t'
        )

        if inset_log # b: analytic folded stable densities
            writedlm(
                savedir("panelB_stepsize.tsv"),
                vcat(
                    ["abs_x" "alpha_$α1" "alpha_$α2"],
                    hcat(collect(xs_pdf), p1, p2)
                ), '\t'
            )
        else # b: binned densities, as `ziggurat!` computes them
            edges = collect(bins)
            zig(x) = StatsBase.normalize(
                StatsBase.fit(StatsBase.Histogram, x, edges); mode = :pdf
            ).weights
            writedlm(
                savedir("panelB_stepsize.tsv"),
                vcat(
                    ["bin_lower" "bin_upper" "alpha_$α1" "alpha_$α2"],
                    hcat(edges[1:(end - 1)], edges[2:end], zig(xs_a1), zig(xs_a2))
                ), '\t'
            )
        end

        writedlm( # c: relaxation functions (t = 0 dropped for the log axis)
            savedir("panelC_relaxation.tsv"),
            vcat(
                ["t" "beta_$β1" "beta_$β2"],
                hcat(collect(ts_log), R1_log, abs.(R2_log))
            ), '\t'
        )

        @assert freqs(s_hi) == freqs(s_lo)
        writedlm( # d: peak-normalised spectra over the plotted band
            savedir("panelD_spectrum.tsv"),
            vcat(
                ["frequency" "beta_$β_hi" "beta_$β_lo"],
                hcat(collect(freqs(s_hi)), collect(s_hi), collect(s_lo))
            ), '\t'
        )

        writedlm( # e: MAD curves over the plotted lags
            savedir("panelE_mad.tsv"),
            vcat(
                ["tau" "unconfined" "unimodal"],
                hcat(collect(τs), collect(mads), collect(gmads))
            ), '\t'
        )

        @assert freqs(psd) == freqs(gpsd)
        writedlm( # f: power spectral densities over the plotted band
            savedir("panelF_psd.tsv"),
            vcat(
                ["frequency" "unconfined" "unimodal"],
                hcat(collect(freqs(psd)), collect(psd), collect(gpsd))
            ), '\t'
        )
        writedlm( # e, f: fitted scaling exponents quoted in the legends
            savedir("panelEF_exponents.tsv"),
            [["a" "b"]; [a_exponent b_exponent]], '\t'
        )

        function writegrid(path, X) # g, h: (α × β) grids, α down the rows
            X = permutedims(X, (:α, :β))
            return writedlm(
                path,
                vcat(
                    hcat("alpha\\beta", permutedims(collect(lookup(X, :β)))),
                    hcat(collect(lookup(X, :α)), parent(X))
                ), '\t'
            )
        end
        writegrid(savedir("panelG_diffusion_exponent.tsv"), ma)
        writegrid(savedir("panelH_spectral_exponent.tsv"), ms)

        vs = map(configs) do c # right column: the four windowed traces, shared time base
            Symbol("alpha$(c.α)_beta$(c.β)_gamma$(c.γ)")
        end
        savetimeseries(
            savedir("timeseries.tsv"),
            Timeseries(hcat(collect.(ts_windows)...), times(first(ts_windows)), vs)
        )
        return @info "wrote source data" outdir
    end
    save_source_data()
end
