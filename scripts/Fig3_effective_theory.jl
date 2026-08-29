#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WorkingRegime"
using CairoMakie
using Fathom
using Rsvg # activates FathomRsvgExt, so `svgimage!` draws true vector graphics
using TimeseriesTools
using Statistics
using DelimitedFiles
import StatsBase

set_theme!(Fathom.fathom())

begin # * Options
    NAME = "Fig3_effective_theory"
    outdir = plotsdir(NAME)
    schematic = projectdir("mean_field_schematic.svg")
    datafile = projectdir("WRTheory", "data", "Fig3_effective_theory.jld2")
end

begin # * Load data (produced by WRTheory/scripts/Fig3_effective_theory_data.jl)
    data = wload(datafile)
    sol = data["sol"] # neuron (V, w), times in s
    input_sol = data["input_sol"]
    spikes = data["spikes"] # s
    mfanos = data["mfanos"] # Obs-mean Fano curves
    rates_αβ = data["rates_αβ"] # Hz, (α × β)
    mcs = data["mcs"] # variability exponent, (α × β)
    ps = data["params"]
    α, β, η, γ = ps.α, ps.β, ps.η, ps.γ # working-regime point

    fr = length(spikes) / duration(sol) # Hz
end

begin # * Render
    f = Figure(size = (720, 560))
    gtop = f[1, 1] = GridLayout()
    gbot = f[2, 1] = GridLayout()

    begin # * Panel a --- mean-field schematic
        ax = Axis(gtop[1, 1]; aspect = DataAspect(), valign = :top) # letterboxes in its tall cell; sit under the panel letter
        hidedecorations!(ax)
        hidespines!(ax)
        svgimage!(ax, schematic)
    end

    g_input = gtop[1, 2] = GridLayout()
    begin # * Panel b --- membrane potential
        ax = Axis(
            g_input[1, 1]; title = "Membrane potential", xlabel = "Time (s)",
            ylabel = "𝑉 (mV)", limits = (nothing, (-71, -49)),
            yticks = WilkinsonTicks(4; k_max = 5)
        )
        hlines!(ax, [-50]; color = bermejo)
        hlines!(ax, [-70]; color = bermejo, linestyle = :dash)
        hlines!(ax, [mean(sol[:, 1])]; color = :gray, linestyle = :dash)
        V_trace = sol[1:10000, 1]
        times(V_trace) .-= first(times(V_trace))
        lines!(ax, V_trace; linewidth = 2)
        axislegend(
            ax, [LineElement(color = :transparent, linestyle = nothing)],
            [L"\nu \approx %$(round(fr, digits = 1)) \textrm{ Hz }"];
            position = :rb, framevisible = true, patchsize = (0.1, 0.1)
        )
    end
    begin # * Panel c --- voltage density
        ax = Axis(
            g_input[1, 2]; title = "Density", titlealign = :right, # narrow panel: clear the letter
            xlabel = "𝑉 (mV)", ylabel = "PDF",
            yticks = WilkinsonTicks(2), xticks = WilkinsonTicks(3),
            limits = ((-70, -50), nothing)
        )
        density!(ax, sol[:, 1])
    end
    begin # * Panel d --- input current
        ax = Axis(
            g_input[2, 1]; title = "Input current", xlabel = "Time (s)",
            ylabel = "𝐼 (nA)", limits = (nothing, (-2, 5)),
            yticks = WilkinsonTicks(3; k_max = 4)
        )
        I_trace = input_sol[1:5000, 1]
        times(I_trace) .-= first(times(I_trace))
        lines!(ax, I_trace; linewidth = 2)
    end
    begin # * Panel e --- input step sizes
        step_bins = (0:0.1:3.1)[2:end]
        steps = abs.(diff(input_sol[1:10:end, 1]))
        ax = Axis(
            g_input[2, 2]; title = "Step sizes", titlealign = :right, # narrow panel: clear the letter
            xlabel = "|Δ𝐼| (nA)",
            ylabel = "Frequency", xscale = log10, yscale = log10,
            xticks = WilkinsonTicks(4; k_max = 4) |> LogTicks,
            yticks = WilkinsonTicks(4; k_max = 4) |> LogTicks
        )
        ziggurat!(ax, steps[:]; bins = step_bins, normalization = :pdf)
    end
    colsize!(g_input, 1, Relative(0.75))

    begin # * Panel f --- firing-rate phase diagram
        ax = Axis(
            gbot[1, 1]; title = "Firing rate (Hz)", xlabel = "α", ylabel = "β",
            backgroundcolor = :gray88
        )
        p = contourf!(
            ax, rates_αβ; colormap = seethrough(:turbo),
            nan_color = :lightgray, levels = 11
        )
        Colorbar(gbot[1, 2], p)
        Label(gbot[1, 2, Top()], L"\nu"; valign = :bottom, halign = :center)
    end
    begin # * Panel g --- Fano factor curves at the working regime
        fano_slice(_β, _γ) = mfanos[α = Near(α), β = Near(_β), η = At(η), γ = At(_γ)]
        fano_curves = [
            "γ = 0" => fano_slice(β, 0.0),
            "γ = $γ" => fano_slice(β, γ),
            "β = 1, γ = 0" => fano_slice(1.0, 0.0),
        ]
        ax = Axis(
            gbot[1, 3]; xscale = log10, yscale = log10, title = "Fano factor",
            xlabel = "Time bin (s)", limits = ((1.0e-3, 1.0e0), nothing),
            xticks = LogTicks(WilkinsonTicks(4))
        )
        for (label, curve) in fano_curves
            lines!(ax, curve; label)
        end
        axislegend(ax; position = :lt, patchsize = (10, 10))
    end
    begin # * Panel h --- variability exponent phase diagram
        colorrange = (0.0, 0.6)
        ax = Axis(
            gbot[1, 4]; title = "Variability\nexponent", xlabel = "α", ylabel = "β",
            backgroundcolor = :gray88, limits = ((1.2, 2.0), (0.2, 1.0))
        )
        p = contourf!(
            ax, mcs; colormap = darksunset, nan_color = :lightgray,
            levels = range(colorrange...; length = 11)
        )
        Colorbar(gbot[1, 5], p; ticks = WilkinsonTicks(4; k_max = 4, k_min = 3))
        Label(gbot[1, 5, Top()], L"c"; valign = :bottom, halign = :center)
    end

    colsize!(gtop, 1, Relative(0.35))
    rowsize!(f.layout, 1, Relative(0.6))
    addlabels!(
        [
            gtop[1, 1], g_input[1, 1], g_input[1, 2], g_input[2, 1], g_input[2, 2],
            gbot[1, 1], gbot[1, 3], gbot[1, 4],
        ], f; fontsize = 16
    )
    display(f)
end

begin # * Save figures
    wsave(joinpath(outdir, "$NAME.pdf"), f)
    wsave(joinpath(outdir, "$NAME.svg"), f)
    wsave(joinpath(outdir, "$NAME.png"), f)
    @info "Saved figure" outdir
end

begin # * Save source data
    """
    One tsv per panel of `$NAME`, reusing the objects handed to each plot call so the saved
    data is exactly what was drawn. Panel a is the static schematic and carries no data.
    """
    function save_source_data()
        mkpath(outdir)
        savedir(x...) = joinpath(outdir, x...)

        savetimeseries(savedir("panelB_potential.tsv"), V_trace) # b: windowed V trace
        writedlm(
            savedir("panelB_summary.tsv"), # b: reference lines + quoted firing rate
            [
                ["V_threshold" "V_reset" "V_mean" "firing_rate_Hz"];
                [-50.0 -70.0 mean(sol[:, 1]) fr]
            ], '\t'
        )
        writedlm(
            savedir("panelC_voltage.tsv"), # c: the V samples passed to `density!`
            vcat(["V"], collect(sol[:, 1])), '\t'
        )
        savetimeseries(savedir("panelD_input.tsv"), I_trace) # d: windowed input trace

        edges = collect(step_bins) # e: binned step-size density, as `ziggurat!` computes it
        w = StatsBase.normalize(
            StatsBase.fit(StatsBase.Histogram, steps[:], edges);
            mode = :pdf
        ).weights
        writedlm(
            savedir("panelE_stepsizes.tsv"),
            vcat(
                ["bin_lower" "bin_upper" "pdf"],
                hcat(edges[1:(end - 1)], edges[2:end], w)
            ), '\t'
        )

        # f, h: (α × β) grids with α down the rows and β across the columns
        function writegrid(path, X)
            return writedlm(
                path, vcat(
                    hcat("alpha\\beta", permutedims(collect(lookup(X, :β)))),
                    hcat(collect(lookup(X, :α)), parent(X))
                ), '\t'
            )
        end
        writegrid(savedir("panelF_rates.tsv"), rates_αβ)
        writegrid(savedir("panelH_exponents.tsv"), mcs)

        savetimeseries(
            savedir("panelG_fano.tsv"), # g: the three curves, shared time base
            Timeseries(
                hcat(collect.(last.(fano_curves))...),
                times(last(first(fano_curves))),
                Symbol.(first.(fano_curves))
            )
        )
        return @info "wrote source data" outdir
    end
    save_source_data()
end
