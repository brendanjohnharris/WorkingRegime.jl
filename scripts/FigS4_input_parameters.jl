#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WorkingRegime"
using JLD2
using CairoMakie
using Fathom
using TimeseriesTools
using Statistics
using DelimitedFiles
import StatsBase

set_theme!(Fathom.fathom())

# Across-neuron distributions of the stable law fitted to each neuron's input current in the
# critical demo run (WRCircuit/scripts/demo_run.jl). These are `fit(Stable, ·)` parameters, so
# α is the stability index and β the SKEWNESS --- neither is the sampler's α or β, which the
# axis labels keep distinct.

begin # * Options
    NAME = "FigS4_input_parameters"
    outdir = plotsdir(NAME)
    datafile = datadir("WRCircuit", "demo_run_stats.jld2")
    nbins = 20
    # (key, panel title, axis label, plotted quantile window). A handful of neurons take
    # pathological location fits (μ spans -20.5 to 47.5 nA against an interquartile width of
    # ~0.1), which collapse an equal-width histogram to a single bar; μ is therefore binned over
    # its central 99%, and the panel says how many neurons fall outside. α, β and σ are drawn
    # over their full range --- β's tails reach the parameter's own ±1 bounds, so clipping them
    # would hide the fits that pinned there.
    series = [
        ("αs", "Stability index", "α", (0.0, 1.0)),
        ("βs", "Skewness", "β", (0.0, 1.0)),
        ("μs", "Location", "μ (nA)", (0.005, 0.995)),
        ("σs", "Scale", "σ (nA)", (0.0, 1.0)),
    ]
end

begin # * Load data (produced by WRCircuit/scripts/demo_run.jl)
    # The stats file also holds the Fano curves, spectra and MAD fits, so pull only these four
    # keys rather than `wload`ing 71 MB. They are stored as ToolsArrays; the typemap rebuilds them
    # without this project defining their custom dims.
    params = jldopen(datafile; typemap = toolsarray_typemap) do f
        Dict(k => collect(f[k]) for (k, _, _, _) in series)
    end
    nneurons = length(first(values(params)))
end

begin # * Render
    f = FourPanel()
    gs = subdivide(f, 2, 2)

    # Explicit edges rather than a bin count, so the saved source data is exactly what is drawn.
    binedges(x, q) = range(quantile(x, q[1]), quantile(x, q[2]), length = nbins + 1)

    for (i, (key, title, label, q)) in enumerate(series)
        x = params[key]
        edges = binedges(x, q)
        ax = Axis(gs[i]; title, xlabel = label, ylabel = "Density")
        ziggurat!(ax, x; bins = edges, normalization = :pdf, color = baikal)
        vlines!(
            ax, [median(x)]; color = bermejo, linestyle = :dash,
            label = "Median = $(round(median(x), digits = 2))"
        )
        axislegend(ax; position = :rt, framevisible = false, patchsize = (10, 10))
        # Neurons outside the drawn range are not annotated on the panel; the count is in
        # `medians.tsv` (`n_beyond_axis`), which is where the cropping is documented.
    end

    addlabels!(f)
    display(f)
end

begin # * Save figure
    mkpath(outdir)
    wsave(joinpath(outdir, "$NAME.pdf"), f)
    wsave(joinpath(outdir, "$NAME.svg"), f)
    wsave(joinpath(outdir, "$NAME.png"), f)
    @info "wrote figure" outdir
end

begin # * Save source data
    """
    One tsv per panel, holding the binned density the panel draws, plus the medians marked on
    them. The medians are only approximately recoverable from the bins, and they are the numbers
    quoted in the text, so they get their own file.
    """
    function save_source_data()
        savedir(x) = joinpath(outdir, x)
        for (i, (key, _, _, q)) in enumerate(series)
            x = params[key]
            edges = collect(binedges(x, q))
            w = StatsBase.normalize(
                StatsBase.fit(StatsBase.Histogram, x, edges); mode = :pdf
            ).weights
            writedlm(
                savedir("panel$(('A':'Z')[i])_$(key).tsv"),
                vcat(
                    ["bin_lower" "bin_upper" "pdf"],
                    hcat(edges[1:(end - 1)], edges[2:end], w)
                ), '\t'
            )
        end
        writedlm(
            savedir("medians.tsv"),
            vcat(
                ["parameter" "median" "n_neurons" "n_beyond_axis"],
                reduce(
                    vcat,
                    map(series) do (k, _, _, q)
                        x = params[k]
                        e = binedges(x, q)
                        n = count(<(first(e)), x) + count(>(last(e)), x)
                        permutedims([k, median(x), nneurons, n])
                    end
                )
            ), '\t'
        )
        return @info "wrote source data" outdir
    end
    save_source_data()
end
