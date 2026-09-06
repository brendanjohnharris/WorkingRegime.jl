#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Figure 1 panel: the distribution of L2/3 LFP increments against its Gaussian null, on a log density
# axis. A Gaussian is an inverted parabola in these axes, so heavier-than-Gaussian tails read
# directly as the data lifting away from the dashed curve in the wings.
#
# The dashed Gaussian IS the null the main text quotes: FT (phase-randomised) surrogates preserve the
# power spectrum but leave Gaussian increments, so their excess kurtosis is exactly zero. The thin
# surrogate curve is drawn from actual surrogates as an empirical check that this holds.
#
# `increment_panel!` is written to take an axis so the Figure 1 script can call it directly rather
# than duplicating the drawing; running this file standalone renders the panel on its own.
using DrWatson
@quickactivate "WorkingRegime"
using JLD2
using CairoMakie
using Fathom
using Statistics
using DelimitedFiles
using Printf

set_theme!(fathom())

const NAME = "increment_distribution"
const outdir = plotsdir(NAME)
const cache = projectdir("WRExperiment", "data", "increment_histograms.jld2")
const XLIM = 8          # standard deviations shown; the cache extends to 15
const YMIN = 1.0e-7

isfile(cache) || error("No $cache --- run WRExperiment/scripts/increment_histograms.jl first")
d = jldopen(f -> Dict(k => f[k] for k in keys(f)), cache)

gaussian(x) = exp(-x^2 / 2) / sqrt(2π)

"""
    increment_panel!(ax, d; showsurrogate = true)

Draw the pooled increment density from cache `d` on `ax`, with its Gaussian null. Returns `ax`.
Bins with no counts are dropped so the log axis stays finite.
"""
function increment_panel!(ax, d; showsurrogate = true)
    x, y, ys = d["centres"], d["density"], d["density_surrogate"]
    keep = (y .> 0) .& (abs.(x) .<= XLIM)
    if showsurrogate
        ks = (ys .> 0) .& (abs.(x) .<= XLIM)
        lines!(ax, x[ks], ys[ks]; color = (abyad, 0.9), linewidth = 1.5, label = "FT surrogate")
    end
    xg = range(-XLIM, XLIM, length = 400)
    lines!(ax, xg, gaussian.(xg); color = chernoe, linestyle = :dash, linewidth = 2, label = "Gaussian")
    lines!(ax, x[keep], y[keep]; color = baikal, linewidth = 2.5, label = "L2/3 data")
    ylims!(ax, YMIN, 1.0)
    xlims!(ax, -XLIM, XLIM)
    return ax
end

"""
Excess kurtosis annotation. The sweep and main text use the median ACROSS SESSIONS of each session's
median over its own L2/3 channels (`collect_surrogates.jl`); the flat per-channel median weights
sessions by channel count and reads higher, which is what made this panel disagree with the text.
Falls back to the flat vector for caches written before `kurtosis_session` existed.
"""
increment_kurtosis(d) = median(haskey(d, "kurtosis_session") ? d["kurtosis_session"] : d["kurtosis"])
kurtlabel(d) = @sprintf("κ = %.2f", increment_kurtosis(d))

begin # * Render
    fig = OnePanel()
    ax = Axis(
        fig[1, 1]; yscale = log10, xlabel = "Increment (SD)", ylabel = "Density",
        title = "L2/3 increments"
    )
    increment_panel!(ax, d)
    text!(
        ax, 0.03, 0.06; text = kurtlabel(d), space = :relative, align = (:left, :bottom),
        fontsize = 14, color = baikal
    )
    axislegend(ax; position = :rt, framevisible = false, patchsize = (14, 2))
    display(fig)
end

begin # * Save
    wsave(joinpath(outdir, "$NAME.pdf"), fig)
    wsave(joinpath(outdir, "$NAME.png"), fig)
    @info "Saved $(joinpath(outdir, "$NAME.pdf"))"
end

begin # * Source data --- exactly the three curves drawn, plus the quoted statistic
    mkpath(outdir)
    x = d["centres"]
    keep = abs.(x) .<= XLIM
    writedlm(
        joinpath(outdir, "panel.tsv"),
        vcat(
            ["increment_sd" "density" "density_surrogate" "gaussian"],
            hcat(x[keep], d["density"][keep], d["density_surrogate"][keep], gaussian.(x[keep]))
        ), '\t'
    )
    writedlm(
        joinpath(outdir, "statistics.tsv"),
        vcat(
            ["quantity" "value"],
            [
                "excess_kurtosis_session_median" increment_kurtosis(d)
                "excess_kurtosis_median" median(d["kurtosis"])
                "excess_kurtosis_surrogate_median" median(d["kurtosis_surrogate"])
                "n_channels" d["nchannels"]
                "n_sessions" length(d["sessions"])
            ]
        ), '\t'
    )
    @info "Saved source data to $outdir"
end
