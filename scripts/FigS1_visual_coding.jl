#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Supplementary figure: the Allen Visual Coding functional-connectivity cohort, shown in the same
# panels the main figures use for Visual Behaviour, so the two datasets can be read against each
# other rather than described separately.
#
#   a-d  the Figure 1 curves for VISp at L2/3 --- increment distribution, diffusion, power spectrum
#        and Fano factor --- one curve per panel, each carrying its exponent, as Figure 1 draws them.
#   e    the (a, b) plane at L2/3, one point per visual area.
#   f    the hierarchy correlation of all three exponents, by layer.
#
# Draws only. Every number comes from `WRExperiment/scripts/collect_calculations_visual_coding.jl`,
# which is fed by `run_calculations_visual_coding.jl`; nothing is fit here. Panels a-d are VISp alone,
# so each dashed guide carries VISp's median MAPPLE exponent over the band that exponent was fit
# over, and reads against VISp's point in panel e rather than against the cohort as a whole.
using DrWatson
@quickactivate "WorkingRegime"
using JLD2
using CairoMakie
using Fathom
using TimeseriesTools
using Statistics
using Random
using Printf
using DelimitedFiles
import StatsBase: corkendall

# The per-session hierarchy correlation, shared verbatim with `WRExperiment` and Figure 4.
# `include`d rather than imported, as WRExperiment is not a dependency of the root project.
include(projectdir("WRExperiment", "src", "SessionKendall.jl"))

set_theme!(fathom())

const NAME = "FigS1_visual_coding"
const outdir = plotsdir(NAME)
mkpath(outdir)

const COHORT = isempty(ARGS) ? "fc" : lowercase(ARGS[1])
const STRUCTURES = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
const LAYERNAMES = Dict(1 => "L1", 2 => "L2/3", 3 => "L4", 4 => "L5", 5 => "L6")
# Exponent colours as in Figure 4's hierarchy panel: a <-> alpha blue, b <-> beta red, c glas.
const EXPCOL = Dict("a" => baikal, "b" => bermejo, "c" => glas)
const PTHR = 1.0e-2
const MIN_AREAS = 4   # areas a session must retain before its tau is used; coverage here is ragged
const L23 = 2 # panels a-e are all L2/3, the layer Figure 1 draws
# Bands each dashed guide spans, matching the window its exponent was fit over: the diffusion
# exponent's fitting window, the spectrum's displayed range (as Figure 1 draws its aperiodic
# line, over the whole panel) and the Fano curve's rising range.
const MAD_BAND_MS = (0.0, 8.0)
const PSD_BAND_HZ = (2.0, 500.0)
const FANO_BAND_MS = (30.0, 1000.0)
const region_colors = cgrad(binarysunset, length(STRUCTURES); categorical = true)
const scolor = Dict(s => region_colors[i] for (i, s) in enumerate(STRUCTURES))

D = jldopen(f -> Dict(k => f[k] for k in keys(f)),
    datadir("WRExperiment", "visual_coding_$(COHORT).jld2"))
INC = jldopen(f -> Dict(k => f[k] for k in keys(f)),
    datadir("WRExperiment", "visual_coding_increments_$(COHORT).jld2"))
@info "loaded" cohort = D["cohort"] blocks = D["nblocks"] sessions = length(D["sessions"])

taus, fr, ftaus = D["taus"], D["freqs"], D["fano_taus"]
taulayers, hvec = D["taulayers"], D["hierarchy"]
const VISP = findfirst(==("VISp"), D["structures"]) # the area the curve panels draw

"Bootstrap median + 95% CI. Local, as in Fig1, so this script needs no Bootstrap.jl."
function bootmedian(x; N = 10_000, α = 0.05)
    x = filter(!isnan, collect(float.(x)))
    isempty(x) && return (NaN, (NaN, NaN))
    rng, n = Random.MersenneTwister(42), length(x)
    meds = [median(x[rand(rng, 1:n, n)]) for _ in 1:N]
    return median(x), Tuple(quantile(meds, (α / 2, 1 - α / 2)))
end

"Column medians of a (session × structure) exponent matrix, with their bootstrap CIs."
function structuremedians(Y)
    ms = [bootmedian(view(Y, :, j)) for j in axes(Y, 2)]
    return first.(ms), getindex.(ms, 2)
end

"Median exponent `k` at layer `l` over VISp's sessions, matching the area the curves show."
vispmedian(k, l = L23) = median(filter(isfinite, view(D[k][l], :, VISP)))

"""
Dashed power-law guide of slope `β` spanning `band`, laid on the curve `(x, y)` at the band's
geometric centre so it reads as a tangent to it, and labelled clear of its own line: a rising guide
takes the label to its right and a falling one to its left, the side on which the line runs above the
anchor. `scale` lifts the guide off the curve where a tangent would be ambiguous.
"""
function guide!(ax, x, y, band, β, label; color, scale = 1.0, n = 20, pad = 7)
    lo, hi = band
    xx = exp10.(range(log10(lo), log10(hi); length = n))
    x0 = exp10((log10(lo) + log10(hi)) / 2)
    y0 = scale * y[argmin(abs.(x .- x0))]
    lines!(ax, xx, y0 .* (xx ./ x0) .^ β; color, linestyle = :dash, linewidth = 2)
    text!(ax, x0, y0; text = label, color, align = (β >= 0 ? :left : :right, :top),
        offset = (β >= 0 ? pad : -pad, -pad), glowcolor = :white, glowwidth = 5)
    return nothing
end

begin # * Render
    fig = FourPanel()  # 720 x 540, laid out as two rows of three so the curve row matches Figure 1's

    # ---------------------------------------------------------------- a: increment distribution
    axa = Axis(fig[1, 1]; yscale = log10, xlabel = "Increment / SD", ylabel = "Density",
        title = "Increments", limits = ((-9, 9), (1.0e-6, 1)))
    lines!(axa, INC["centres"], max.(INC["density"], 1.0e-12); color = baikal, linewidth = 2.5,
        label = "VISp L2/3")
    lines!(axa, INC["centres"], max.(INC["density_surrogate"], 1.0e-12); color = (:black, 0.7),
        linewidth = 2, linestyle = :dash, label = "FT surrogate")
    axislegend(axa; position = :lt, framevisible = false, patchsize = (18, 2))

    # ---------------------------------------------------------------- b: diffusion
    a23 = vispmedian("a")
    madband = (max(MAD_BAND_MS[1], 1000 * first(taus)), MAD_BAND_MS[2])
    axb = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = "Lag (ms)", ylabel = "MAD (V)",
        title = "Diffusion")
    vspan!(axb, madband...; color = (:gray, 0.12), strokewidth = 0) # the fitted band
    lines!(axb, 1000 .* taus, D["mad_visp"][L23]; color = baikal, linewidth = 2.5)
    guide!(axb, 1000 .* taus, D["mad_visp"][L23], madband, a23,
        "a = $(round(a23, sigdigits = 2))"; color = baikal)

    # ---------------------------------------------------------------- c: power spectrum
    b23 = vispmedian("b")
    axc = Axis(fig[1, 3]; xscale = log10, yscale = log10, xlabel = "Frequency (Hz)",
        ylabel = "PSD (V² Hz⁻¹)", title = "Spectrum", xticks = [3, 10, 30, 100],
        limits = ((2, 500), nothing))
    lines!(axc, fr, max.(D["psd_visp"][L23], 1.0e-20); color = baikal, linewidth = 2.5)
    guide!(axc, fr, D["psd_visp"][L23], PSD_BAND_HZ, b23,
        "b = $(round(b23, sigdigits = 3))"; color = baikal)

    # ---------------------------------------------------------------- d: Fano factor
    c23 = vispmedian("c")
    axd = Axis(fig[2, 1]; xscale = log10, yscale = log10, xlabel = "Bin width (ms)",
        ylabel = "Fano factor", title = "Fano factor")
    let y = D["fano_visp"][L23], k = findall(!isnan, y)
        lines!(axd, ftaus[k], y[k]; color = baikal, linewidth = 2.5)
        guide!(axd, ftaus, y, FANO_BAND_MS, c23, "c = $(round(c23, digits = 2))"; color = baikal)
    end

    # ---------------------------------------------------------------- e: the (a, b) plane at L2/3
    axe = Axis(fig[2, 2]; xlabel = "Diffusion exponent  a", ylabel = "Spectral exponent  b",
        title = "L2/3 exponents")
    am, aci = structuremedians(D["a"][2])
    bm, bci = structuremedians(D["b"][2])
    for (j, s) in enumerate(STRUCTURES)
        (isnan(am[j]) || isnan(bm[j])) && continue
        rangebars!(axe, [am[j]], [bci[j][1]], [bci[j][2]]; color = (:gray, 0.7), whiskerwidth = 6)
        rangebars!(axe, [bm[j]], [aci[j][1]], [aci[j][2]]; color = (:gray, 0.7), whiskerwidth = 6,
            direction = :x)
        scatter!(axe, [am[j]], [bm[j]]; color = scolor[s], markersize = 15, strokewidth = 1,
            strokecolor = :black)
        text!(axe, am[j], bm[j]; text = s, fontsize = 8, align = (:center, :bottom),
            offset = (0, 9), glowcolor = :white, glowwidth = 5)
    end

    # ---------------------------------------------------------------- f: hierarchy correlation
    axf = Axis(fig[2, 3]; yreversed = true, xlabel = "Kendall's 𝜏", ylabel = "Cortical layer",
        yticks = (1:length(taulayers), [LAYERNAMES[l] for l in taulayers]),
        title = "Hierarchy", xticks = -1:0.5:1,
        limits = ((-1.08, 1.08), (1 - 0.62, length(taulayers) + 0.62))) # as Figure 4
    # Layer separators and the zero line, drawn as Figure 4's equivalent panel does.
    hlines!(axf, (1:(length(taulayers) - 1)) .+ 0.5; color = (:gray, 0.4), linewidth = 0.5)
    vlines!(axf, [0]; color = :gray, linestyle = :dash, linewidth = 1)
    offsets = Dict("a" => -0.18, "b" => 0.0, "c" => 0.18)
    # Recomputed here from the stored `(session x area)` exponent matrices through the same
    # `sessionkendall` Figure 4c uses, rather than read back as a bare list of taus: the interval and
    # the test have to come from the same construction in both figures. Coverage is ragged in this
    # cohort --- no functional-connectivity session records all six areas --- so `minareas` does real
    # work here, unlike in Visual Behaviour where every retained session is complete.
    taucells = NamedTuple[]
    for sym in ("a", "b", "c"), (i, l) in enumerate(taulayers)
        haskey(D[sym], l) || continue
        r = sessionkendall(hvec, Float64.(D[sym][l]); minareas = MIN_AREAS)
        r.nsessions < 5 && continue
        push!(taucells, (; sym, layer = l, row = i, r...))
    end
    # Benjamini-Hochberg across the 12 cells this panel draws, as in Figure 4c.
    taucells = [(; c..., padj = q) for (c, q) in zip(taucells, bhadjust([c.p for c in taucells]))]
    taurows = [[c.sym c.layer c.nsessions c.tau c.ci[1] c.ci[2] c.meantau c.p c.padj (c.padj < PTHR)]
               for c in taucells]

    jrng = Random.MersenneTwister(7)
    for sym in ("a", "b", "c")
        cells = [c for c in taucells if c.sym == sym]
        isempty(cells) && continue
        c = EXPCOL[sym]
        # The per-session taus behind each median, as a jittered strip: with at most six areas the
        # values sit on a coarse grid, which the median and its interval otherwise hide.
        for cell in cells
            y0 = cell.row + offsets[sym]
            scatter!(axf, cell.taus, y0 .+ (rand(jrng, length(cell.taus)) .- 0.5) .* 0.24;
                color = (c, 0.22), markersize = 3, strokewidth = 0)
        end
        # Marker, whisker and open/filled convention as in Figure 4's hierarchy panel: filled where
        # the BH-adjusted permutation p clears PTHR, open where it does not.
        ys = [cell.row + offsets[sym] for cell in cells]
        ms = [cell.tau for cell in cells]
        los, his = [cell.ci[1] for cell in cells], [cell.ci[2] for cell in cells]
        filled = [cell.padj < PTHR for cell in cells]
        lines!(axf, ms, ys; color = (c, 0.4), linewidth = 1.5)
        rangebars!(axf, ys, los, his; direction = :x, color = c, linewidth = 1.5, whiskerwidth = 6)
        scatter!(axf, ms[filled], ys[filled]; color = c, markersize = 10, label = sym)
        any(.!filled) && scatter!(axf, ms[.!filled], ys[.!filled]; color = :transparent,
            strokecolor = c, strokewidth = 1, markersize = 10, label = sym)
    end
    axislegend(axf; position = :rb, framevisible = false, merge = true, patchsize = (10, 10))

    addlabels!([fig[1, 1], fig[1, 2], fig[1, 3], fig[2, 1], fig[2, 2], fig[2, 3]], fig)
    fig
end

begin # * Save
    wsave(joinpath(outdir, "$NAME.pdf"), fig)
    wsave(joinpath(outdir, "$NAME.png"), fig)   # raster preview

    writedlm(joinpath(outdir, "panelA.tsv"),
        vcat(["increment_sd" "density" "density_surrogate"],
            hcat(INC["centres"], INC["density"], INC["density_surrogate"])), '\t')
    for (nm, key, x, xnm, exponent) in (("panelB", "mad_visp", 1000 .* taus, "lag_ms", "a"),
        ("panelC", "psd_visp", fr, "frequency_hz", "b"),
        ("panelD", "fano_visp", ftaus, "bin_width_ms", "c"))
        writedlm(joinpath(outdir, "$(nm).tsv"),
            vcat([xnm "VISp_L2/3" "exponent_$(exponent)"],
                hcat(x, D[key][L23], fill(vispmedian(exponent), length(x)))), '\t')
    end
    am, aci = structuremedians(D["a"][2])
    bm, bci = structuremedians(D["b"][2])
    writedlm(joinpath(outdir, "panelE.tsv"),
        vcat(["structure" "hierarchy" "a" "a_lo" "a_hi" "b" "b_lo" "b_hi"],
            hcat(STRUCTURES, hvec, am, first.(aci), last.(aci), bm, first.(bci), last.(bci))), '\t')
    writedlm(joinpath(outdir, "panelF.tsv"),
        vcat(["exponent" "layer" "n_sessions" "tau" "ci_lo" "ci_hi" "mean_tau" "p" "p_adj" "significant"],
            reduce(vcat, taurows)), '\t')
    # The per-session values behind each median, long format, as Figure 4c's `panelC.tsv`.
    writedlm(joinpath(outdir, "panelF_sessions.tsv"),
        vcat(["exponent" "layer" "session_tau" "n_areas"],
            reduce(vcat, [permutedims([c.sym, c.layer, t, na])
                          for c in taucells for (t, na) in zip(c.taus, c.nareas)])), '\t')
    @info "saved" outdir
end
