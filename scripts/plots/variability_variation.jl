#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Variability exponent against the cortical hierarchy.
#
# The companion to the (a, b) panels of hierarchical_variation.jl, for the third exponent: the
# VARIABILITY exponent, the log-log slope of a unit's Fano-factor curve over 10^1.5-10^3 ms. Panel a
# is the per-region scatter at L2/3, panel b the hierarchy correlation (Kendall's 𝜏) at each layer,
# drawn in the same band-plus-filled/open-marker style.
#
# The exponent is RE-DERIVED here from the per-unit Fano curves rather than read from the stored
# `fano_slopes`, which is unusable: collect_calculations.jl reduces each session's per-unit slopes with
# a plain `mean`, so a single NaN unit (11% of units carry one) NaNs the whole session and 77% of the
# session x layer cells are lost. No cell has EVERY unit NaN, so a NaN-aware median across units
# recovers all of them. Deriving it here also keeps the script self-contained --- it needs only this
# project's own dependencies, never the WRExperiment environment.
#
# Resolution note: the reference panel resolves 20 cortical depths, but a unit's layer is the finest
# binning the stored Fano tables carry (`unitdepths.layer`), so 𝜏 is estimated per LAYER here. Putting
# it on the depth grid would mean choosing a depth-binning scheme for units, which is a modelling
# decision this script deliberately does not make on its own.

using DrWatson
@quickactivate "WorkingRegime"
using JLD2
using DataFrames
using TimeseriesTools
using TimeseriesBase
using CairoMakie
using Fathom
using MoreMaps
using Statistics
using Random
using DelimitedFiles

set_theme!(fathom())

const structures = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
const hierarchy_scores = Dict(
    "VISp" => -0.357, "VISl" => -0.093, "VISrl" => -0.059,
    "VISal" => 0.152, "VISpm" => 0.327, "VISam" => 0.441
)
const stim = "spontaneous"

# Layer integer codes in the saved data, and the layer the scatter panel is drawn at (L2/3, matching
# hierarchical_variation.jl's `points_l23`).
const layer_codes = 2:5
const layer_names = Dict(2 => "L2/3", 3 => "L4", 4 => "L5", 5 => "L6")
const SCATTER_LAYER = 2

# Band the Fano slope is fit over, in ms --- the same band collect_calculations.jl and fano_factors.jl
# use, so this exponent is the same quantity they plot.
const FANO_BAND_MS = (10^(1.5), 1.0e3)
const FANO_BAND = FANO_BAND_MS[1] .. FANO_BAND_MS[2]
const CURVE_WINDOW = 1.0 .. 1.0e3      # timescales drawn in the curve panel

const PTHR = 1.0e-2      # matches WRExperiment.PTHR; p-values below are BH-adjusted across layers
const NBOOT = 10_000
const SEED = 42

const NAME = "variability_variation"
const PATH = plotsdir(NAME)
const inpath = projectdir("WRExperiment", "data", "WRExperiment.jld2")

# ──────────────────────────────────────────────────────────────────────────────
# Derive the variability exponent per (structure, session, layer)
# ──────────────────────────────────────────────────────────────────────────────

"Log-log slope of one unit's Fano-factor curve over `FANO_BAND`; `NaN` if the curve is missing."
function fano_slope(fano)
    (fano isa Number) && isnan(fano) && return NaN
    y = log10.(fano[FANO_BAND])
    t = log10.(times(y))
    return last(hcat(ones(length(t)), t) \ y)
end

"""
    nanmean(v)

The reduction across a session's units: `mean` over the units that produced a slope. Matches
collect_calculations.jl's `mean` over `Unit` now that it drops the NaN units first, so this figure and
the stored `fano_slopes` compute the same quantity. `NaN` only if every unit failed (which happens
for no cell in this dataset).
"""
nanmean(v) = (w = filter(!isnan, v); isempty(w) ? NaN : mean(w))

"""
    session_curve(df) -> (t, fano) | nothing

Median Fano-factor curve across one session's units at [`SCATTER_LAYER`](@ref). A unit with no curve
is stored as a scalar `NaN` rather than a vector, so those are dropped by type before the median ---
the same guard collect_calculations.jl uses for its VISp curve, and the reason the curve pipeline
there is sound while the slope pipeline is not.
"""
function session_curve(df)
    u = subset(df, :layer => ByRow(==(SCATTER_LAYER)))
    curves = [c for c in u.fano_factor if c isa AbstractVector]
    isempty(curves) && return nothing
    n = length(first(curves))
    all(c -> length(c) == n, curves) || return nothing   # off-grid session; drop rather than misalign
    return collect(times(first(curves))), vec(median(reduce(hcat, collect.(curves)); dims = 2))
end

"""
    derive_exponents(_) -> Dict

`"exponents"` maps each layer code to a `(SessionID × Structure)` matrix of variability exponents on
that layer's shared session set, plus `"sessions"` (the session ids of each matrix's rows), and
`"curves"`/`"curve_t"`, the median Fano-factor curve per region at [`SCATTER_LAYER`](@ref).
"""
function derive_exponents(_)
    @info "Loading unit-level Fano curves from $inpath (this is the slow step)"
    unitdepths = jldopen(f -> f["fano_data"], inpath, "r"; typemap = toolsarray_typemap)[stim]["unitdepths"]

    # One (session, structure) task per stored table; each returns that session's per-layer exponents.
    tasks = [(si, df) for si in eachindex(structures) for df in unitdepths[si]]
    @info "Fitting Fano slopes over $(length(tasks)) (structure, session) tables"
    rows = map(Chart(LogLogger(), Threaded()), tasks) do (si, df)
        sid = only(unique(df.ecephys_session_id))
        vals = map(layer_codes) do l
            u = subset(df, :layer => ByRow(==(l)))
            isempty(u) ? NaN : nanmean(fano_slope.(u.fano_factor))
        end
        return (; structure = structures[si], session = sid, vals, curve = session_curve(df))
    end

    # Per layer: keep the sessions every structure recorded, then lay them out (session × structure).
    bysl = Dict((r.structure, r.session) => r.vals for r in rows)
    exponents, sessions = Dict{Int, Matrix{Float64}}(), Dict{Int, Vector{Int}}()
    for (k, l) in enumerate(layer_codes)
        have = [Set(r.session for r in rows if r.structure == s && isfinite(r.vals[k])) for s in structures]
        common = sort(collect(intersect(have...)))
        exponents[l] = [bysl[(s, sid)][k] for sid in common, s in structures]
        sessions[l] = common
        @info "Layer $(layer_names[l]): $(length(common)) sessions × $(length(structures)) structures"
    end

    # Median Fano curve per region: median across each session's units (done in `session_curve`), then
    # median across sessions. Every session shares the stored τ grid, so take it from the first curve.
    curve_t = collect(first(r.curve for r in rows if r.curve !== nothing)[1])
    curves = Dict(
        s => vec(
                median(
                    reduce(hcat, [r.curve[2] for r in rows if r.structure == s && r.curve !== nothing]);
                    dims = 2
                )
            ) for s in structures
    )
    for s in structures
        n = count(r -> r.structure == s && r.curve !== nothing, rows)
        @info "$s: median Fano curve over $n sessions"
    end
    return Dict(
        "exponents" => exponents, "sessions" => sessions,
        "curves" => curves, "curve_t" => curve_t,
    )
end

data, datapath = produce_or_load(
    derive_exponents, Dict("stim" => stim, "band" => "1.5-3"), datadir(NAME);
    filename = "variability_exponents", tag = true
)
exponents = data["exponents"]
curves = data["curves"]
curve_t = data["curve_t"]

# ──────────────────────────────────────────────────────────────────────────────
# Statistics --- implemented here so the script needs no dependency beyond this project's own
# ──────────────────────────────────────────────────────────────────────────────

"""
    kendall(x, y)

Kendall's 𝜏-b. Tie-corrected, which matters here: `x` repeats each region's hierarchy score once per
session, so a large fraction of pairs are tied in `x`.
"""
function kendall(x, y)
    n = length(x)
    n0 = n * (n - 1) ÷ 2
    n1 = n2 = nc = nd = 0
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dx == 0 && (n1 += 1)
        dy == 0 && (n2 += 1)
        (dx == 0 || dy == 0) && continue
        dx * dy > 0 ? (nc += 1) : (nd += 1)
    end
    den = sqrt(float((n0 - n1)) * float((n0 - n2)))
    return den == 0 ? NaN : (nc - nd) / den
end

"""
    hierarchy_tau(Y; N = NBOOT) -> (τ, (lo, hi), p)

Kendall's 𝜏 between the hierarchy score and the variability exponent, pooling every
(session, region) pair of one `(SessionID × Structure)` matrix. The interval is a percentile
bootstrap over those pairs (the reference panel uses a BCa interval via Bootstrap.jl; percentile
keeps this script dependency-free and matches `bootstrapmedian` elsewhere in these plot scripts).
`p` is a permutation test that reshuffles region labels independently within each session, so it
preserves each session's spread and destroys only the hierarchy ordering.
"""
function hierarchy_tau(Y; N = NBOOT)
    nsesh, nstruct = size(Y)
    X = repeat(permutedims([hierarchy_scores[s] for s in structures]), nsesh, 1)
    x, y = vec(X), vec(Y)
    τ = kendall(x, y)

    rng = Random.MersenneTwister(SEED)
    idx = eachindex(x)
    boot = map(1:N) do _
        k = rand(rng, idx, length(idx))
        kendall(view(x, k), view(y, k))
    end
    lo, hi = quantile(filter(!isnan, boot), (0.025, 0.975))

    surrogate = map(1:N) do _
        Ys = reduce(vcat, permutedims(Y[i, randperm(rng, nstruct)]) for i in 1:nsesh)
        kendall(x, vec(Ys))
    end
    p = mean(abs(τ) .< abs.(filter(!isnan, surrogate)))
    return τ, (lo, hi), p
end

"Benjamini-Hochberg step-up adjustment, matching the correction the reference panel applies."
function bh_adjust(p)
    n = length(p)
    order = sortperm(p; rev = true)
    q = similar(p)
    running = 1.0
    for (rank, i) in zip(n:-1:1, order)
        running = min(running, p[i] * n / rank)
        q[i] = running
    end
    return q
end

"""
    selfcheck()

The statistics above are hand-rolled, so this is the smallest thing that fails if they break: `kendall`
against cases whose 𝜏 is known by inspection, then `hierarchy_tau` end to end --- it must recover a
planted hierarchy signal, and reject the same data with region labels shuffled, in exactly the
`(session × structure)` layout the real call uses. Without this a bug would surface as a null result,
which is indistinguishable from the finding.
"""
function selfcheck()
    @assert kendall([1, 2, 3], [1, 2, 3]) ≈ 1
    @assert kendall([1, 2, 3], [3, 2, 1]) ≈ -1
    @assert isnan(kendall([1, 1, 1], [1, 2, 3]))   # every pair tied in x
    rng = Random.MersenneTwister(0)
    h = [hierarchy_scores[s] for s in structures]
    planted = reduce(vcat, (permutedims(2 .* h .+ 0.3 .* randn(rng, length(h))) for _ in 1:40))
    τ, _, p = hierarchy_tau(planted; N = 200)
    @assert τ > 0.5 "planted hierarchy signal not recovered (𝜏 = $τ)"
    @assert p < 0.05 "planted hierarchy signal not significant (p = $p)"
    shuffled = reduce(vcat, (permutedims(planted[i, randperm(rng, length(h))]) for i in axes(planted, 1)))
    @assert abs(first(hierarchy_tau(shuffled; N = 200))) < 0.3 "shuffled labels still correlate"
    return true
end
@assert selfcheck()

"Percentile-bootstrap median and 95% interval of one region's exponents across sessions."
function bootstrapmedian(v; N = NBOOT)
    w = filter(!isnan, v)
    isempty(w) && return (NaN, (NaN, NaN))
    rng = Random.MersenneTwister(SEED)
    boot = [median(w[rand(rng, eachindex(w), length(w))]) for _ in 1:N]
    return median(w), Tuple(quantile(boot, (0.025, 0.975)))
end

taus = map(collect(layer_codes)) do l
    τ, σ, p = hierarchy_tau(exponents[l])
    (; layer = l, τ, σ, p)
end
adjusted = bh_adjust([t.p for t in taus])
taus = [(; t..., q = q) for (t, q) in zip(taus, adjusted)]

for t in taus
    @info "$(layer_names[t.layer]): 𝜏 = $(round(t.τ; digits = 3)) " *
        "[$(round(t.σ[1]; digits = 3)), $(round(t.σ[2]; digits = 3))], q = $(round(t.q; sigdigits = 3))"
end

# Per-region points at the scatter layer, ordered low → high hierarchy.
const points = let Y = exponents[SCATTER_LAYER]
    pts = map(enumerate(structures)) do (j, s)
        m, (lo, hi) = bootstrapmedian(Y[:, j])
        (; structure = s, h = hierarchy_scores[s], m, lo, hi)
    end
    sort(pts; by = p -> p.h)
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

# Region colours keep hierarchical_variation.jl's dark → light hierarchy ordering, but the gradient is
# truncated before its near-white end (luminance 0.95), which is invisible as a line on the page.
const region_colors = [
    get(cgrad(binarysunset), x) for x in range(0, 0.7, length = length(structures))
]
const structure_color = Dict(s => region_colors[i] for (i, s) in enumerate(structures))
const SERIES = ianthina

begin # * Render
    # Three panels in a row. Fathom has no ThreePanel preset, so this is a plain `Figure` sized as
    # three OnePanel cells (360 × 270 each) to stay on the theme's proportions.
    f = Figure(; size = (1080, 270))
    gs = subdivide(f, 1, 3)

    begin # * a --- the curves the exponent is a slope of, one per region
        ax = Axis(
            gs[1][1, 1]; xlabel = "Timescale (ms)", ylabel = "Fano factor",
            xscale = log10, yscale = log10,
            title = "$stim, $(layer_names[SCATTER_LAYER])"
        )
        vspan!(ax, FANO_BAND_MS...; color = (:gray, 0.12), strokewidth = 0)   # the band the slope is fit over
        keep = findall(t -> t in CURVE_WINDOW, curve_t)
        for s in structures   # `structures` is already ordered low → high hierarchy
            lines!(ax, curve_t[keep], curves[s][keep]; color = structure_color[s], linewidth = 2, label = s)
        end
        axislegend(ax; position = :lt, framevisible = false, labelsize = 9, nbanks = 2)
    end

    begin # * b --- the correlation itself: each region's exponent against its hierarchy score
        ax = Axis(
            gs[2][1, 1]; xlabel = "Hierarchy score",
            ylabel = "Variability exponent",
            title = "Exponent vs hierarchy",
            yautolimitmargin = (0.08, 0.22)   # headroom for the region labels above each point
        )
        xs = [p.h for p in points]
        ys = [p.m for p in points]
        errorbars!(
            ax, xs, ys, ys .- [p.lo for p in points], [p.hi for p in points] .- ys;
            color = :gray70, whiskerwidth = 6
        )
        scatter!(
            ax, xs, ys; color = [structure_color[p.structure] for p in points],
            markersize = 18, strokecolor = :black, strokewidth = 0.8
        )
        for p in points   # regions are few enough to name directly; no legend needed
            text!(
                ax, p.h, p.m; text = p.structure, fontsize = 10,
                align = (:center, :bottom), offset = (0, 10)
            )
        end
        t = taus[findfirst(t -> t.layer == SCATTER_LAYER, taus)]
        text!(
            ax, 0.02, 0.02; text = "𝜏 = $(round(t.τ; digits = 3)), q = $(round(t.q; sigdigits = 2))",
            space = :relative, align = (:left, :bottom), fontsize = 12
        )
    end

    begin # * c --- the hierarchy correlation at each layer, in the reference panel's style
        ax = Axis(
            gs[3][1, 1]; xlabel = "Kendall's 𝜏", ylabel = "Cortical layer",
            yticks = (collect(layer_codes), [layer_names[l] for l in layer_codes]),
            title = "Hierarchy correlation", yreversed = true
        )
        vlines!(ax, 0; color = :gray, linestyle = :dash, linewidth = 1)
        ls = [Float64(t.layer) for t in taus]
        band!(
            ax, Point2f.([t.σ[1] for t in taus], ls), Point2f.([t.σ[2] for t in taus], ls);
            color = (SERIES, 0.25)
        )
        lines!(ax, [t.τ for t in taus], ls; color = SERIES, linewidth = 2)
        sig = [t.q < PTHR for t in taus]
        scatter!(ax, [t.τ for t in taus][sig], ls[sig]; color = SERIES, markersize = 12)
        scatter!(
            ax, [t.τ for t in taus][.!sig], ls[.!sig]; color = :transparent,
            strokecolor = SERIES, strokewidth = 1, markersize = 12
        )
        @info "$(count(sig))/$(length(sig)) layers significant at q < $PTHR"
    end

    addlabels!(f)
    display(f)
end

begin # * Save figure
    wsave(PATH * ".pdf", f)
    wsave(PATH * ".png", f)
    @info "Saved $PATH"
end

begin # * Save source data
    mkpath(PATH)
    keep = findall(t -> t in CURVE_WINDOW, curve_t)
    writedlm(
        joinpath(PATH, "panelA.tsv"),
        vcat(
            hcat("t_ms", permutedims(structures)),
            hcat(curve_t[keep], reduce(hcat, [curves[s][keep] for s in structures]))
        ), '\t'
    )
    writedlm(
        joinpath(PATH, "panelB.tsv"),
        vcat(
            ["structure" "hierarchy" "median" "lo" "hi"],
            reduce(vcat, [permutedims([p.structure, p.h, p.m, p.lo, p.hi]) for p in points])
        ), '\t'
    )
    writedlm(
        joinpath(PATH, "panelC.tsv"),
        vcat(
            ["layer" "tau" "lo" "hi" "p" "q"],
            reduce(vcat, [permutedims([layer_names[t.layer], t.τ, t.σ[1], t.σ[2], t.p, t.q]) for t in taus])
        ), '\t'
    )
    @info "Saved source data to $PATH"
end
