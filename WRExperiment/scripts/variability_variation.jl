#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Variability exponent against the cortical hierarchy --- a CALCULATION, not a figure.
#
# Derives the third exponent for Figure 4: the VARIABILITY exponent, the scaling slope of each
# session's unit-median Fano-factor curve (the BIC-selected MAPPLE fit, `variability_exponent` from
# this package). The result is cached under `datadir(NAME)` and read by
# `scripts/Fig4_hierarchical_variation.jl`; the standalone figure this script used to draw is gone.
#
# The hierarchy statistics below are kept because they are self-checking (see `selfcheck`) and their
# per-layer 𝜏 is worth having in the log, but nothing downstream reads them --- Figure 4 recomputes
# 𝜏 from the cached exponents so that a, b and c are correlated identically.
#
# The exponent is RE-DERIVED here from the unit Fano curves rather than read from the stored
# `fano_slopes` (which are per-unit OLS slopes over a fixed 31.6-1000 ms band, session-meaned with a
# NaN-poisoning bug): each (session, layer) cell's exponent is the unified BIC-selected MAPPLE fit
# to that cell's unit-MEDIAN curve, and Figure 4 medians across sessions. Aggregating curves before
# fitting is essential --- per-unit curves lack the SNR for any free-knot fit --- and the fitted
# knots measure the scaling band instead of assuming it, which is what lets the same estimator serve
# the circuit (whose regime sits at 10-100 ms, outside the old fixed band) and the mean-field sweep.
# It lives in WRExperiment (rather than being shared through a script `include`) so that the
# estimator comes from the package it belongs to; see WRExperiment/src/Variability.jl.
#
# Resolution note: Figure 4 resolves a and b over 20 cortical depths, but a unit's layer is the
# finest binning the stored Fano tables carry (`unitdepths.layer`), so c is per LAYER only. Putting
# it on the depth grid would mean choosing a depth-binning scheme for units, which is a modelling
# decision this script deliberately does not make on its own.

using DrWatson
@quickactivate "WRExperiment"
import WRExperiment: variability_exponent  # named import: the script defines its own `structures` etc.
using JLD2
using DataFrames
using TimeseriesTools
using MoreMaps
using Statistics
using Random
using Optim, ForwardDiff # TimeseriesTools' OptimExt: MAPPLE `fit!` (and its `fix` keyword) need both


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

const CURVE_WINDOW = 1.0 .. 1.0e3      # timescales drawn in the curve panel
const MIN_UNITS = 5                    # fewest unit curves for a (session, layer) median to be fit

const PTHR = 1.0e-2      # matches WRExperiment.PTHR; p-values below are BH-adjusted across layers
const NBOOT = 10_000
const SEED = 42

const NAME = "variability_variation"
const inpath = datadir("WRExperiment.jld2")

# ──────────────────────────────────────────────────────────────────────────────
# Derive the variability exponent per (structure, session, layer)
# ──────────────────────────────────────────────────────────────────────────────

# `variability_exponent` is vendored per package; this copy comes from WRExperiment, see
# WRExperiment/src/Variability.jl. Here it is fit to each (session, layer) cell's unit-MEDIAN curve,
# with a per-refine time cap so a rare ill-conditioned median curve cannot stall the sweep.
session_exponent(t, fano) = variability_exponent(t, fano; time_limit = 20.0)

"""
    session_curve(df, layer) -> (t, fano) | nothing

Median Fano-factor curve across one session's units at `layer`. A unit with no curve is stored as
a scalar `NaN` rather than a vector, so those are dropped by type before the median (the same
guard collect_calculations.jl uses for its VISp curve); sessions with fewer than [`MIN_UNITS`](@ref)
curves are dropped rather than fit through a barely-averaged median.
"""
function session_curve(df, layer)
    u = subset(df, :layer => ByRow(==(layer)))
    curves = [c for c in u.fano_factor if c isa AbstractVector]
    length(curves) < MIN_UNITS && return nothing
    n = length(first(curves))
    all(c -> length(c) == n, curves) || return nothing   # off-grid session; drop rather than misalign
    M = reduce(hcat, collect.(curves))
    med = [ (w = filter(!isnan, view(M, i, :)); isempty(w) ? NaN : median(w)) for i in 1:n ]
    return collect(times(first(curves))), med # NaN-safe per lag: units carry scattered NaN lags
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

    # One (session, structure) task per stored table; each returns that session's per-layer exponents
    # (one fit per layer's unit-median curve) and its fitted band at the scatter layer.
    tasks = [(si, df) for si in eachindex(structures) for df in unitdepths[si]]
    @info "Fitting unit-median Fano curves over $(length(tasks)) (structure, session) tables"
    nofit = VARIABILITY_NOFIT
    rows = map(Chart(LogLogger(), Threaded()), tasks) do (si, df)
        sid = only(unique(df.ecephys_session_id))
        fits = map(layer_codes) do l
            c = session_curve(df, l)
            isnothing(c) && return nofit
            try
                session_exponent(c...)
            catch
                nofit
            end
        end
        vals = [fit.β for fit in fits]
        band = (fits[findfirst(==(SCATTER_LAYER), layer_codes)].lo,
            fits[findfirst(==(SCATTER_LAYER), layer_codes)].hi)
        println("[fit] $(structures[si]) $sid: ", join(round.(vals; digits = 2), " "))
        flush(stdout) # stderr/logger output is buffered when redirected; this is the live signal
        return (; structure = structures[si], session = sid, vals, band,
            curve = session_curve(df, SCATTER_LAYER))
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
    # Median fitted scaling band across sessions: the estimator measures its band per session
    # rather than assuming one, so this records where those bands typically sit.
    los = filter(isfinite, [r.band[1] for r in rows])
    his = filter(isfinite, [r.band[2] for r in rows])
    return Dict(
        "exponents" => exponents, "sessions" => sessions,
        "curves" => curves, "curve_t" => curve_t,
        "band" => (median(los), median(his)),
        "estimator" => ESTIMATOR,
    )
end

# The filename is fixed --- `scripts/Fig4_hierarchical_variation.jl` reads exactly this name --- so
# DrWatson cannot vary the cache by config, and a changed estimator would otherwise be served
# silently from the old file. The estimator is therefore recorded IN the result and checked on load.
# Files predating that key carry no estimator and are accepted: they came from this same fit.
const ESTIMATOR = "floor-bic"
data, datapath = produce_or_load(
    derive_exponents, Dict("stim" => stim, "estimator" => ESTIMATOR), datadir(NAME);
    filename = "variability_exponents", tag = true
)
let cached = get(data, "estimator", nothing)
    isnothing(cached) || cached == ESTIMATOR ||
        error("$(datapath) was produced by estimator $(repr(cached)), not $(repr(ESTIMATOR)); \
               delete it to refit")
end
exponents = data["exponents"]

# ──────────────────────────────────────────────────────────────────────────────
# Statistics --- hand-rolled and self-checked below
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
keeps this script dependency-free and matches `percentilebootmedian` elsewhere in these plot scripts).
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
function percentilebootmedian(v; N = NBOOT)
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
