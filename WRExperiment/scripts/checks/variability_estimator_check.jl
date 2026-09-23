#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Diagnostic (not in the paper): validate `variability_exponent` (WRExperiment/src/Variability.jl)
# on both arms it has to serve. Checks the four things a Fano-curve estimator can quietly get wrong:
# which component count BIC selects, whether the fitted knots land inside the data (an upper knot on
# the last sample means the band is the recording length, not a measurement), split-half stability
# (odd/even sessions for the experiment, odd/even neurons for the circuit), and reproducibility
# across repeated calls. Also prints the rolling local log-log slope of each curve, which is what the
# fitted β is a chord of. Writes plots/checks/variability_estimator.txt.
using DrWatson
@quickactivate "WRExperiment"
import WRExperiment: variability_exponent, variability_model, VARIABILITY_PINS,
    VARIABILITY_WIDTH, VARIABILITY_SEED, rootdatadir
using JLD2
using DataFrames
using TimeseriesTools
using Optim, ForwardDiff # TimeseriesTools' OptimExt: MAPPLE refinement and its `fix` keyword
using Statistics
using Printf


const OUTDIR = plotsdir("checks")
const SPLIT_TOLERANCE = 0.03 # the scatter that ruled out every free-knot read-out
mkpath(OUTDIR)

# ──────────────────────────────────────────────────────────────────────────────
# Curves
# ──────────────────────────────────────────────────────────────────────────────

"""
Per-session unit-median Fano curves for VISp L2/3, rebuilt exactly as collect_calculations.jl
builds the drawn Fig 1 median (median across units, then across sessions, over 1-1000 ms).
"""
function experiment_sessions()
    ud = jldopen(
        f -> f["fano_data"], rootdatadir("WRExperiment.jld2"),
        "r"; typemap = toolsarray_typemap
    )["spontaneous"]["unitdepths"][1] # [1] is VISp
    curves = map(ud) do units
        u = subset(units, :layer => ByRow(==(2)))
        cs = [c for c in u.fano_factor if c isa AbstractVector] # a unitless unit stores scalar NaN
        isempty(cs) && return nothing
        (collect(times(first(cs))), vec(median(reduce(hcat, collect.(cs)); dims = 2)))
    end
    return filter(!isnothing, curves)
end

"Median across a set of session curves, over the 1-1000 ms window the figure draws."
function session_median(sessions)
    t = first(first(sessions))
    k = findall(x -> 1.0 ≤ x ≤ 1000.0, t)
    Timeseries(vec(median(view(reduce(hcat, [s[2] for s in sessions]), k, :); dims = 2)), t[k])
end

@info "Loading experiment unit curves (the slow step)"
const SESSIONS = experiment_sessions()
@info "  $(length(SESSIONS)) VISp L2/3 sessions"
@info "Loading circuit per-neuron Fano curves"
const CFANO = jldopen(
    f -> f["fano"], joinpath(dirname(rootdatadir()), "WRCircuit", "demo_run_stats.jld2"),
    "r"; typemap = toolsarray_typemap
)
neuron_median(js) = dropdims(median(CFANO[:, js], dims = 2), dims = 2)

const CURVES = [
    ("experiment", "all sessions") => session_median(SESSIONS),
    ("experiment", "odd sessions") => session_median(SESSIONS[1:2:end]),
    ("experiment", "even sessions") => session_median(SESSIONS[2:2:end]),
    ("circuit", "all neurons") => neuron_median(1:size(CFANO, 2)),
    ("circuit", "odd neurons") => neuron_median(1:2:size(CFANO, 2)),
    ("circuit", "even neurons") => neuron_median(2:2:size(CFANO, 2)),
]

"Rolling OLS log-log slope over a ±`half` decade window: what the fitted β is a chord of."
function local_slopes(y, targets; half = 0.25)
    t, v = log10.(lookup(y, 1)), log10.(parent(y))
    map(targets) do target
        i = argmin(abs.(exp10.(t) .- target))
        k = findall(j -> abs(t[j] - t[i]) ≤ half, eachindex(t))
        length(k) < 5 && return (exp10(t[i]), NaN)
        x, z = t[k], v[k]
        (exp10(t[i]), sum((x .- mean(x)) .* (z .- mean(z))) / sum((x .- mean(x)) .^ 2))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────────────

models = Dict(k => variability_model(y) for (k, y) in CURVES)   # one fit per curve
fits = Dict(k => variability_exponent(v...) for (k, v) in models)

"Worst |log10 model - log10 data| above `hi`: whether the selected top segment tracks the tail."
function tail_error(k)
    m, y = models[k]
    isnothing(m) && return NaN
    t = lookup(y, 1)
    above = findall(>(fits[k].hi), t)
    isempty(above) && return 0.0
    maximum(abs.(log10.(parent(y)[above]) .- log10.(predict(m, t)[above])))
end
failures = String[]
fail(msg) = (push!(failures, msg); msg)

open(joinpath(OUTDIR, "variability_estimator.txt"), "w") do io
    println(io, "variability_exponent: BIC-selected floor-plus-free-segments MAPPLE fit")
    println(io, "pins = $(VARIABILITY_PINS), width = $VARIABILITY_WIDTH, seed = $VARIABILITY_SEED\n")

    println(io, "=== rolling local log-log slope (±0.25 decade) ===")
    for (arm, curve) in (("experiment", CURVES[1][2]), ("circuit", CURVES[4][2]))
        s = local_slopes(curve, (3.0, 10.0, 30.0, 60.0, 100.0, 200.0, 400.0, 700.0))
        println(io, "  $arm: ", join((@sprintf("%.0fms %+.3f", t, b) for (t, b) in s), "  "))
    end

    println(io, "\n=== fits ===")
    for ((arm, half), y) in CURVES
        r = fits[(arm, half)]
        t = lookup(y, 1)
        # `censored` means the reported segment is the topmost, so `hi` is the window end by
        # construction. Only a FITTED knot resting on the last sample is the failure this checks for.
        inside = r.censored || r.hi < 0.98last(t)
        above = count(>(r.hi), t) # samples carrying the selected top segment
        err = tail_error((arm, half))
        @printf(
            io, "  %-10s %-14s β = %+.4f  band = [%.1f, %.1f%s]  βs = [%s]  n>hi = %-3d  R² = %.4f  tail err = %.4f dex  %s\n",
            arm, half, r.β, r.lo, r.hi, r.censored ? "+" : "",
            join((@sprintf("%+.2f", b) for b in r.βs), ", "), above,
            rsquared(models[(arm, half)][1], models[(arm, half)][2]), err,
            inside ? "" : fail("$arm/$half: upper knot on the data edge")
        )
        # The model must actually track the data above the band; a large tail error means the top
        # segment is absorbing a windowing artifact rather than describing the curve.
        err > 0.02 && fail(@sprintf("%s/%s: tail error %.3f dex above hi", arm, half, err))
        r.lo ≤ 1.02first(t) && fail("$arm/$half: lower knot on the data edge")
        # A top segment that is neither flat nor carried by real data is the fit chasing the
        # terminal samples, which is the failure the plateau pin was hiding rather than fixing.
        abs(r.βs[end]) > 0.5 && count(>(exp10(breakpoints(models[(arm, half)][1])[end - 1])), t) < 10 &&
            fail(@sprintf("%s/%s: top segment β = %+.2f over too few samples", arm, half, r.βs[end]))
    end

    println(io, "\n=== split-half |Δβ| (reject above $SPLIT_TOLERANCE) ===")
    for (arm, a, b) in
        (("experiment", "odd sessions", "even sessions"), ("circuit", "odd neurons", "even neurons"))
        Δ = abs(fits[(arm, a)].β - fits[(arm, b)].β)
        @printf(io, "  %-10s %+.4f / %+.4f   |Δ| = %.4f  %s\n", arm, fits[(arm, a)].β,
            fits[(arm, b)].β, Δ, Δ ≤ SPLIT_TOLERANCE ? "" : fail("$arm: split-half |Δβ| = $Δ"))
    end

    println(io, "\n=== reproducibility (repeat calls must agree exactly) ===")
    for ((arm, half), y) in CURVES[[1, 4]]
        again = variability_exponent(variability_model(y)...)
        same = again == fits[(arm, half)]
        @printf(io, "  %-10s %+.6f vs %+.6f  %s\n", arm, fits[(arm, half)].β, again.β,
            same ? "identical" : fail("$arm: repeat call disagrees"))
    end

    println(io, "\n", isempty(failures) ? "PASS" : "FAIL\n  " * join(failures, "\n  "))
end

print(read(joinpath(OUTDIR, "variability_estimator.txt"), String))
isempty(failures) || @warn "variability_exponent check FAILED" failures
