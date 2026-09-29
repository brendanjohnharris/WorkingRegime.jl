# The variability (Fano) exponent, vendored into WRExperiment.
#
# Vendored rather than shared: this package's estimators (`diffusion_fit`, `mapple_fit`, `madev`,
# `fano_factor`) already live in `src/`, and a library `include`d from a scripts directory couples
# every caller to that file's path --- which broke `Fig1_combined_curves.jl` the moment the file
# moved. Each package keeping the part it needs costs a copy and buys independence; the copies are
# small, and the risk they guard against (a path that silently stops resolving) has already fired.
#
# This copy differs from WRCircuit's and WRTheory's by design (2026-09-24): the experimental data use
# the flat-rise fit below, the circuit and theory the floor-bic steepest-segment fit. Only `β` is read
# by this package's pipelines.
#
# Requires TimeseriesTools' OptimExt for MAPPLE's `fit!`, which WRExperiment activates by importing
# both Optim and ForwardDiff.

"""
Multistart restarts are perturbed with the global RNG, so two runs of the same fit on the same curve
disagree (0.206 / 0.214 / 0.198 on the Figure 1 experiment median) unless it is seeded.
[`variability_exponent`](@ref) reseeds on every call.
"""
const VARIABILITY_SEED = 42

"""
Held rather than fitted by [`variability_exponent`](@ref): the shot-noise floor, where `F → 1`
exactly as the counting window shrinks and so is a constant rather than a measurement. The knee
width is pinned too ([`VARIABILITY_WIDTH`](@ref)); the outermost knot needs no pin here, as
TimeseriesTools holds it at the top of the fitted band on its own.
"""
const VARIABILITY_PINS = ["components[1].β" => 0.0]

"""
Knee width in decades, pinned rather than fitted: a free transition width makes (slope × segment
width) a degeneracy ridge, which is what made every free-knot read-out scatter by 0.1--1.0 under
unit-half resampling. Pass `width = nothing` to fit it instead. The value trades one bias for
another --- too sharp and a broad floor-to-scaling crossover is bought as a second straight segment,
too wide and the crossover eats the scaling band.
"""
const VARIABILITY_WIDTH = 0.1

"""
    goodunit(u) -> Bool

Unit-quality filter for the Fano-factor analyses, applied to a unit-table row `u`: sorted as `"good"`
(not a noise cluster), present for more than 90% of the recording (`presence_ratio > 0.9`) and with
`isi_violations < 0.5`. There is no amplitude cutoff: with it (the full Allen filter) too few L2/3
units survive to leave most (session, area) cells their 5-unit minimum. A metric absent from a
cohort's unit table is not applied. Low-quality units flatten the Fano curve at long windows, so
the filter raises `c` (L2/3 flat-rise medians ~0.14-0.19 unfiltered, ~0.19-0.25 filtered).
"""
function goodunit(u)
    q(k, default) = hasproperty(u, k) ? coalesce(getproperty(u, k), default) : default
    return q(:quality, "good") == "good" && q(:presence_ratio, 1.0) > 0.9 && q(:isi_violations, 0.0) < 0.5
end

"""
What [`variability_exponent`](@ref) returns when no candidate fits, so a caller that skips a curve
can supply the same shape rather than build its own sentinel and drift from this one.
"""
const VARIABILITY_NOFIT = (;
    β = NaN, lo = NaN, hi = NaN, ncomponents = 0, βs = Float64[], censored = false
)

"""
    variability_exponent(ff; kwargs...) -> (; β, lo, hi, ncomponents, βs, censored)
    variability_exponent(t, vals; kwargs...)

Variability exponent of a Fano-factor curve `ff` (or lags `t` and values `vals`): `β`, the log--log
slope of a single rise from the shot-noise floor to the end of the window, over `lo .. hi` in the
curve's own time units.

The model is a two-component MAPPLE fit, a floor pinned flat ([`VARIABILITY_PINS`](@ref)) and one free
rise; `lo` is the fitted knee and `hi` the end of the window, so `censored` is always `true`. Where a
curve rises in two stages (the experimental median: +0.10 over 13-60 ms, then +0.21 above), `β` is the
overall slope of both rather than either stage. That is deliberate: a read-out that picks one segment
(the steepest, or the first) reads different timescales in different areas, since which stage is
steeper varies by area, and so is not comparable across the hierarchy (in L2/3 the floor-bic fit read
its top segment in 2% of VISp cells against 28-43% elsewhere).

Fit aggregated curves (medians across units, then medians of exponents across sessions). Extra
`kwargs` (e.g. `time_limit`) go to the refinement.
"""
function variability_exponent(ff; kwargs...)
    m, y = variability_model(ff; kwargs...)
    return variability_exponent(m, y)
end

variability_exponent(::Nothing, y) = VARIABILITY_NOFIT

"Read the exponent and band off an already-fitted model, so a diagnostic holding one need not refit."
function variability_exponent(m::MAPPLE, y)
    βs, bps = betas(m), exp10.(breakpoints(m))
    # The outermost knot is held at the top of the fitted band, so `hi` is where the window ends.
    return (; β = βs[2], lo = bps[1], hi = bps[2], ncomponents = length(βs), βs, censored = true)
end

variability_exponent(t, vals; kwargs...) = variability_exponent(Timeseries(vals, t); kwargs...)

"""
    variability_model(ff; kwargs...) -> (model, curve) | (nothing, curve)

The flat-rise MAPPLE model behind [`variability_exponent`](@ref), with the NaN-stripped curve it was
fit to. Separate so a diagnostic can score the fitted curve against the data without refitting it
under slightly different settings, which is how earlier copies of this estimator drifted apart.
"""
function variability_model(
        ff; seed = VARIABILITY_SEED, multistart = 4, width = VARIABILITY_WIDTH, refine...
    )
    k = findall(!isnan, parent(ff)) # scattered all-unit-NaN lags survive a median; NaN grinds Optim
    y = Timeseries(collect(parent(ff))[k], collect(lookup(ff, 1))[k])
    length(k) < length(ff) ÷ 2 && return (nothing, y)
    pins = isnothing(width) ? VARIABILITY_PINS :
        [VARIABILITY_PINS; "transition_width" => width]
    Random.seed!(seed) # `_perturb` draws multistart restarts from the global RNG
    m = try
        mm = fit(MAPPLE, y; peaks = 0, components = 2)
        fit!(mm, y; fix = pins, multistart, refine...)
        mm
    catch
        nothing # a curve that will not fit is dropped, not fatal to a sweep over many
    end
    return (m, y)
end
