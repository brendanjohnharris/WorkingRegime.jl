# The variability (Fano) exponent, vendored into WRCircuit.
#
# Vendored rather than shared: a library `include`d from the repository's `scripts/` directory
# couples every caller to that file's path, which breaks silently the moment a script moves. Each
# package keeping its own copy costs a duplicate and buys independence.
#
# Keep the copies in step if the estimator itself changes; the others are in
# WRExperiment/src/Variability.jl and WRTheory/src/Variability.jl. Only the header differs between them.
#
# Requires TimeseriesTools' OptimExt for MAPPLE's `fit!`, which is active whenever Optim and
# ForwardDiff are loaded.
using TimeseriesTools
using Random
import Optim, ForwardDiff  # activate TimeseriesTools' OptimExt (MAPPLE fit!)

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
What [`variability_exponent`](@ref) returns when no candidate fits, so a caller that skips a curve
can supply the same shape rather than build its own sentinel and drift from this one.
"""
const VARIABILITY_NOFIT = (;
    β = NaN, lo = NaN, hi = NaN, ncomponents = 0, βs = Float64[], censored = false
)

"""
    variability_exponent(ff; kwargs...) -> (; β, lo, hi, ncomponents, βs, censored)
    variability_exponent(t, vals; kwargs...)

Unified variability exponent of a Fano-factor curve `ff` (or lags `t` and values `vals`): `β`, the
log--log slope of the scaling segment, over the fitted band `lo .. hi` in the curve's own time units.

The model is a MAPPLE fit whose first component is pinned flat ([`VARIABILITY_PINS`](@ref)) and whose
remaining segments are free, with the component count chosen by BIC. Two components give a floor and
a rise that runs to the end of the window; three add a free segment above the scaling band, which is
how a saturating curve is expressed. Only the floor is imposed, so a curve that saturates and one
that does not are both fit by the same estimator rather than by two --- pinning the outer segment
flat instead forces a plateau onto a curve that has none, and on the experimental median it drove the
upper knot onto the last sample, where the quoted band was the recording length rather than a
measurement.

`β` is the STEEPEST rising segment, and `lo .. hi` is that segment's own span. One rule serves both
shapes: where a plateau sits above the rise the steepest segment IS the rise, and where the curve
instead rises in two stages (the experimental median: +0.10 over 13-60 ms, then +0.21 above, at every
knee width tried including a free one) it is the asymptotic stage rather than the floor crossover.
`censored` marks that the reported segment is the topmost, so its `hi` is where the fitted window
ends rather than a measured crossover --- the scaling continues to at least there.

Fit AGGREGATED curves only: per-unit and per-repeat curves have no SNR for a free-knot fit (per-neuron
circuit read-outs median 0.78 against 0.28 for the neuron-median curve), so take medians of curves
first and medians of exponents across sessions after. Extra `kwargs` (e.g. `time_limit`) go to each
candidate's refinement.
"""
function variability_exponent(ff; kwargs...)
    m, y = variability_model(ff; kwargs...)
    return variability_exponent(m, y)
end

variability_exponent(::Nothing, y) = VARIABILITY_NOFIT

"Read the exponent and band off an already-selected model, so a diagnostic holding one need not refit."
function variability_exponent(m::MAPPLE, y)
    βs, bps = betas(m), exp10.(breakpoints(m))
    i = argmax(view(βs, 2:length(βs))) + 1 # steepest rising segment; the floor is pinned flat
    # TimeseriesTools holds the outermost knot at the top of the fitted band, so for the topmost
    # segment `hi` is where the window ends rather than a measured crossover: flag it `censored`.
    return (;
        β = βs[i], lo = bps[i - 1], hi = bps[i],
        ncomponents = length(βs), βs, censored = i == length(βs)
    )
end

variability_exponent(t, vals; kwargs...) = variability_exponent(Timeseries(vals, t); kwargs...)

"""
    variability_model(ff; kwargs...) -> (model, curve) | (nothing, curve)

The selected MAPPLE model behind [`variability_exponent`](@ref), with the NaN-stripped curve it was
fit to. Separate so a diagnostic can score the fitted curve against the data without refitting it
under slightly different settings, which is how earlier copies of this estimator drifted apart.
"""
function variability_model(
        ff; seed = VARIABILITY_SEED, max_components = 3, multistart = 4,
        width = VARIABILITY_WIDTH, refine...
    )
    k = findall(!isnan, parent(ff)) # scattered all-unit-NaN lags survive a median; NaN grinds Optim
    y = Timeseries(collect(parent(ff))[k], collect(lookup(ff, 1))[k])
    length(k) < length(ff) ÷ 2 && return (nothing, y)
    pins = isnothing(width) ? VARIABILITY_PINS :
        [VARIABILITY_PINS; "transition_width" => width]
    Random.seed!(seed) # `_perturb` draws multistart restarts from the global RNG
    m = try
        fit(
            MAPPLE, y; peaks = 0, components = :auto, max_components,
            refine = (; fix = pins, multistart, refine...)
        )
    catch
        nothing # a curve no candidate will fit is dropped, not fatal to a sweep over many
    end
    return (m, y)
end
