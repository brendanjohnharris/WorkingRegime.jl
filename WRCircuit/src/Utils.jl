using Bootstrap
using Normalization
using StatsBase
using Random
import Accessors: @set

export connector, bootstrapmedian, to_ms

function _preamble()
    return quote
        using DrWatson
        using Unitful
        using Statistics
        using Bootstrap
        using TimeseriesTools
        using Makie
        using TimeseriesMakie
        using Fathom
        using LinearAlgebra
        using Distributed
        using Term
        using MoreMaps
        using SparseArrays
        using MeanSquaredDisplacement
        using Distributions
        using StableDistributions
        using JLD2
        using Random
    end
end
macro preamble()
    return _preamble()
end
@preamble

const connector = '&'

to_ms(x::Real) = x * u"ms" # Assume ms already
to_ms(x::Quantity) = uconvert(u"ms", x)

function convert2(u::Unitful.Units, x::AbstractRange)
    a = uconvert(u, first(x))
    b = uconvert(u, last(x))
    return range(a, b, length = length(x))
end

# ──────────────────────────────────────────────────────────────────────────────
# Per-neuron exponent fits (MAPPLE)
# ──────────────────────────────────────────────────────────────────────────────

export per_neuron, diffusion_exponents, diffusion_knees, spectral_exponents

# Band over which the diffusion exponent is fit, IN MILLISECONDS (the circuit's MAD lag axis is in
# ms; WRExperiment's is in seconds, so the numeric values there and here differ by 1000x even though
# they denote the SAME physical band).
#
# 0.8-8 ms: the first decade of the *experimental* lag grid, from one LFP sample at 1250 Hz to a
# decade later. The circuit is fit over the identical physical band so the two exponents are the
# same quantity; its own grid starts at 1 ms (10 steps at dt = 0.1 ms), so in practice it
# contributes 1-8 ms and the lower edge never binds. Roughly 30 points here against the data's 10.
const MAD_BAND_MS = [0.0, 8.0]

"""
    per_neuron(f, data; step = 1)

Apply the per-column fit `f` to every `step`-th column (neuron) of `data`,
returning `NaN` for any column whose fit throws. `step = 1` fits every neuron.
"""
function per_neuron(f, data; step = 1)
    return map(eachcol(data[:, 1:step:end])) do y
        try
            return f(y)
        catch e
            @warn "Failed neuron fit" e
            return NaN
        end
    end
end

"""
    diffusion_exponents(mad; step = 1, band = MAD_BAND_MS)

Per-neuron diffusion exponents: the slope of a 1-component MAPPLE fit to each neuron's mean
absolute displacement (MAD) curve, restricted to `band` (milliseconds). One value per fitted neuron.

One component means the model is a single power law, so `β` IS the log-log slope over the band.
The previous 2-component form returned `first(β)`, the τ→0 asymptote of a segment that blends with
its neighbour through a tanh crossfade; on curves where the fit chooses a wide transition that
asymptote is not a slope the data exhibits anywhere. Restricting to the pre-knee decade removes
both the breakpoint and the two saturated decades that carry no exponent. Mirrors
`WRExperiment.diffusion_fit` exactly, so the circuit and data exponents are the same quantity.
Use [`diffusion_knees`](@ref) to confirm the band ends before the knee.
"""
function diffusion_exponents(mad; step = 1, band = MAD_BAND_MS)
    return per_neuron(mad; step) do col
        y = ustripall(col)[𝑡 = band[1] .. band[2]]
        m = fit(MAPPLE, y; components = 1, peaks = 0)
        fit!(m, y; w = true)
        first(m.params.components.β)
    end
end

"""
    diffusion_knees(mad; step = 1)

Per-neuron crossover lag (ms) where the MAD curve leaves its scaling regime: the first breakpoint
of an unconstrained 2-component MAPPLE fit over the whole curve. Fit separately from
[`diffusion_exponents`](@ref) and used only to locate the knee, never to read an exponent off.
"""
function diffusion_knees(mad; step = 1)
    return per_neuron(mad; step) do col
        y = ustripall(col)
        m = fit(MAPPLE, y; components = 2, peaks = 0)
        fit!(m, y; w = true)
        first(breakfrequencies(m))
    end
end

"""
    spectral_exponents(psd; step = 1)

Per-neuron spectral exponents: the aperiodic exponent (the single component) of a
1-component, 1-peak MAPPLE fit to each neuron's 10-1000 Hz power spectral density
(PSD). The Gaussian peak absorbs the ~50 Hz oscillation so it does not bias the
aperiodic slope. One value per fitted neuron.
"""
function spectral_exponents(psd; step = 1)
    return per_neuron(psd; step) do col
        p = logsample(ustripall(col[𝑓 = 10u"Hz" .. 1000u"Hz"]))
        m = fit(MAPPLE, p; components = 1, peaks = 1)
        fit!(m, p)
        last(m.params.components.β)
    end
end
