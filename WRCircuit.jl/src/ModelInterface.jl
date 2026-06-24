using Statistics
using TimeseriesTools
using TimeseriesBase: spiketrain
using DimensionalData
using IntervalSets
using Unitful
import Dewdrop
using Dewdrop: solve, FixedStep, Trace, Spikes

export simulate, bpformat, bpsolve, Neuron, Population, Monitor, Spatial, PRNGKey,
    models, defaults

# Reuse Dewdrop's TimeseriesBase-extension output dimensions (registered when TimeseriesBase loads),
# rather than re-declaring `Neuron`/`Population` --- a second `@dim` of those names would clash with the
# ext's during precompilation ("method overwriting"). `basetypeof ∘ name2dim` recovers the constructable
# dim type by name (the ext's `ToolsDim` types; a generic `Dim{:name}` if the ext is not present, which
# still indexes by name). `Var`/`𝑡` come from TimeseriesBase; `Monitor` is WRCircuit-only (no ext
# counterpart, so a plain `@dim` is safe).
const Neuron = DimensionalData.basetypeof(DimensionalData.name2dim(Val(:Neuron)))
const Population = DimensionalData.basetypeof(DimensionalData.name2dim(Val(:Population)))
DimensionalData.@dim Monitor ToolsDim

# Row-major (Python/BrainPy) flat → 2D reshape: reverse the dims then permute, so a flattened
# `(n, n)` neuron list lays out the same way it did under the BrainPy backend (used by `Plots.infer_geometry`).
function python_reshape(A, ds...)
    return PermutedDimsArray(reshape(A, ds), reverse(1:length(ds)))
end

popvars2monitors(populations, vars) =
    Tuple("$p.$v" for (p, v) in collect(Iterators.product(populations, vars))[:])
function monitors2popvars(monitors)
    ps = [(Symbol(p[1]), Symbol(p[2])) for p in split.(monitors, ".")]
    return unique(first.(ps)), unique(last.(ps))
end

# Seeds were JAX PRNG keys; natively a seed is just an integer fed to Dewdrop's counter RNG.
PRNGKey(seed::Integer) = UInt64(seed)
PRNGKey(seed::Unsigned) = UInt64(seed)

# --- The spatial FNS "working-regime" model -----------------------------------------------------
# A thin parameter holder. The heavy network --- connectome, weights, geometry, external drive --- is
# assembled by `build_spatial` at run time (`simulate`/`bpsolve`), since the run's
# `dt`/`tspan` set every delay. Geometry is no longer duplicated here: the simulated E/I positions and
# subpopulation ranges are read back from the solved network (`sol[:E].positions`, `sol.subpops`), which
# is the single source of truth (the old holder re-derived them from `Dewdrop._grid_centered`/`_subseed`,
# which silently diverges if the engine's geometry changes).
struct SpatialModel{NT}
    params::NT        # build_spatial kwargs (rho, dx, gamma, sigma_*, K_*, nu, n_ext, Delta_g_K, …; NOT tspan/dt/seed)
    seed::UInt64
end

# The full `build_spatial` defaults (mirrors `build_spatial`'s keyword signature). The single place a
# script reads the canonical working-regime point (`defaults(models.Spatial)[:delta]`, …) before sweeping.
const SPATIAL_DEFAULTS = (;
    rho = 20000, dx = 0.5, gamma = 4,
    sigma_ee = 0.06, sigma_ei = 0.07, sigma_ie = 0.14, sigma_ii = 0.14,
    K_ee = 260, K_ei = 340, K_ie = 225, K_ii = 290,
    delta = 4.0, J_ee = 0.00105, J_ei = 0.00145, nu = 10.0, n_ext = 100,
    tau_r_e = 1.0, tau_r_i = 2.0, tau_d_e = 5.0, tau_d_i = 4.5,
    V_rev_e = 0.0, V_rev_i = -80.0, e_delay = 1.5, i_delay = 1.5,
    Delta_g_K = 0.002, tau_K = 40.0,
)

"""
    Spatial(; key=nothing, kwargs...)

The spatial FNS E/I working-regime network (the model the BrainPy `Spatial` class built, now native).
A thin holder of its `build_spatial` parameters plus the reproducibility seed; `simulate`/`bpsolve`
assemble and run it through `build_spatial`. `key` is the seed (a [`PRNGKey`](@ref) or
any integer). All other keywords (`rho`, `dx`, `gamma`, `sigma_ee`, `K_ee`, `nu`, `n_ext`, `Delta_g_K`,
`delta`, …) pass straight through to `build_spatial`; unset ones take its defaults
(see [`defaults`](@ref)). The simulated geometry is read back from the solution (`sol[:E].positions`).
"""
function Spatial(; key = nothing, kwargs...)
    seed = key === nothing ? 0x05fd % UInt64 : UInt64(key)
    return SpatialModel((; kwargs...), seed)
end

# `models.Spatial` namespace (mirrors the old `WRCircuit.models.Spatial` access).
const models = (; Spatial = Spatial)

"""
    defaults(models.Spatial) -> NamedTuple

The canonical `build_spatial` parameter defaults (the working-regime point). Use to read a baseline value
before sweeping around it, e.g. `defaults(models.Spatial)[:delta]`.
"""
defaults(::typeof(Spatial)) = SPATIAL_DEFAULTS
defaults(::SpatialModel) = SPATIAL_DEFAULTS

function Base.show(io::IO, ::MIME"text/plain", m::SpatialModel)
    print(io, "WRCircuit Spatial model (seed = ", repr(m.seed))
    isempty(m.params) || print(io, "; ", join(("$k = $v" for (k, v) in pairs(m.params)), ", "))
    print(io, ")")
    # render the cheap, unmaterialised spec tree (populations + projections; no connectome built)
    try
        print(io, "\n")
        show(io, MIME"text/plain"(), build_spatial(; m.params..., seed = m.seed))
    catch e
        print(io, "\n  (spec preview unavailable: ", sprint(showerror, e), ")")
    end
    return nothing
end
Base.show(io::IO, m::SpatialModel) =
    print(
    io, "Spatial(", join(("$k = $v" for (k, v) in pairs(m.params)), ", "),
    isempty(m.params) ? "" : "; ", "seed = ", repr(m.seed), ")"
)

# --- Recording specs ----------------------------------------------------------------------------
# Map a requested variable to the Dewdrop state/accumulator it records. `:input` is the total synaptic
# input current (Dewdrop's `:itot` accumulator --- the same quantity BrainPy monitored as `<pop>.input`);
# `:gK`/`:w` the adaptation conductance; `:spike` is a `Spikes()` monitor (handled separately).
_dwvar(v::Symbol) = v === :input ? :itot : (v === :gK || v === :w) ? :w : v
_monitor_spec(v::Symbol, of) = v === :spike ? Spikes(of = of) : Trace(_dwvar(v); of = of)

# Per-(population, variable) monitors recorded only over the requested subpopulations (`of = :E`), keyed
# `Symbol(p, :_, dwvar)` --- matches the old memory profile (records E only when `populations = [:E]`).
_perpop_key(p::Symbol, v::Symbol) = Symbol(p, :_, v === :spike ? :spikes : _dwvar(v))
_perpop_record(populations, vars) =
    NamedTuple(_perpop_key(p, v) => _monitor_spec(v, p) for p in populations for v in vars)

# --- Run (raw solution) -------------------------------------------------------------------------
"""
    simulate(model, time; populations=[:E,:I], vars=[:V], dt=0.1, progress=:auto, arch=DEWDROP_BACKEND(), kwargs...)

Assemble `model` (a [`Spatial`](@ref)) over `(0, time)` and run it on the native Dewdrop engine, returning
the raw `DewdropSolution`. `time` accepts a unitful `Quantity` or a plain `Real` (ms); `dt` (ms) is the
integration step. The solution carries the named subpopulations (`sol[:E]`, `sol[:I]` --- each with its
simulated `.positions`) and the recorded monitors, so you can read native observables directly
(`susceptibility(sol)`, `radial_autocorrelation(sol)`, [`stats`](@ref)) or shape the BrainPy-style output
with [`bpformat`](@ref). `progress` is forwarded to `solve` (`:auto`/`true`/`false`/a name/Int); extra
keywords (`rho`, `J_ee`, `nu`, …) pass through to `build_spatial` as parameter overrides.
"""
function simulate(
        model::SpatialModel, time; populations = [:E, :I], vars = [:V], dt = 0.1,
        progress = :auto, arch::Dewdrop.AbstractArchitecture = DEWDROP_BACKEND(), kwargs...
    )
    tmax_ms = ustrip(to_ms(time))
    spec = build_spatial(; model.params..., seed = model.seed, arch = arch, kwargs...)   # unmaterialised spec
    record = _perpop_record(populations, vars)
    # materialise (build the connectome) + run, injecting the real run window `tspan`
    return solve(spec, FixedStep(dt); tspan = (0.0, tmax_ms), v0 = (-70.0, -50.0), record = record, progress = progress)
end

# --- Format (BrainPy-identical output) ----------------------------------------------------------
# Stamp the established byte-compatible axes onto a `step × neuron` data matrix: the `𝑡` axis in `ms` (so
# downstream `firingrate`/`coarsegrain` see unitful time), per-population `:E1, :E2, …` neuron labels, and
# a transient slice. Bool data becomes a `SpikeTrain` (a `Timeseries` of Bool) automatically. `times_ms`
# is the per-column time in ms (numbers); the same shim serves the single-run and batched paths.
function _label_cell(data_tn::AbstractMatrix, p::Symbol, times_ms::AbstractVector, transient_ms::Real)
    nrec = size(data_tn, 2)
    t = 𝑡(times_ms .* u"ms")   # range .* scalar stays a StepRangeLen (collect/general-broadcast would materialise)
    labels = Neuron(Symbol.(string(p) .* string.(1:nrec)))
    ts = Timeseries(data_tn, t, labels)
    return @view ts[𝑡(OpenInterval(transient_ms * u"ms", Inf * u"s"))]   # view, not a copy of the kept window
end

# One (population, variable) cell of a single solution, via Dewdrop's native TimeseriesBase ext: the ext
# does the subpop selection + `(neuron × step)`→`(step × neuron)` transpose (and any `every` subsampling);
# `_label_cell` then restores the byte-compatible axes. `of = :all` keeps every recorded row (we recorded
# exactly population `p` into this monitor).
function _cell(sol, p::Symbol, v::Symbol, transient_ms::Real)
    mon = _perpop_key(p, v)
    base = v === :spike ? permutedims(spiketrain(sol, mon; of = :all)) :  # spiketrain is Neuron×𝑡 → 𝑡×Neuron
        Timeseries(sol, mon; of = :all, lazy = true)                  # lazy: a VIEW over sol (no copy)
    return _label_cell(parent(base), p, parent(lookup(base, 𝑡)), transient_ms)   # raw range, not a materialising broadcast
end

"""
    bpformat(sol; populations=[:E,:I], vars=[:V], transient=0u"ms")

Shape a raw [`simulate`](@ref) solution into the BrainPy-compatible output: a `Population × Var`
`ToolsArray` whose entries are `Timeseries`/`SpikeTrain`s over `(𝑡, Neuron)`, with the `𝑡` axis in `ms`,
`:E1, :E2, …` neuron labels, and the leading `transient` dropped --- byte-identical to the old
`bpsolve`/`bpformat` layout. The traces are extracted with Dewdrop's native `Timeseries(sol, …)` /
`spiketrain(sol, …)` methods.
"""
function bpformat(sol; populations = [:E, :I], vars = [:V], transient = 0u"ms")
    transient_ms = ustrip(to_ms(transient))
    X = [_cell(sol, p, v, transient_ms) for p in populations, v in vars]
    return ToolsArray(X, (Population(collect(populations)), Var(collect(vars))))
end

"""
    bpsolve(model, time; populations=[:E,:I], vars=[:V], transient=0u"ms", dt=0.1, progress=:auto, kwargs...)

Run `model` over `(0, time)` and return the BrainPy-style `Population × Var` output in one call ---
`bpformat(simulate(model, time; …); …)`. The returned array and the saved `.jld2` format are unchanged
from the BrainPy backend. Use [`simulate`](@ref) directly when you also need the raw solution (positions,
native stats). Extra keywords pass through to `build_spatial`.
"""
function bpsolve(
        model::SpatialModel, time; populations = [:E, :I], vars = [:V],
        transient = 0u"ms", dt = 0.1, progress = :auto,
        arch::Dewdrop.AbstractArchitecture = DEWDROP_BACKEND(), kwargs...
    )
    sol = simulate(
        model, time; populations = populations, vars = vars, dt = dt,
        progress = progress, arch = arch, kwargs...
    )
    return bpformat(sol; populations = populations, vars = vars, transient = transient)
end
