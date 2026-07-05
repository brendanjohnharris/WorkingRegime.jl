using Statistics
using TimeseriesTools
using TimeseriesBase: spiketrain
using DimensionalData
using IntervalSets
using Unitful
import Dewdrop
using Dewdrop: solve, FixedStep, Trace, Spikes, MADev, Welch, SpikeRate, Fano

export simulate, simulate_batch, bpformat, bpsolve, Neuron, Population, Monitor, Spatial, PRNGKey,
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
_perpop_record(populations, vars::AbstractVector) =
    NamedTuple(_perpop_key(p, v) => _monitor_spec(v, p) for p in populations for v in vars)
# Per-population vars: pass `vars` as a NamedTuple mapping each population to its own variable list,
# so one run can record e.g. full traces for :E but only spikes for :I --- keeping recording memory
# O(what you asked for) rather than O(all populations × all vars). `simulate` forwards `vars`
# unchanged, so `simulate(m, t; populations=[:E,:I], vars=(; E=[:spike,:V,:input], I=[:spike]))` works.
_perpop_record(populations, vars::NamedTuple) =
    NamedTuple(_perpop_key(p, v) => _monitor_spec(v, p) for p in populations for v in vars[p])

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

# --- Batched sweep over (delta, Delta_g_K) at a fixed connectome ---------------------------------
"""
    simulate_batch(model, time, deltas, dgks; dt=0.1, progress=:auto, arch=DEWDROP_BACKEND(), kwargs...) -> Dewdrop.BatchedSolution

Co-execute `B = length(deltas)` members of `model` over ONE shared connectome (fixed seed + `sigma_ee`),
sweeping `delta` (`deltas[m]`) and `Delta_g_K` (`dgks[m]`) per member through Dewdrop's `(N,B)` batched
ensemble --- memory `O(edges)` (not `O(B·edges)`), one connectome build, and a kernel that fills the
launch-bound GPU. `Delta_g_K` is an `N×B` neuron-model override (E rows get `dgks[m]`, I rows stay 0);
`delta` is a per-member conductance gain on the inhibitory projections (I→E, I→I) --- the per-edge weight is
linear in `delta`, so scaling the synapse coefficient on a connectome built at `delta = 1` reproduces each
member's `delta` network. Fix the connectome's `seed` (via `model`) and `sigma_ee` (a keyword) per job; sweep
`(delta, Delta_g_K)` here. The streaming Poisson drive is SHARED across members (same realization). Member
`m`'s E `input` (`itot`) trace and `spike` raster are `bs.record.input.data[:, m, :]` /
`bs.record.spike.data[:, m, :]` (a `(NE, B, nsteps)` array each). Note: each E `input` trace is
`NE × nsteps × sizeof(T)`, so bound `B` (the chunk size) by memory.

Pass any of the following to record streaming ON-DEVICE statistics instead of the raw trace/raster --- only
the small reduced result is kept (`O(NE·nstat)`), so the long signals are never materialised (host or device),
and host memory stops scaling with `nsteps`:
- `mad_lags` (integer step lags) → `bs.record.mad.data` `(NE, B, nlags)`, the [`Dewdrop.MADev`](@ref) input MAD;
- `psd_fmin` (min frequency) → `bs.record.psd.data` `(NE, B, nfreq)`, the [`Dewdrop.Welch`](@ref) input power spectrum;
- `fano_taus` (timescales) → `bs.record.fano.data` `(NE, B, ntau)`, the [`Dewdrop.Fano`](@ref) spike Fano-factor curve;
- `rate = true` → `bs.record.rate.data` `(NE, B)`, the [`Dewdrop.SpikeRate`](@ref) per-neuron mean firing rate.
`transient` (recorded steps) is dropped by the reductions. The full spike raster is kept only with
`record_spikes = true` (default: only in the no-reduction fallback). `scatter` is forwarded to `solve`
(`:auto`, or `:compacted` for sparse firing over a large connectome).
"""
function simulate_batch(
        model::SpatialModel, time, deltas::AbstractVector{<:Real}, dgks::AbstractVector{<:Real};
        dt = 0.1, progress = :auto, arch::Dewdrop.AbstractArchitecture = DEWDROP_BACKEND(),
        mad_lags = nothing, psd_fmin = nothing, fano_taus = nothing, rate = false,
        transient = 0, scatter = :auto, record_spikes = nothing, kwargs...
    )
    B = length(deltas)
    length(dgks) == B || throw(ArgumentError("deltas and dgks must be equal length (got $B and $(length(dgks)))"))
    tmax_ms = ustrip(to_ms(time))
    delta0 = 1.0    # connectome built at unit delta; per-member delta is a synapse `a` gain (linear in weight)
    bparams = merge((; model.params...), (; kwargs...), (; delta = delta0))   # force the unit-delta connectome
    spec = build_spatial(; bparams..., seed = model.seed, arch = arch)
    net = Dewdrop.materialize(spec, FixedStep(dt); tspan = (0.0, tmax_ms))
    N = net.n
    Erange = net.subpops[:E]
    T = Dewdrop.float_type(net.model)
    # Δg_K: per-(neuron, member) override --- E rows get the member's value, I rows stay 0 (the E/I split).
    ΔgK = T[(i in Erange) ? dgks[m] : 0.0 for i in 1:N, m in 1:B]
    # delta: per-member conductance gain on the inhibitory projections (I→E = 3, I→I = 4 in build order).
    inh = net.projections[3].synapse
    inh isa Dewdrop.FrozenDualExpSynapse ||
        error("simulate_batch: expected the inhibitory FrozenDualExpSynapse at projection 3 (got $(typeof(inh)))")
    a_vec = T[Dewdrop._dualexp_a(inh.τr, inh.τd) * deltas[m] / delta0 for m in 1:B]
    syn_over = Dict(3 => (; a = a_vec), 4 => (; a = a_vec))
    of = collect(Erange)
    tr = Int(transient)
    # Recording. Each statistic is OPT-IN via its parameter and streamed ON-DEVICE (the long trace/raster is
    # never materialised --- host or device --- only the small reduced result is kept; `transient` recorded
    # steps are dropped):
    #   `mad_lags` → MADev (E `itot` mean-absolute-displacement at integer step lags)
    #   `psd_fmin` → Welch  (E `itot` power spectrum, min resolved frequency `psd_fmin`)
    #   `fano_taus` → Fano  (E spike-count Fano-factor curve at timescales `fano_taus`)
    #   `rate = true` → SpikeRate (E mean firing rate per neuron)
    # If none are requested, fall back to the raw E `itot` trace. The full spike raster is recorded only when
    # `record_spikes = true` (default `nothing` → only in the raw fallback, since the reductions replace it).
    # `scatter` is forwarded to `solve` (`:auto` picks edge/compacted; pass `:compacted` for sparse firing).
    reduced = !(mad_lags === nothing && psd_fmin === nothing && fano_taus === nothing) || rate
    rec = reduced ? (;) : (; input = Trace(:itot; of = of))
    mad_lags === nothing || (rec = merge(rec, (; mad = MADev(:itot; of = of, lags = mad_lags, transient = tr))))
    psd_fmin === nothing || (rec = merge(rec, (; psd = Welch(:itot; of = of, f_min = psd_fmin, transient = tr))))
    fano_taus === nothing || (rec = merge(rec, (; fano = Fano(; of = of, taus = fano_taus, transient = tr))))
    rate && (rec = merge(rec, (; rate = SpikeRate(; of = of, transient = tr))))
    (record_spikes === nothing ? !reduced : record_spikes) && (rec = merge(rec, (; spike = Spikes(of = of))))
    return solve(
        net, FixedStep(dt); batch = B, v0 = (-70.0, -50.0),
        model_overrides = (; ΔgK = ΔgK), syn_overrides = syn_over,
        record = rec, scatter = scatter, progress = progress,
    )
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
