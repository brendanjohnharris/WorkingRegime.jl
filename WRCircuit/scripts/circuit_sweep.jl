#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 --handle-signals=yes -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate :WRCircuit
using WRCircuit
using JLD2
using MoreMaps
WRCircuit.@preamble

begin # * Sweep configuration
    B = 32
    seeds = 1:10
    arch = WRCircuit.DEWDROP_BACKEND()
    path = datadir("circuit_sweep")

    defaults = WRCircuit.defaults(WRCircuit.models.Spatial)
    tmax = 35u"s"
    transient = 5u"s"                          # discarded transient; simulations always begin at 0
    dt = 0.1u"ms"
    dt_ms = ustrip(u"ms", dt)
    transient_ms = ustrip(u"ms", transient)

    const mad_lags = unique(round.(Int, logrange(10, 10000, length = 100)))
    const τs = mad_lags .* dt
    const psd_fmin = (1 / dt_ms) / 8192   # nfft = 8192 (power of 2 → small cuFFT workspace; nfft=10000 OOMs from its 5^4 radix). ~1.2 Hz, ~40 log bins in 10-1000 Hz after logsample. Was 0.25 = 250 Hz → only 4 bins → spurious ~-4 exponent.
    const fano_taus = collect(logrange(dt_ms * 10, dt_ms * 1000, length = 200))
    const transient_steps = round(Int, transient_ms / dt_ms)   # transient dropped on-device by the reductions
end

begin # * Sweep axes + the working-regime point each plane pivots around
    delta = round.(range(2.5, 5, length = 51); sigdigits = 3)   # step 0.05; old 3.5..5 hit exactly
    Delta_g_K = round.(range(0, 0.005, length = 26); sigdigits = 3)
    sigma_ee = round.(range(0.03, 0.075, length = 19); sigdigits = 3)
    tau_r_e = round.(range(0.5, 2.0, length = 31); sigdigits = 3)
    tau_d_e = round.(range(4, 6, length = 41); sigdigits = 3)

    # Working-regime defaults (rounded like the axes): `send_sweep` fills any non-axis delta/Delta_g_K/sigma_ee
    # from here when building each batch, so a plane holds its two non-swept siblings at the working point.
    defaults0 = (;
        delta = round(Float64(defaults[:delta]); sigdigits = 3),
        Delta_g_K = round(Float64(defaults[:Delta_g_K]); sigdigits = 3),
        sigma_ee = round(Float64(defaults[:sigma_ee]); sigdigits = 3),
    )

    isdir(path) || mkpath(path)
end

# Shared run helpers: the resume path (defined once, used by both filters and `save_member!`), the fixed
# reduction kwargs, and the run-a-batch-then-save loop, so `send_sweep` stays thin.
savepath(p, seed) = joinpath(path, savename((; p..., seed = seed), "jld2"; connector))
# A result file is complete iff it carries all of these keys. `haskey` is a metadata lookup (nested paths
# resolve), so this validates WITHOUT deserialising the arrays --- ~6 ms/file, ~5 min for a full resume scan.
const REQUIRED_KEYS = ("parameters", "rate", "fano", "inputs/mad", "inputs/psd")
_complete(f) = try
    jldopen(g -> all(k -> haskey(g, k), REQUIRED_KEYS), f, "r")
catch
    false          # unreadable / truncated header
end
# Resume guard: skip a file only if it exists AND is complete. A present-but-truncated file (an interrupted
# `wsave`) is DELETED here so the sweep regenerates it --- bare `isfile` would skip the stub forever.
function saved(p, seed)
    f = savepath(p, seed)
    isfile(f) || return false
    _complete(f) && return true
    @warn "Incomplete sweep file (missing keys) --- deleting to regenerate" file = f
    rm(f; force = true)
    return false
end
const RED = (;
    dt = dt_ms, arch, progress = true, scatter = :compacted,
    mad_lags, psd_fmin, fano_taus, rate = true, transient = transient_steps,
)
function run_and_save!(model, chunk, seed, deltas, dgks; member...)
    bs = simulate_batch(model, tmax, deltas, dgks; member..., RED...)
    for (i, c) in enumerate(chunk)
        save_member!(bs, i, c, seed)
    end
    bs = nothing
    GC.gc()
    return nothing
end

# Rebuild member `i`'s labelled (stat × Neuron) arrays from its on-device-reduced slice and save one result
# file. `c` is the member's varied-parameter NamedTuple --- (delta, Delta_g_K, sigma_ee) for the main planes,
# (tau_r_e, tau_d_e) for the τ_syn plane --- and sets BOTH the saved `parameters` (merged onto `defaults`) and
# the savename; `seed` is the Obs dimension. Shared across every plane so the save recipe lives in one place.
function save_member!(bs, i, c, seed)
    NE = size(bs.record.rate.data, 1)
    neuron_labels = Neuron(Symbol.("E" .* string.(1:NE)))
    lag_axis = 𝑡(τs)                                                       # MAD lag axis (physical time)
    nfreq = size(bs.record.psd.data, 3)
    freq_axis = 𝑓(range(0, inv(2 * dt_ms), length = nfreq) .* u"ms^-1")    # Welch one-sided frequencies
    fano_axis = 𝑡(fano_taus .* u"ms")                                      # Fano timescale axis
    mad = ToolsArray(permutedims(bs.record.mad.data[:, i, :]), (lag_axis, neuron_labels))
    # Log-sample each neuron's fine PSD onto demo_run's log-frequency grid (geometric mean per equal-width
    # log10 bin): the exponent fit sees the SAME resolution as scripts/demo_run.jl and the saved
    # array stays tiny (~50 pts vs ~5000 linear bins). ustripall first (logsample takes log10 of the
    # frequencies), select the 10-1000 Hz band by its ms^-1 value (a u"Hz" selector on the fresh range axis
    # throws), then re-attach ms^-1 units so downstream `spectral_exponents` can still select in u"Hz".
    psd_fine = ustripall(ToolsArray(permutedims(bs.record.psd.data[:, i, :]), (freq_axis, neuron_labels)))
    cols = map(col -> logsample(col[𝑓 = 0.01 .. 1.0]), eachslice(psd_fine; dims = Neuron))
    logf = 𝑓(collect(lookup(first(cols), 𝑓)) .* u"ms^-1")
    psd = ToolsArray(reduce(hcat, map(collect, cols)), (logf, neuron_labels))
    fano = ToolsArray(permutedims(bs.record.fano.data[:, i, :]), (fano_axis, neuron_labels))
    rate = ToolsArray(uconvert.(u"Hz", bs.record.rate.data[:, i] .* u"ms^-1"), neuron_labels)
    out = Dict(
        "parameters" => (; defaults..., c..., seed = seed),   # `seed` is the Obs dimension
        "rate" => rate, "fano" => fano, "inputs/mad" => mad, "inputs/psd" => psd,
    )
    return wsave(savepath(c, seed), out)
end

"""
    send_sweep(:name1 => vec1, :name2 => vec2, seed; batch = B)

Grid-sweep ONE 2-D plane at ONE seed. The two named axes vary over their vectors; every other parameter is
held at its working-regime default (`defaults0`). Partitioning is decided from the axis NAMES: `sigma_ee` sets
the connectome, so it can't vary within a batch --- cells are grouped by `sigma_ee` (one connectome per group)
--- while `delta`/`Delta_g_K`/`tau_r_e`/`tau_d_e` are per-member synapse overrides that co-execute in one
`(N,B)` batch.

Each result file is named for exactly its two axes + seed (`axis1 & axis2 & seed`), so distinct `(seed, plane)`
calls write DISJOINT file sets --- resumable, order-independent, and race-free to distribute one-per-GPU (e.g.
on Gadi). Returns the number of cells (re)computed.
"""
function send_sweep(ax1::Pair, ax2::Pair, seed; batch = B)
    n1, v1 = ax1
    n2, v2 = ax2
    axes = (n1, n2)
    # The cell NamedTuple = savename keys + saved `parameters` overrides: exactly the two swept axes. Every
    # non-swept parameter stays at its default (recorded inside `parameters` by `save_member!`, not the name).
    cell(a, b) = NamedTuple{axes}((a, b))
    cells = vec([cell(a, b) for a in v1, b in v2])

    model = WRCircuit.Spatial(; key = seed)
    remaining = filter(c -> !saved(c, seed), cells)
    isempty(remaining) && return 0

    # sigma_ee is the connectome axis → constant per batch: group by it. One group for every other plane.
    groups = if :sigma_ee in axes
        [filter(c -> c.sigma_ee == σ, remaining) for σ in unique(c.sigma_ee for c in remaining)]
    else
        [remaining]
    end
    # Flatten (group → B-chunks) into a task list so the progress logger sees the whole plane's batch count.
    tasks = NamedTuple[]
    for group in groups
        σ = :sigma_ee in axes ? first(group).sigma_ee : defaults0.sigma_ee
        for chunk in Iterators.partition(group, batch)
            push!(tasks, (; sigma_ee = σ, chunk = collect(chunk)))
        end
    end

    @info "seed $seed: $n1 × $n2 --- $(length(remaining)) cells in $(length(tasks)) batch(es)"
    map(Chart(LogLogger(length(tasks))), tasks) do t   # sequential (one GPU): batches run one at a time
        chunk = t.chunk
        deltas = [get(c, :delta, defaults0.delta) for c in chunk]
        dgks = [get(c, :Delta_g_K, defaults0.Delta_g_K) for c in chunk]
        member = (; sigma_ee = t.sigma_ee)
        :tau_r_e in axes && (member = merge(member, (; tau_r_e = [c.tau_r_e for c in chunk])))
        :tau_d_e in axes && (member = merge(member, (; tau_d_e = [c.tau_d_e for c in chunk])))
        run_and_save!(model, chunk, seed, deltas, dgks; member...)
        nothing
    end
    return length(remaining)
end

# The five planes, each an (axis1, axis2) pair = one `send_sweep` call. Looping (seed × PLANES) runs the whole
# sweep; distributing that product across GPUs is the parallel entry point (a job for later).
PLANES = [
    (:tau_r_e => tau_r_e, :tau_d_e => tau_d_e),        # τ_syn
    (:delta => delta, :tau_d_e => tau_d_e),           # δ/τ_d
    # (:delta => delta, :Delta_g_K => Delta_g_K),       # δ × Δg_K
    # (:delta => delta, :sigma_ee => sigma_ee),         # δ × σ_ee
    # (:Delta_g_K => Delta_g_K, :sigma_ee => sigma_ee), # Δg_K × σ_ee
]

begin # * Run: every plane at every seed (resume skips finished cells; shared central cells compute once)
    for seed in seeds, (ax1, ax2) in PLANES
        send_sweep(ax1, ax2, seed)
    end
    nfiles = count(endswith(".jld2"), readdir(path))
    @info "Sweep done: $nfiles result files in $path"
end
