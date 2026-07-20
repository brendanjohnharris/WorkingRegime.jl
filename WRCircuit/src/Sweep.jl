# Circuit parameter sweep --- the machinery behind `scripts/circuit_sweep.jl`. `send_sweep(:a => v1, :b => v2,
# seed)` grid-sweeps ONE 2-D plane at ONE connectome seed and writes one result file per cell, named for its two
# axes + seed. It is the unit distributed across GPUs (one job per (plane, seed)); the script holds only the
# axis vectors and either loops `send_sweep` locally or fans it out with `AcademicClusters` `runscripts` on Gadi.

# Fixed measurement protocol: batch size, sim durations, and the on-device reduction bands. Bundled into one
# module const so `send_sweep`/`save_member!` share it without polluting the WRCircuit namespace. The reduction
# fields are interdependent (τs ← mad_lags·dt, psd_fmin ← dt), so assemble them in a `let`.
const SWEEP = let
    dt = 0.1u"ms"
    dt_ms = ustrip(u"ms", dt)
    transient = 5u"s"                          # discarded transient; simulations always begin at 0
    mad_lags = unique(round.(Int, logrange(10, 10000, length = 100)))
    d = defaults(models.Spatial)               # working-regime point (NamedTuple of model defaults)
    (;
        B = 32,
        tmax = 35u"s",
        dt_ms,
        mad_lags,
        τs = mad_lags .* dt,                    # MAD lag axis (physical time)
        psd_fmin = (1 / dt_ms) / 8192,         # nfft = 8192 (power of 2 → small cuFFT workspace); ~1.2 Hz.
        fano_taus = collect(logrange(dt_ms * 10, dt_ms * 1000, length = 200)),
        transient_steps = round(Int, ustrip(u"ms", transient) / dt_ms),
        model_defaults = d,
        # Pivot each plane holds its two non-swept siblings at (rounded like the axes so they match a savename).
        defaults0 = (;
            delta = round(Float64(d[:delta]); sigdigits = 3),
            Delta_g_K = round(Float64(d[:Delta_g_K]); sigdigits = 3),
            sigma_ee = round(Float64(d[:sigma_ee]); sigdigits = 3),
        ),
        required_keys = ("parameters", "rate", "fano", "inputs/mad", "inputs/psd"),
    )
end

# A result file is complete iff it carries all of `SWEEP.required_keys`. `haskey` is a metadata lookup (nested
# paths resolve), so this validates WITHOUT deserialising the arrays --- ~6 ms/file. `savepath`/`saved` take the
# sweep directory explicitly so `send_sweep` can be pointed at any `datadir`.
savepath(path, c, seed) = joinpath(path, savename((; c..., seed = seed), "jld2"; connector))
_complete(f) = try
    jldopen(g -> all(k -> haskey(g, k), SWEEP.required_keys), f, "r")
catch
    false          # unreadable / truncated header
end
# Resume guard: skip a file only if it exists AND is complete. A present-but-truncated file (an interrupted
# `wsave`) is DELETED here so the sweep regenerates it --- bare `isfile` would skip the stub forever.
function saved(path, c, seed)
    f = savepath(path, c, seed)
    isfile(f) || return false
    _complete(f) && return true
    @warn "Incomplete sweep file (missing keys) --- deleting to regenerate" file = f
    rm(f; force = true)
    return false
end

function run_and_save!(model, chunk, seed, deltas, dgks, path; member...)
    # `arch` is constructed here, not at module load, so precompilation never touches the GPU backend.
    red = (;
        dt = SWEEP.dt_ms, arch = DEWDROP_BACKEND(), progress = true, scatter = :compacted,
        SWEEP.mad_lags, SWEEP.psd_fmin, SWEEP.fano_taus, rate = true, transient = SWEEP.transient_steps,
    )
    bs = simulate_batch(model, SWEEP.tmax, deltas, dgks; member..., red...)
    for (i, c) in enumerate(chunk)
        save_member!(bs, i, c, seed, path)
    end
    bs = nothing
    GC.gc()
    return nothing
end

# Rebuild member `i`'s labelled (stat × Neuron) arrays from its on-device-reduced slice and save one result
# file. `c` is the member's varied-parameter NamedTuple (exactly the two swept axes) --- it sets BOTH the saved
# `parameters` (merged onto the model defaults) and the savename.
function save_member!(bs, i, c, seed, path)
    NE = size(bs.record.rate.data, 1)
    neuron_labels = Neuron(Symbol.("E" .* string.(1:NE)))
    lag_axis = 𝑡(SWEEP.τs)
    nfreq = size(bs.record.psd.data, 3)
    freq_axis = 𝑓(range(0, inv(2 * SWEEP.dt_ms), length = nfreq) .* u"ms^-1")    # Welch one-sided frequencies
    fano_axis = 𝑡(SWEEP.fano_taus .* u"ms")
    mad = ToolsArray(permutedims(bs.record.mad.data[:, i, :]), (lag_axis, neuron_labels))
    # Log-sample each neuron's fine PSD onto the log-frequency grid (geometric mean per equal-width log10 bin):
    # the exponent fit sees the same resolution as demo_run.jl and the saved array stays tiny (~50 pts). Select
    # the 10-1000 Hz band by its ms^-1 value (a u"Hz" selector on the fresh range axis throws), then re-attach.
    psd_fine = ustripall(ToolsArray(permutedims(bs.record.psd.data[:, i, :]), (freq_axis, neuron_labels)))
    cols = map(col -> logsample(col[𝑓 = 0.01 .. 1.0]), eachslice(psd_fine; dims = Neuron))
    logf = 𝑓(collect(lookup(first(cols), 𝑓)) .* u"ms^-1")
    psd = ToolsArray(reduce(hcat, map(collect, cols)), (logf, neuron_labels))
    fano = ToolsArray(permutedims(bs.record.fano.data[:, i, :]), (fano_axis, neuron_labels))
    rate = ToolsArray(uconvert.(u"Hz", bs.record.rate.data[:, i] .* u"ms^-1"), neuron_labels)
    out = Dict(
        "parameters" => (; SWEEP.model_defaults..., c..., seed = seed),   # `seed` is the Obs dimension
        "rate" => rate, "fano" => fano, "inputs/mad" => mad, "inputs/psd" => psd,
    )
    return wsave(savepath(path, c, seed), out)
end

"""
    send_sweep(:name1 => vec1, :name2 => vec2, seed; batch = SWEEP.B, path = datadir("circuit_sweep"))

Grid-sweep ONE 2-D plane at ONE connectome seed. The two named axes vary over their vectors; every other
parameter is held at its working-regime default. Partitioning is decided from the axis NAMES: `sigma_ee` sets
the connectome, so it can't vary within a batch --- cells are grouped by `sigma_ee` (one connectome per group)
--- while `delta`/`Delta_g_K`/`tau_r_e`/`tau_d_e` are per-member synapse overrides that co-execute in one `(N,B)`
batch. Sequential over batches (one GPU).

Each result file is named for exactly its two axes + seed (`axis1 & axis2 & seed`), so distinct `(seed, plane)`
calls write DISJOINT file sets --- resumable, order-independent, and race-free to fan out one-per-GPU (e.g. via
`AcademicClusters.NCIGadi.runscripts` on Gadi). Returns the number of cells (re)computed.
"""
function send_sweep(ax1::Pair, ax2::Pair, seed; batch = SWEEP.B, path = datadir("circuit_sweep"))
    n1, v1 = ax1
    n2, v2 = ax2
    axes = (n1, n2)
    # The cell NamedTuple = savename keys + saved `parameters` overrides: exactly the two swept axes. Every
    # non-swept parameter stays at its default (recorded inside `parameters` by `save_member!`, not the name).
    cell(a, b) = NamedTuple{axes}((a, b))
    cells = vec([cell(a, b) for a in v1, b in v2])

    isdir(path) || mkpath(path)
    model = Spatial(; key = seed)
    remaining = filter(c -> !saved(path, c, seed), cells)
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
        σ = :sigma_ee in axes ? first(group).sigma_ee : SWEEP.defaults0.sigma_ee
        for chunk in Iterators.partition(group, batch)
            push!(tasks, (; sigma_ee = σ, chunk = collect(chunk)))
        end
    end

    @info "seed $seed: $n1 × $n2 --- $(length(remaining)) cells in $(length(tasks)) batch(es)"
    map(Chart(LogLogger(length(tasks))), tasks) do t   # sequential (one GPU): batches run one at a time
        chunk = t.chunk
        deltas = [get(c, :delta, SWEEP.defaults0.delta) for c in chunk]
        dgks = [get(c, :Delta_g_K, SWEEP.defaults0.Delta_g_K) for c in chunk]
        member = (; sigma_ee = t.sigma_ee)
        :tau_r_e in axes && (member = merge(member, (; tau_r_e = [c.tau_r_e for c in chunk])))
        :tau_d_e in axes && (member = merge(member, (; tau_d_e = [c.tau_d_e for c in chunk])))
        run_and_save!(model, chunk, seed, deltas, dgks, path; member...)
        nothing
    end
    return length(remaining)
end
export send_sweep
