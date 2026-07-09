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
    path = datadir("critical_sweep")

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

begin # * Parameter planes: three 2-D planes through the default working-regime point
    delta = round.(range(3.5, 5, length = 31); sigdigits = 3)
    Delta_g_K = round.(range(0, 0.005, length = 26); sigdigits = 3)
    sigma_ee = round.(range(0.03, 0.075, length = 19); sigdigits = 3)

    delta_0 = round(Float64(defaults[:delta]); sigdigits = 3)
    Delta_g_K_0 = round(Float64(defaults[:Delta_g_K]); sigdigits = 3)
    sigma_ee_0 = round(Float64(defaults[:sigma_ee]); sigdigits = 3)
    plane_dg = [(; delta = d, Delta_g_K = gk, sigma_ee = sigma_ee_0) for d in delta, gk in Delta_g_K]
    plane_ds = [(; delta = d, Delta_g_K = Delta_g_K_0, sigma_ee = s) for d in delta, s in sigma_ee]
    plane_gs = [(; delta = delta_0, Delta_g_K = gk, sigma_ee = s) for gk in Delta_g_K, s in sigma_ee]
    parameter_vector = unique(vcat(vec(plane_dg), vec(plane_ds), vec(plane_gs)))

    tau_r_e = round.(range(0.5, 2.0, length = 31); sigdigits = 3)
    tau_d_e = round.(range(2.0, 8.0, length = 25); sigdigits = 3)
    plane_td = vec([(; tau_r_e = tr, tau_d_e = td) for tr in tau_r_e, td in tau_d_e])

    isdir(path) || mkpath(path)
end

# Shared run helpers: the resume path (defined once, used by both filters and `save_member!`), the fixed
# reduction kwargs, and the run-a-batch-then-save loop, so both sweep blocks below stay thin.
savepath(p, seed) = joinpath(path, savename((; p..., seed = seed), "jld2"; connector))
saved(p, seed) = isfile(savepath(p, seed))
const RED = (; dt = dt_ms, arch, progress = true, scatter = :compacted,
             mad_lags, psd_fmin, fano_taus, rate = true, transient = transient_steps)
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
# the savename; `seed` is the Obs dimension. Shared by both run blocks so the save recipe lives in one place.
function save_member!(bs, i, c, seed)
    NE = size(bs.record.rate.data, 1)
    neuron_labels = Neuron(Symbol.("E" .* string.(1:NE)))
    lag_axis = 𝑡(τs)                                                       # MAD lag axis (physical time)
    nfreq = size(bs.record.psd.data, 3)
    freq_axis = 𝑓(range(0, inv(2 * dt_ms), length = nfreq) .* u"ms^-1")    # Welch one-sided frequencies
    fano_axis = 𝑡(fano_taus .* u"ms")                                      # Fano timescale axis
    mad = ToolsArray(permutedims(bs.record.mad.data[:, i, :]), (lag_axis, neuron_labels))
    # Log-sample each neuron's fine PSD onto critical_demo's log-frequency grid (geometric mean per equal-width
    # log10 bin): the exponent fit sees the SAME resolution as scripts/plots/critical_demo.jl and the saved
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

begin # * Run: outer over (seed × sigma_ee) --- one connectome each; inner (N,B) batch over (delta, Delta_g_K)
    for seed in seeds
        model = WRCircuit.Spatial(; key = seed)
        # this seed's not-yet-saved combos (resume), grouped by sigma_ee (the connectome axis → one build each)
        remaining = filter(p -> !saved(p, seed), parameter_vector)
        isempty(remaining) && continue
        @info "Computing seed $seed/$(length(seeds))"
        sigmas = unique(p.sigma_ee for p in remaining)
        map(Chart(LogLogger(length(sigmas))), sigmas) do σ
            combos = filter(p -> p.sigma_ee == σ, remaining)
            for chunk in Iterators.partition(combos, B)
                @debug "seed $seed / sigma_ee $σ: batch of $(length(chunk))"
                run_and_save!(model, chunk, seed, [c.delta for c in chunk], [c.Delta_g_K for c in chunk]; sigma_ee = σ)
            end
            nothing
        end
    end
    nfiles = count(endswith(".jld2"), readdir(path))
    @info "Sweep done: $nfiles result files in $path"
end

begin # * τ_syn plane: (tau_r_e, tau_d_e) at the default point --- batched over the shared connectome
    # One connectome per seed (sigma_ee = sigma_ee_0 fixed); the (N,B) batch co-executes B τ-cells as per-member
    # excitatory-synapse overrides (`simulate_batch(; tau_r_e, tau_d_e)`), same cost profile as the other planes.
    for seed in seeds
        model = WRCircuit.Spatial(; key = seed)
        remaining = filter(p -> !saved(p, seed), plane_td)
        isempty(remaining) && continue
        chunks = collect(Iterators.partition(remaining, B))
        @info "τ_syn plane: seed $seed/$(length(seeds)) --- $(length(remaining)) cells in $(length(chunks)) batch(es)"
        map(Chart(LogLogger(length(chunks))), chunks) do chunk
            n = length(chunk)
            run_and_save!(model, chunk, seed, fill(delta_0, n), fill(Delta_g_K_0, n);
                          sigma_ee = sigma_ee_0, tau_r_e = [c.tau_r_e for c in chunk], tau_d_e = [c.tau_d_e for c in chunk])
            nothing
        end
    end
    nfiles = count(endswith(".jld2"), readdir(path))
    @info "τ_syn plane done: $nfiles total result files in $path"
end
