#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 --handle-signals=yes -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Critical sweep --- the working-regime phase diagram over (delta, Delta_g_K, sigma_ee), now run on the native
# Dewdrop (N,B) BATCHED ensemble (no JAX/BrainPy). Factoring: the connectome-changing axes (random `seed` and
# `sigma_ee`) become OUTER jobs --- one connectome each --- while the weight/param axes (`delta`, `Delta_g_K`)
# become the BATCH columns, co-executed over that shared connectome by `simulate_batch`. The 5 random seeds are
# the 'Obs' dimension stamped onto every saved result (plumbed into downstream scripts later). Set the env var
# `WRCIRCUIT_SMOKE` for a tiny end-to-end run.
using DrWatson
DrWatson.@quickactivate :WRCircuit
using WRCircuit
using JLD2
using MoreMaps
using Statistics
WRCircuit.@preamble

const SMOKE = haskey(ENV, "WRCIRCUIT_SMOKE")   # tiny config for a quick end-to-end validation

begin # * Sweep configuration
    # B = batch chunk size (members co-executed per `solve`). All saved statistics (MAD, PSD, rate, Fano) are
    # reduced ON-DEVICE by streaming monitors --- no trace or raster is stored --- so host memory no longer
    # scales with `nsteps`; B is now bounded only by the connectome + per-member reducer state (small).
    B = SMOKE ? 2 : 32
    seeds = SMOKE ? (1:1) : (1:5)              # connectome realizations → the 'Obs' (integer seed) dimension
    model_kwargs = SMOKE ? (; rho = 400, dx = 0.6, nu = 30.0, n_ext = 60) : (;)   # shrink the net under SMOKE
    arch = WRCircuit.DEWDROP_BACKEND()         # CPU by default; `Dewdrop.GPU()` fills the GPU with the batch
    path = SMOKE ? joinpath(tempdir(), "wrcircuit_sweep_smoke") : datadir("critical_sweep")
end

begin # * Fixed run parameters
    defaults = WRCircuit.defaults(WRCircuit.models.Spatial)
    tmax = SMOKE ? 200u"ms" : 35u"s"
    transient = SMOKE ? 50u"ms" : 5u"s"        # discarded transient; simulations always begin at 0
    dt = 0.1u"ms"
    dt_ms = ustrip(u"ms", dt)
    transient_ms = ustrip(u"ms", transient)
end

begin # * Parameter planes: three 2-D planes through the default working-regime point
    n = SMOKE ? (3, 3, 2) : (31, 26, 19)
    delta = round.(range(3.5, 5, length = n[1]); sigdigits = 3)
    Delta_g_K = round.(range(0, 0.005, length = n[2]); sigdigits = 3)
    sigma_ee = round.(range(0.03, 0.12, length = n[3]); sigdigits = 3)
    delta_0 = round(Float64(defaults[:delta]); sigdigits = 3)
    Delta_g_K_0 = round(Float64(defaults[:Delta_g_K]); sigdigits = 3)
    sigma_ee_0 = round(Float64(defaults[:sigma_ee]); sigdigits = 3)
    plane_dg = [(; delta = d, Delta_g_K = gk, sigma_ee = sigma_ee_0) for d in delta, gk in Delta_g_K]
    plane_ds = [(; delta = d, Delta_g_K = Delta_g_K_0, sigma_ee = s) for d in delta, s in sigma_ee]
    plane_gs = [(; delta = delta_0, Delta_g_K = gk, sigma_ee = s) for gk in Delta_g_K, s in sigma_ee]
    parameter_vector = unique(vcat(vec(plane_dg), vec(plane_ds), vec(plane_gs)))
    isdir(path) || mkpath(path)
end

# τ lags for the input MAD. The streaming MADev monitor works in integer RECORDED-SAMPLE (step) units, so we
# pass step `mad_lags`; `τs = mad_lags .* dt` is the matching physical-time axis used to label the saved result.
const mad_lags = unique(round.(Int, logrange(10, 10000, length = 100)))
const τs = mad_lags .* dt
const psd_fmin = 0.25                       # min resolved frequency for the streaming Welch PSD (≡ old spectrum(_, 0.25))
# Fano-factor window sizes (ms), matching plot_critical_demo.jl's `logrange(dt*10, dt*1000, length=200)`.
const fano_taus = collect(logrange(dt_ms * 10, dt_ms * 1000, length = 200))
const transient_steps = round(Int, transient_ms / dt_ms)   # transient dropped on-device by the reductions

begin # * Run: outer over (seed × sigma_ee) --- one connectome each; inner (N,B) batch over (delta, Delta_g_K)
    for seed in seeds
        model = WRCircuit.Spatial(; key = seed)
        # this seed's not-yet-saved combos (resume), grouped by sigma_ee (the connectome axis → one build each)
        remaining = filter(parameter_vector) do p
            !isfile(joinpath(path, savename((; p..., seed = seed), "jld2"; connector)))
        end
        isempty(remaining) && continue
        for σ in unique(p.sigma_ee for p in remaining)
            combos = filter(p -> p.sigma_ee == σ, remaining)
            for chunk in Iterators.partition(combos, B)
                deltas = [Float64(c.delta) for c in chunk]
                dgks = [Float64(c.Delta_g_K) for c in chunk]
                @info "seed $seed / sigma_ee $σ: batch of $(length(chunk))"
                # All four statistics --- input MAD + PSD (from `itot`), and spike RATE + FANO --- are reduced
                # ON-DEVICE by streaming monitors. No trace or raster is ever materialised (host or device), so
                # host memory no longer scales with `nsteps`. `scatter = :compacted` is the advisor's pick for
                # this sparse-firing large network (processes only active synapses; ~30× faster scatter).
                bs = simulate_batch(
                    model, tmax, deltas, dgks;
                    sigma_ee = σ, dt = dt_ms, arch = arch, progress = true, scatter = :compacted,
                    mad_lags = mad_lags, psd_fmin = psd_fmin, fano_taus = fano_taus, rate = true,
                    transient = transient_steps, model_kwargs...
                )
                NE = size(bs.record.rate.data, 1)
                neuron_labels = Neuron(Symbol.("E" .* string.(1:NE)))
                lag_axis = 𝑡(τs)                                                       # MAD lag axis (physical time)
                nfreq = size(bs.record.psd.data, 3)
                freq_axis = 𝑓(range(0, inv(2 * dt_ms), length = nfreq) .* u"ms^-1")    # Welch one-sided frequencies
                fano_axis = 𝑡(fano_taus .* u"ms")                                      # Fano timescale axis
                for (i, c) in enumerate(chunk)
                    # Rebuild the labelled (stat × Neuron) arrays from member i's on-device-reduced slice.
                    mad = ToolsArray(permutedims(bs.record.mad.data[:, i, :]), (lag_axis, neuron_labels))
                    psd = ToolsArray(permutedims(bs.record.psd.data[:, i, :]), (freq_axis, neuron_labels))
                    fano = ToolsArray(permutedims(bs.record.fano.data[:, i, :]), (fano_axis, neuron_labels))
                    rate = ToolsArray(uconvert.(u"Hz", bs.record.rate.data[:, i] .* u"ms^-1"), neuron_labels)
                    parameters = (; defaults..., c..., seed = seed)   # `seed` is the Obs dimension
                    out = Dict(
                        "parameters" => parameters, "rate" => rate, "fano" => fano,
                        "inputs/mad" => mad, "inputs/psd" => psd,
                    )
                    wsave(joinpath(path, savename((; c..., seed = seed), "jld2"; connector)), out)
                end
                bs = nothing
                GC.gc()
            end
        end
    end
    nfiles = count(endswith(".jld2"), readdir(path))
    @info "Sweep done: $nfiles result files in $path"
end
