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
using Statistics
WRCircuit.@preamble

begin # * Sweep configuration
    B = 32
    seeds = 1:3                                # connectome realizations → the 'Obs' (integer seed) dimension
    arch = WRCircuit.DEWDROP_BACKEND()         # CPU by default; `Dewdrop.GPU()` fills the GPU with the batch
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
    isdir(path) || mkpath(path)
end

begin # * Run: outer over (seed × sigma_ee) --- one connectome each; inner (N,B) batch over (delta, Delta_g_K)
    for seed in seeds
        model = WRCircuit.Spatial(; key = seed)
        # this seed's not-yet-saved combos (resume), grouped by sigma_ee (the connectome axis → one build each)
        remaining = filter(parameter_vector) do p
            !isfile(joinpath(path, savename((; p..., seed = seed), "jld2"; connector)))
        end
        isempty(remaining) && continue
        @info "Computing seed $seed/$(length(seeds))"
        sigmas = unique(p.sigma_ee for p in remaining)
        C = Chart(LogLogger(length(sigmas)))
        map(C, sigmas) do σ
            combos = filter(p -> p.sigma_ee == σ, remaining)
            for chunk in Iterators.partition(combos, B)
                deltas = [Float64(c.delta) for c in chunk]
                dgks = [Float64(c.Delta_g_K) for c in chunk]
                @debug "seed $seed / sigma_ee $σ: batch of $(length(chunk))"
                bs = simulate_batch(
                    model, tmax, deltas, dgks;
                    sigma_ee = σ, dt = dt_ms, arch = arch, progress = true, scatter = :compacted,
                    mad_lags = mad_lags, psd_fmin = psd_fmin, fano_taus = fano_taus, rate = true,
                    transient = transient_steps,
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
                    # Log-sample each neuron's fine PSD onto critical_demo's log-frequency grid (equal-width
                    # log10 bins, geometric mean per bin): the exponent fit then sees the SAME resolution as
                    # scripts/plots/critical_demo.jl, and the saved array stays tiny (~50 pts vs ~5000 linear bins).
                    # ustripall first (logsample takes log10 of the frequencies), select the fit band by its
                    # ms^-1 value (0.01-1.0 = 10-1000 Hz; a u"Hz" selector on the fresh range axis throws), then
                    # re-attach ms^-1 units so downstream `spectral_exponents` can still select in u"Hz".
                    psd_fine = ustripall(ToolsArray(permutedims(bs.record.psd.data[:, i, :]), (freq_axis, neuron_labels)))
                    cols = map(c -> logsample(c[𝑓 = 0.01 .. 1.0]), eachslice(psd_fine; dims = Neuron))
                    logf = 𝑓(collect(lookup(first(cols), 𝑓)) .* u"ms^-1")
                    psd = ToolsArray(reduce(hcat, map(collect, cols)), (logf, neuron_labels))
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
            nothing
        end
    end
    nfiles = count(endswith(".jld2"), readdir(path))
    @info "Sweep done: $nfiles result files in $path"
end
