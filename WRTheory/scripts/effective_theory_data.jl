#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
using ForwardDiff # with Optim, triggers TimeseriesTools' OptimExt; without both, MAPPLE `fit!` silently degrades
WRTheory.@preamble()
import FractionalNeuralSampling.Samplers: gen_lfsm_fns
import FFTW
FFTW.set_num_threads(1) # FFTW's own threads segfault (ip: nil) under `julia -t auto` on cartman; these 1-D FFTs lose nothing
include(joinpath(@__DIR__, "..", "..", "scripts", "variability_exponent.jl"))

# Produces datadir("Fig3_effective_theory.jld2"), plotted by the top-level
# scripts/Fig3_effective_theory.jl.

begin # * Options
    transient = 5000.0 # ms
    # Working-regime point: shared by the single-neuron simulation and the sweep slices
    α = 1.5
    β = 0.8
    η = 0.03
    γ = 0.01
end

begin # * Single-neuron mean field simulation
    tspan = 55000.0 # ms
    dt = 0.1 # ms, save resolution
    _dt = 0.05 # ms, solver step
    τ = 1000.0
    seed = 42
    𝜋 = Stable(1.5, 0, 0.14, 0.2) |> Density # (α, β, σ, μ)

    noise = gen_lfsm_fns(α, β; tspan, dt, seed, nhist = round(Int, τ / dt))
    params = (;
        α, β, γ, η, 𝜋,
        domain = -10 .. 10,
        boundaries = PeriodicBox(-5 .. 5),
        u0 = [0.0, 0.0],
        τ, approx_n_modes = 1000, λ = 1.0e-4,
        dt = _dt, saveat = dt, tspan, noise, seed,
    )
    S = bFNS(; params...) |> solve

    input_sol = S |> Timeseries
    input_sol = input_sol[𝑡 = transient .. tspan]

    prob = NeuronSampler(S; tspan, u0 = [-60.0, 0.0]) # filter the inputs through a model neuron
    sol = solve(prob) |> Timeseries
    sol = sol[𝑡 = transient .. tspan]
    spikes = times(DimensionalData.metadata(sol)[:callback_values])
    spikes = spikes[spikes .> transient]

    times(sol) ./= 1000 # to s
    times(input_sol) ./= 1000
    spikes ./= 1000
end

begin # * Load mean-field sweep
    sweepmeta = wload(datadir("mean_field_sweep", "metadata.jld2"))
    @unpack named, unnamed = sweepmeta
    parameter_grid = Iterators.product(values(sweepmeta["params"])...) |> collect
    parameter_grid = parameter_grid[η = At([η]), γ = At([0.0, γ])]

    files = map(Chart(Threaded(), ProgressLogger(1000)), parameter_grid) do p
        ps = Dict(name.(dims(parameter_grid)) .=> p)
        savename((; ps..., named...), "tsv")
    end
    @assert all(isfile, datadir.("mean_field_sweep", files))

    @info "Loading spike times..."
    sweep_spikes = map(Chart(Threaded(), ProgressLogger()), files) do file
        file = datadir("mean_field_sweep", file)
        filesize(file) == 0 ? [] : filter(>(transient), vec(readdlm(file)))
    end
    @info "Loading Fano factors..."
    fanos = map(Chart(Threaded(), ProgressLogger()), files) do file
        loadtimeseries(datadir("mean_field_sweep", "fano_$file"))
    end |> stack
    times(fanos) ./= 1000 # to s

    sweep_rates = map(s -> length(s) / (named.tspan - transient) * 1000, sweep_spikes) # Hz
end

begin # * MAPPLE Fano-curve fits
    # `variability_exponent` is shared with the circuit and experiment pipelines; see
    # scripts/variability_exponent.jl for the model and its rationale. The repeat-median is the
    # aggregated curve it needs --- per-repeat curves have no SNR for a free-knot fit.
    fann = fanos[η = Near(η), γ = Near(γ)]
    fann = mapslices(v -> all(isnan, v) ? NaN : nansafe(median)(v), fann, dims = Obs)
    fann = dropdims(fann, dims = Obs) # aggregate curves FIRST, one fit per cell
    fann = eachslice(fann, dims = setdiff(dims(fann), [dims(fann, 𝑡)]) |> Tuple)
    mcs = map(Chart(ProgressLogger(), Threaded()), fann) do ff
        try
            any(isnan, ff) ? NaN : variability_exponent(ff).β
        catch
            NaN
        end
    end
    mcs = permutedims(mcs, (:α, :β))
end

begin # * Reduce over repeats
    mfanos = Dropdims(mean)(fanos, dims = Obs) # Obs-mean Fano curves over the whole grid

    rates_αβ = sweep_rates[η = At(η), γ = At(γ)]
    rates_αβ = dropdims(median(rates_αβ, dims = Obs), dims = Obs)
    rates_αβ = permutedims(rates_αβ, (:α, :β))
    rates_αβ[rates_αβ .== 0.0] .= NaN # regimes that never spiked
end

begin # * Save
    tagsave(
        datadir("Fig3_effective_theory.jld2"),
        Dict(
            "sol" => sol, # neuron (V, w), transient removed, times in s
            "input_sol" => input_sol, # sampler input, transient removed, times in s
            "spikes" => spikes, # s
            "mfanos" => mfanos, # Obs-mean Fano curves, times in s
            "rates_αβ" => rates_αβ, # Hz, (α × β) at the working-regime (η, γ)
            "mcs" => mcs, # variability exponent, (α × β)
            "params" => (; α, β, η, γ, transient, tspan, dt, seed),
        )
    )
    @info "wrote" datadir("Fig3_effective_theory.jld2")
end
