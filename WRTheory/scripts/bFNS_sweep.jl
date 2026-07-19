#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
using StochasticDiffEq
using Distributed
using AcademicClusters
WRTheory.@preamble()
set_theme!(foresight(:physics))

import FractionalNeuralSampling: Density

begin # * Add procs
    AcademicClusters.USydPhysics.addprocs(
        24; ncpus = 1, mem = "6GB", walltime = "23:00:00",
        queue = `taiji`
    )
    AcademicClusters.USydPhysics.addprocs(32; ncpus = 1, mem = "6GB", walltime = "23:00:00")
    addprocs(10) # Local

    @everywhere using WRTheory
    @everywhere using FractionalNeuralSampling
    @everywhere using MoreMaps
end

# * The aim is to sweep over alpha, beta, and eta
# * We extract the sampling accuracy, sampling efficiency,
# * spectral exponent, and diffusion exponent (and seed)
begin # * Create parameter grid
    βs = range(0.2, 1.0; length = 32) |> Dim{:β}
    αs = range(1.2, 2.0; length = 32) |> Dim{:α}
    obs = 1:10 |> Obs

    γ = 0.03
    η = 0.01

    γs = [γ] |> Dim{:γ}
    ηs = [η] |> Dim{:η}

    shared_params = (;
        tspan = 25000.0, # ms
        dt = 0.1,
        u0 = [-0.0, 0.0],
        domain = -10.0 .. 10.0,
        boundaries = PeriodicBox(-5 .. 5),
        approx_n_modes = 1000,
        τ = 1000.0,
        λ = 1.0e-4,
    )
end

if false # * Heatmap of H values
    _αs = range(1.0, 2.0; length = 100) |> Dim{:α}
    _βs = range(0.0, 1.0; length = 100) |> Dim{:β}
    h = map((α, β) -> (1 - β) / 2 + 1 / α, Chart(Iterators.product), _αs, _βs)
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "α", ylabel = "β", title = "H = (1 - β)/2 + 1/α")
    p = heatmap!(ax, h; colormap = darksunset)
    contour!(ax, h; levels = [0, 1], color = :white, linestyle = :dash)
    Colorbar(f[1, 2], p)
    display(f)
end

begin # * Map over parameters for flat potential
    params = (;
        𝜋 = test_density(:flat),
        shared_params...,
    )

    @info "Running flat sweep..."
    C = Chart(
        Iterators.product,
        Pmap(),
        ProgressLogger(1000)
    )
    res = map(simulate_bFNS_sweep(params), C, αs, βs, γs, ηs, obs)
    out = map(keys(first(res))) do k
        string(k) => map(Base.Fix2(getindex, k), res)
    end |> Dict{String, Any}
    out["params"] = params
    tagsave(datadir("bFNS_sweep", "flat_γ=$(γ)_η=$(η).jld2"), out; safe = true)
end

begin # * Map over parameters for unimodal potential
    params = (;
        𝜋 = test_density(:unimodal),
        shared_params...,
    )

    @info "Running unimodal sweep..."
    C = Chart(
        Iterators.product,
        Pmap(),
        ProgressLogger(1000)
    )
    res = map(simulate_bFNS_sweep(params), C, αs, βs, γs, ηs, obs)
    out = map(keys(first(res))) do k
        string(k) => map(Base.Fix2(getindex, k), res)
    end |> Dict{String, Any}
    out["params"] = params
    tagsave(datadir("bFNS_sweep", "unimodal_γ=$(γ)_η=$(η).jld2"), out; safe = true)
end

begin # * Map over parameters for bimodal potential
    params = (;
        𝜋 = test_density(:bimodal),
        shared_params...,
    )

    @info "Running bimodal sweep..."
    C = Chart(
        Iterators.product,
        Pmap(),
        ProgressLogger(1000)
    )
    res = map(simulate_bFNS_sweep(params), C, αs, βs, γs, ηs, obs)
    out = map(keys(first(res))) do k
        string(k) => map(Base.Fix2(getindex, k), res)
    end |> Dict{String, Any}
    out["params"] = params
    tagsave(datadir("bFNS_sweep", "bimodal_γ=$(γ)_η=$(η).jld2"), out; safe = true)
end
