#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.13 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
using StochasticDiffEq
using Distributed
using AcademicClusters
WRTheory.@preamble()

import FractionalNeuralSampling: Density

begin # * Add procs
    AcademicClusters.USydPhysics.distributeprocs(
        66; ncpus = 1, mem = "6GB", walltime = "23:00:00",
        hpcs = ["cartman", "karl"]
    )

    @everywhere using WRTheory
    @everywhere using FractionalNeuralSampling
    @everywhere using MoreMaps
    @everywhere (import FFTW; FFTW.set_num_threads(1)) # FFTW's own threads segfault under -t auto
end

# * The aim is to sweep over alpha, beta, and eta
# * We extract the sampling accuracy, sampling efficiency,
# * spectral exponent, and diffusion exponent (and seed)
begin # * Create parameter grid
    βs = range(0.2, 1.0; length = 32) |> Dim{:β}
    αs = range(1.2, 2.0; length = 32) |> Dim{:α}
    obs = 1:10 |> Obs

    η = 0.01
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

begin # * One sweep per test potential. γ is set per sweep: bFNS_data.jl reads the bimodal sweep at
    # γ = 0.02, and the flat γ = 0 sweep feeds only checks/diffusion_relation_check.jl, so it runs last
    sweeps = ((:flat, 0.03), (:unimodal, 0.03), (:bimodal, 0.02), (:flat, 0.0))
    sweepfiles = map(sweeps) do (density, γ)
        params = (; 𝜋 = test_density(density), shared_params...)
        @info "Running $density sweep at γ = $γ..."
        C = Chart(Iterators.product, Pmap(), ProgressLogger(1000))
        res = map(simulate_bFNS_sweep(params), C, αs, βs, [γ] |> Dim{:γ}, ηs, obs)
        out = map(keys(first(res))) do k
            string(k) => map(Base.Fix2(getindex, k), res)
        end |> Dict{String, Any}
        out["params"] = params
        file = rootdatadir("bFNS_sweep", "$(density)_γ=$(γ)_η=$(η).jld2")
        tagsave(file, out)
        file
    end
end
