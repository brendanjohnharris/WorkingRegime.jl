#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
WRTheory.@preamble()
using AcademicClusters
using Distributed
set_theme!(foresight(:physics))

begin # * Add workers
    AcademicClusters.USydPhysics.addprocs(
        30; ncpus = 1, mem = "6GB", walltime = "12:00:00",
        queue = `taiji`
    )
    AcademicClusters.USydPhysics.addprocs(12; ncpus = 1, mem = "6GB", walltime = "12:00:00")
    AcademicClusters.USydPhysics.addprocs(12; ncpus = 1, mem = "6GB", walltime = "12:00:00")
    AcademicClusters.USydPhysics.addprocs(12; ncpus = 1, mem = "6GB", walltime = "12:00:00")
    addprocs(10) # Local

    @everywhere using WRTheory
    @everywhere using MoreMaps
end

begin # * Parameter ranges
    αs = range(1.2, 2.0, length = 32) |> Dim{:α}
    βs = range(0.2, 1.0, length = 32) |> Dim{:β}

    γs = 0.0:0.01:0.06 |> Dim{:γ}
    ηs = [0.02, 0.03, 0.04] |> Dim{:η}

    # γs = [0.0, 0.06] |> Dim{:γ} # Fixed
    # ηs = [0.03] |> Dim{:η} # Fixed

    obs = Obs(1:10) # Repeats
end

begin # * Named and unnamed parameters
    named = (;
        tspan = 25000.0,
    ) # ms
    unnamed = (;
        #    𝜋 = Normal(0.3, 0.5) |> Density, # Parameters guessed from dewdrop
        # 𝜋 = Stable(1.50, 0.1, 0.5, 0.25) |> Density
        #    𝜋 = Stable(1.5, -0.1, 0.13, 0.22) |> Density, # (α, β, σ, μ)
        𝜋 = Stable(1.5, 0, 0.14, 0.2) |> Density, # (α, β, σ, μ)
        u0_input = [0.0, 0.0],
        u0_neuron = [-60.0, 0.0],
        boundaries = PeriodicBox(-5 .. 5),
        domain = -10 .. 10,
        approx_n_modes = 1000,
        λ = 1.0e-4,
        dt = 0.1,
        τ = 1000.0,
    )
end

begin # * Run loop, in batches over α, β, obs (we can re-use noise over γ, η)
    @info "Starting mean field sweep..."
    mkpath(datadir("mean_field_sweep"))
    @everywhere begin
        named = $named
        unnamed = $unnamed
        γs = $γs
        ηs = $ηs

        function runsimsave(α, β, obs)
            WRTheory.simsave(α, β, obs; named, unnamed, γs, ηs)
        end
    end
    map(
        runsimsave, Chart(Iterators.product, ProgressLogger(5000), Pmap()), αs,
        βs, obs
    )
    tagsave(
        datadir("mean_field_sweep", "metadata.jld2"),
        Dict(
            "params" => Dict(
                "αs" => αs,
                "βs" => βs,
                "γs" => γs,
                "ηs" => ηs,
                "obs" => obs
            ),
            "named" => named,
            "unnamed" => unnamed
        );
        safe = true
    )
end
