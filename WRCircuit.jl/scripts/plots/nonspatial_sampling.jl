#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
using LinearAlgebra
using Random
using SparseArrays
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Sampling parameters
    batch_size = 16
    path = datadir("nonspatial_sampling")
end

begin # * Fixed parameters
    model = WRCircuit.models.Nonspatial
    default_params = WRCircuit.defaults(model)

    tmax = 30u"s"
    transient = 5u"s" # The transient. Simulations always begin at 0
    key = 42

    monitors = ["E.spike"] |> pytuple
    stat_funcs = Dict("monitor" => WRCircuit.stats.monitor)

    dt = pyconvert(Float32, WRCircuit.brainpy.share["dt"]) * u"ms"
    metadata = (; tmax, transient, monitors, dt)
end

begin # * The idea is to randomly sample some parameters from discretized parameter distributions. This script can be left running to populate the model cache.
    # * Start by defining parameter ranges to sample from
    parameter_grid = [
        Dim{:K_ee}(round.(Int, range(10, 200, length = 8))),
        Dim{:K_ei}(round.(Int, range(10, 200, length = 8))),
        Dim{:K_ie}(round.(Int, range(10, 200, length = 8))),
        Dim{:K_ii}(round.(Int, range(10, 200, length = 8))),
        Dim{:nu}(range(4.0, 12.0, length = 8)),
        Dim{:J_ei}(range(0.0003, 0.0011, length = 8)),
        Dim{:Delta_g_K}([0.001, 0.002, 0.003, 0.004, 0.005])
        # Dim{:sigma_ee}(range(0.04, 0.1, length = 8)),
        # Dim{:sigma_ei}(range(0.04, 0.12, length = 8)),
        # Dim{:sigma_ie}(range(0.1, 0.2, length = 8)),
        # Dim{:sigma_ii}(range(0.2, 0.2, length = 8)),
        # Dim{:J_ee}(range(0.0005, 0.0015, length = 16)),
        # Dim{:J_ei}(range(0.0005, 0.002, length = 16)),
        # Dim{:tau_r_e}(range(0.5, 3.0, length = 8)),
        # Dim{:tau_r_i}(range(0.5, 3.0, length = 8)),
        # Dim{:tau_d_e}(range(2.0, 8.0, length = 8)),
        # Dim{:tau_d_i}(range(2.0, 8.0, length = 8))
    ]

    static_argnames = ("K_ee", "K_ii", "K_ei", "K_ie") # Jit-incompatible parameters (because they change array sizes)

    parameter_grid = Iterators.product(parameter_grid...) |> collect
    parameter_grid = map(parameter_grid) do ps
        NamedTuple(name.(dims(parameter_grid)) .=> ps)
    end

    hash_grid = map(parameter_grid) do p
        merged_params = (; default_params[:parameters]..., p..., key = [0, key])
        Base.hash(merged_params |> WRCircuit.sortparams) # Order matters for named tuples
    end

    tagsave(joinpath(path, "parameter_grid.jld2"),
            Dict("parameter_grid" => parameter_grid, "default_params" => default_params,
                 "metadata" => metadata, "hash_grid" => hash_grid))
end

while true
    begin # * Load computed grid
        computed_hashes = splitext.(readdir(path)) .|> first
        computed_hashes = filter(x -> all(isdigit, x), computed_hashes)
        computed_hashes = map(Base.Fix1(parse, UInt), computed_hashes)
        Q = map(hash_grid) do h
            h in computed_hashes
        end
    end
    if all(Q)
        @info "All parameter combinations computed, exiting"
        break
    end
    begin # * Then, randomly select batches of un-computed parameter combinations. Start by picking ONE value for the static args, then batch_size from the non-static args
        these_ps = parameter_grid[.!Q]
        static_options = map(static_argnames) do n
            getindex.(these_ps, Symbol(n))
        end
        static_options = zip(static_options...) |> collect
        static_choice = static_options[rand(1:length(static_options))]
        idxs = shuffle(findall(static_options .== [static_choice]))
        idxs = idxs[1:min(batch_size, length(idxs))]
        these_ps = these_ps[idxs]
    end
    begin # And format parameters
        jax_keys = WRCircuit.jax.numpy.stack([WRCircuit.PRNGKey(key)
                                              for _ in 1:batch_size]) # Important, must be python array

        pnames = keys(these_ps |> first)
        these_ps = map(pnames) do n
            n => getindex.(these_ps, n)
        end |> Dict{Symbol, Any}
        push!(these_ps, :key => jax_keys)
    end

    begin # * And run the batch simulation
        run = WRCircuit.create_run(model; monitors, tmax, transient)
        stats_run = WRCircuit.create_stats_run(run, stat_funcs)
        stats, sweep_parameters = WRCircuit.partial_vmap(stats_run; static_argnames)(these_ps)
    end

    begin # * Format the monitors
        res = WRCircuit.batchformat(pyconvert(Dict, stats), sweep_parameters; metadata)
    end

    begin # * Save each result, combining swept params with default model values
        map((collect ∘ keys ∘ last ∘ first)(res)) do params
            merged_params = (; default_params[:parameters]..., params..., key = [0, key])
            hsh = Base.hash(WRCircuit.sortparams(merged_params))
            @assert hsh ∈ hash_grid
            filename = hsh
            r = map(collect(res)) do (k, v)
                k => v[params]
            end |> Dict
            out = Dict("parameters" => merged_params, r...)
            tagsave(joinpath(path, "$filename.jld2"), out)
            return merged_params
        end
    end
end
