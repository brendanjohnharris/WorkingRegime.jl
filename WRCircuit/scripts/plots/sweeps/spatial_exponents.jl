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
using Optim
using TimeseriesTools
using MoreMaps
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Sampling parameters
    batch_size = 4
    path = datadir("spatial_sampling")
end

begin # * Fixed parameters
    model = WRCircuit.models.Spatial

    tmax = 10u"s"
    transient = 2u"s" # The transient. Simulations always begin at 0

    monitors = ["E.input", "E.V"] |> pytuple
    stat_funcs = Dict("monitor" => WRCircuit.stats.monitor)

    dt = pyconvert(Float32, WRCircuit.brainpy.share["dt"]) * u"ms"
    metadata = (; tmax, transient, monitors, dt)

    savepath = datadir("spatial_exponents")
end

begin # * The idea is to randomly sample some parameters from discretized parameter distributions. This script can be left running to populate the model cache.
    # * Start by defining parameter ranges to sample from
    deltas = range(2, 6, length = 16)

    repeats = 1:1

    static_argnames = ("K_ee", "K_ii", "K_ei", "K_ie") # Jit-incompatible parameters (because they change array sizes)

    result_dims = (Monitor([Symbol("E.V"), Symbol("E.input")]),
                   Var([:psd, :mad]),
                   Dim{:delta}(deltas),
                   Obs(repeats))
    result_grid = ToolsArray(Array{Any}(undef, length.(result_dims)...), result_dims)
end

begin # * Fit functions
    f_range = 6u"Hz" .. 1000u"Hz"
    tau_range = 0u"s" .. 1u"s"
    function fit_spectrum(s; components, peaks, f_range)
        negdims = [i for i in 1:ndims(s) if i != dimnum(s, 𝑓)] |> Tuple
        s = s[𝑓 = f_range] |> ustripall
        s = mean(s, dims = negdims)
        s = dropdims(s, dims = negdims)
        _s = logsample(s)
        m = fit(MAPPLE, _s; components, peaks)
        fit!(m, _s)
        fitted_s = predict(m, s)
        return m # (; m, s, fitted_s, _s)
    end
    function fit_mad(s; components, peaks, tau_range)
        negdims = [i for i in 1:ndims(s) if i != dimnum(s, 𝑡)] |> Tuple
        s = s[𝑡 = tau_range] |> ustripall
        s = mean(s, dims = negdims)
        s = dropdims(s, dims = negdims)
        # _s = logsample(s)
        m = fit(MAPPLE, s; components, peaks)
        fit!(m, s)
        fitted_s = predict(m, s)
        return m # (; m, s, fitted_s)
    end
end

begin
    batch_idxs = Iterators.product(deltas, repeats)
    batch_idxs = Iterators.partition(batch_idxs, batch_size)

    for bidxs in batch_idxs
        delta = first.(bidxs)
        repeats = last.(bidxs)
        key = WRCircuit.jax.numpy.stack([WRCircuit.PRNGKey(i) for i in repeats]) # Important, must be python array

        these_ps = (; delta, key)

        run = WRCircuit.create_run(model; monitors, tmax, transient)
        stats_run = WRCircuit.create_stats_run(run, stat_funcs)
        stats, sweep_parameters = WRCircuit.partial_vmap(stats_run; static_argnames)(these_ps)
        out = WRCircuit.batchformat(pyconvert(Dict, stats), sweep_parameters; metadata,
                                    delete_key = false)
        out = out.monitor

        for (k, vars) in pairs(out)
            delta = k[:delta]
            repeat = k[:key][k[:key] .> 0] |> unique |> only |> Int

            # * Now generate critical exponents
            @info "Calculating spectra for δ = $delta, repeat = $repeat"
            spectra = map(vars::NamedTuple) do v
                v = v .- mean(v, dims = 𝑡)
                map(Chart(ProgressLogger(), Threaded()),
                    eachslice(v, dims = Neuron)) do x
                    spectrum(x, 0.5u"Hz")
                end |> stack
            end

            @info "Calculating mad for δ = $delta, repeat = $repeat"
            mads = map(vars::NamedTuple) do v
                map(Chart(ProgressLogger(), Threaded()),
                    eachslice(v, dims = Neuron)) do x
                    madev(x, round.(Int, logrange(10, 10000, length = 100) |> unique))
                end |> stack
            end

            spectra = map(spectra::NamedTuple) do s
                fit_spectrum(s; components = 1, peaks = 1, f_range)
            end
            mads = map(mads::NamedTuple) do m
                fit_mad(m; components = 2, peaks = 0, tau_range)
            end

            for outkey in keys(spectra)
                if outkey ∈ lookup(result_grid, Monitor)
                    result_grid[Monitor = At(outkey), Var = At(:psd), delta = At(delta), Obs = At(repeat)] = spectra[outkey]
                    result_grid[Monitor = At(outkey), Var = At(:mad), delta = At(delta), Obs = At(repeat)] = mads[outkey]
                end
            end

            # * Save current copy to file
            tagsave(joinpath(savepath, "spatial_exponents.jld2"),
                    Dict("result_grid" => result_grid, "metadata" => metadata))
        end
    end
end
