#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 --handle-signals=yes -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
using LinearAlgebra
using Random
using SparseArrays
using MoreMaps
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Sampling parameters
    batch_size = 1
    path = datadir("critical_sweep")
end

begin # * Fixed parameters
    model = WRCircuit.models.Spatial
    default_params = WRCircuit.defaults(model)

    tmax = 35u"s"
    transient = 5u"s" # The transient. Simulations always begin at 0
    dx = default_params[:parameters][:dx]
    rho = default_params[:parameters][:rho]

    monitors = ["E.spike", "E.input"] |> pytuple
    stat_funcs = Dict("monitor" => WRCircuit.stats.monitor)

    dt = pyconvert(Float32, WRCircuit.brainpy.share["dt"]) * u"ms"
    metadata = (; tmax, transient, monitors, dt)
end

begin # * Format parameters: three 2D planes through the default point
    # * 1D ranges for each parameter
    delta = range(3.5, 5, length = 31)
    Delta_g_K = range(0, 0.005, length = 26)
    # * Spatial range of the E2E connectivity kernel (mm); default sigma_ee = 0.06
    sigma_ee = range(0.03, 0.12, length = 19)

    delta = round.(delta; sigdigits = 3)
    Delta_g_K = round.(Delta_g_K; sigdigits = 3)
    sigma_ee = round.(sigma_ee; sigdigits = 3)

    delta_0 = round(Float64(default_params[:parameters][:delta]); sigdigits = 3)
    Delta_g_K_0 = round(Float64(default_params[:parameters][:Delta_g_K]); sigdigits = 3)
    sigma_ee_0 = round(Float64(default_params[:parameters][:sigma_ee]); sigdigits = 3)

    # * Each plane spans two parameters; the third is held at its default
    plane_dg = [
        (; delta = d, Delta_g_K = gk, sigma_ee = sigma_ee_0)
            for d in delta, gk in Delta_g_K
    ]            # delta × Delta_g_K
    plane_ds = [
        (; delta = d, Delta_g_K = Delta_g_K_0, sigma_ee = s)
            for d in delta, s in sigma_ee
    ]              # delta × sigma_ee
    plane_gs = [
        (; delta = delta_0, Delta_g_K = gk, sigma_ee = s)
            for gk in Delta_g_K, s in sigma_ee
    ]         # Delta_g_K × sigma_ee

    # * Concatenate and drop the shared cross-lines / centre point
    parameter_vector = unique(vcat(vec(plane_dg), vec(plane_ds), vec(plane_gs)))

    if !isdir(path)
        mkpath(path)
    end
end

begin
    parameter_vector = filter(parameter_vector) do p
        filename = savename(p; connector = string(connector))
        exists = isfile(joinpath(path, filename) * ".jld2")
        return !exists
    end
    sort!(parameter_vector; by = Base.Fix2(getindex, :Delta_g_K)) # Group by Delta_G_K
    batches = Iterators.partition(parameter_vector[:], batch_size)
end
if isempty(parameter_vector)
    @info "All parameter combinations already simulated. Exiting."
    exit()
end
begin # * Run simulation
    for (j, _params) in enumerate(batches)
        @info "Batch $j / $(length(batches))"

        @debug "Creating keys..."
        keys = rand(UInt32, length(_params))
        jax_keys = WRCircuit.jax.numpy.stack([WRCircuit.PRNGKey(k) for k in keys])
        # # ? Important, must be python array

        @debug "Creating params..."
        # * Vector of tuples to tuple of vectors
        keys = propertynames(first(_params))
        params = (; (k => getfield.(_params, k) for k in keys)..., key = jax_keys)

        @debug "Creating runner..."
        runner = WRCircuit.create_run(model; monitors, tmax, transient)

        @debug "Creating stats_run..."
        stats_run = WRCircuit.create_stats_run(runner, stat_funcs)

        @debug "Running partial_vmap..."
        stats, sweep_parameters = WRCircuit.partial_vmap(stats_run)(params)

        @debug "Formatting results..."
        res = WRCircuit.batchformat(pyconvert(Dict, stats), sweep_parameters; metadata)

        @debug "Processing individual params..."
        for i in eachindex(_params)
            @debug "  Processing $i / $(length(_params))"
            filename = savename(_params[i], "jld2"; connector)

            merged_params = (; default_params[:parameters]..., _params[i]...)
            ks = Base.keys(res.monitor)
            idx = Dict.(pairs.(ks)) .== [Dict(pairs(_params[i]))]
            r = res.monitor[only(collect(ks)[idx])]

            begin # * Input statistics
                x = r.var"E.input"
                x = x .- mean(x, dims = 𝑡) # Remove DC offset
                # * MAD of inputs
                τs = logrange(10, 10000, length = 100)
                τs = unique(round.(Int, τs)) .* dt
                mad = map(Chart(Threaded(), LogLogger()), eachslice(x, dims = Neuron)) do _x
                    madev(_x, τs)
                end |> stack

                # * PSD of inputs
                psd = map(Chart(Threaded(), LogLogger()), eachslice(x, dims = Neuron)) do _x
                    _x = _x .- mean(_x)
                    spectrum(_x, 0.25)
                end |> stack

                # * Downsample for distribution estimate
                # distribution = x[1:10:end, :]
            end

            out = Dict(
                "parameters" => merged_params,
                "spikes" => r.var"E.spike",
                "inputs/mad" => mad,
                "inputs/psd" => psd
                #    "inputs/distribution" => distribution
            )
            wsave(joinpath(path, filename), out)
        end

        @debug "Batch $j complete"
        stats = sweep_parameters = res = runner = stats_run = nothing
        GC.gc()
        PythonCall.GC.gc()
        WRCircuit.jax.clear_caches()
        flush(stderr)
        flush(stdout)
    end
end
