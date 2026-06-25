#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin
    model = WRCircuit.models.Nonspatial
    begin
        N_e = 2000
        J_e = 0.0008 # Microsiemens
        nu = 7
        n_ext = 100
        k = 0.7
        K_ee = 72 * k
        K_ie = 28 * k
        K_ei = 88 * k
        K_ii = 39 * k
    end
end
begin
    tmax = 10u"s"
    tmin = 1u"s" # The transient. Simulations always begin at 0
    fixed_params = (; N_e, J_e, nu, n_ext, K_ee, K_ie, K_ei, K_ii)

    mua_dt = 2u"ms" # Gives mua spectrum max freq of 250 Hz

    monitors = ("E.spike",)#, "E.V", "E.input")
    stat_funcs = Dict("rate" => WRCircuit.stats.firing_rate,
                      "susceptibility" => WRCircuit.stats.susceptibility(bin = 10),
                      "spike_spectrum" => WRCircuit.stats.spike_spectrum(n_segments = 10),
                      #   "temporal_average" => WRCircuit.stats.temporal_average,
                      #   "grand_distribution" => WRCircuit.stats.grand_distribution(n_bins = 1000),
                      "mua" => WRCircuit.stats.mua(bin = ustrip(to_ms(mua_dt))))

    metadata = (; mua_dt, tmax, tmin, monitors)
end
begin# * Generate dict of parameter vectors
    sweep = (; delta = range(2.0, 7.0, length = 5))
    pnames = map(string, keys(sweep))
    pvals = stack(Iterators.product(values(sweep)...), dims = 1)
    sweep_params = Dict{String, Any}(zip(pnames, eachcol(pvals))) # Now a good shape for jax
    n_iters = length(first(values(sweep_params)))
    jax_keys = WRCircuit.jax.random.split(WRCircuit.jax.random.PRNGKey(42), n_iters)
    sweep_params["key"] = WRCircuit.numpy.array.(jax_keys) # * So that each run is independent
end
begin # * Create sweep function
    run = WRCircuit.stats.create_run(model, pydict(fixed_params), monitors,
                                     ustrip(to_ms(tmax)),
                                     ustrip(to_ms(tmin)))
    stats_run = WRCircuit.stats.create_stats_run(run, pydict(stat_funcs))
end
begin # * Run simulation
    stats, sweep_parameters = WRCircuit.stats.progress_vmap(stats_run, batch_size = 10)(pydict(sweep_params))
end
begin
    WRCircuit.stats.save("nonspatial_sweep.pickle",
                         (stats, sweep_params, fixed_params, metadata))
end
