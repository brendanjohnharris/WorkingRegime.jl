#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate :WRCircuit
using Bootstrap
using JLD2
using LinearAlgebra
using Optim
using MoreMaps
WRCircuit.@preamble
set_theme!(foresight(:physics))
outfile = datadir("critical_demo.jld2")

begin
    model = WRCircuit.models.Spatial
    begin # FNS parameters
        rho = 20000
        dx = 0.5
        sigma_ee = 0.06  # from decay=7.5
        sigma_ei = 0.07  # from decay=9.5
        sigma_ie = 0.14  # from decay=19
        sigma_ii = 0.14  # from decay=19
        K_ee = 260
        K_ei = 340
        K_ie = 225
        K_ii = 290
        nu = 10.0
        n_ext = 100
        Delta_g_K = 0.002
    end
end

begin
    tmax = 55u"s"
    tmin = 5u"s" # The transient. Simulations always begin at 0
    fixed_params = (;
        rho,
        dx,
        sigma_ee,
        sigma_ei,
        sigma_ie,
        sigma_ii,
        K_ee,
        K_ei,
        K_ie,
        K_ii,
        nu,
        n_ext,
        Delta_g_K,
        key = WRCircuit.PRNGKey(52),
    )
end

begin # * Run simulation
    m = model(; fixed_params...)
    sol = simulate(m, tmax; populations = [:E], vars = [:spike, :V, :input])
    x = bpformat(sol; populations = [:E], vars = [:spike, :V, :input], transient = tmin)

    @info "Saving data"
    epositions = [collect(pos) for pos in sol[:E].positions]
    ipositions = [collect(pos) for pos in sol[:I].positions]
    tagsave(
        outfile,
        Dict(
            "x" => x,
            "fixed_params" => fixed_params,
            "epositions" => epositions,
            "ipositions" => ipositions
        ), safe = true
    )
end
