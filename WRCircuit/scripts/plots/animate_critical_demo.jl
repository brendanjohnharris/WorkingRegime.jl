#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate "WRCircuit"
using WRCircuit
using JLD2
WRCircuit.@preamble
set_theme!(fathom(:physics))

begin # * Load data
    x = load(datadir("critical_demo.jld2"), "x")
    fixed_params = load(datadir("critical_demo.jld2"), "fixed_params")
    dx = fixed_params.dx
    spikes = x[Population = At(:E), Var = At(:spike)]
end

begin # * Animate
    @info "Animating rates"
    rates = WRCircuit.compute_rates(spikes, 50u"ms")
    WRCircuit.animate_rates(rates, dx; filename = plotdir("critical_demo", "critical_demo.mp4"))
end
