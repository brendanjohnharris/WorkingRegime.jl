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
using Optim
using MoreMaps
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Define model
    function dual_expon(t, t₀; τr::Real, τd::Real, gmax::Real = 1.0)
        Δt = t - t₀
        Δt <= 0 && return 0.0
        A = (τd / (τd - τr)) * (τr / τd)^(τr / (τr - τd))
        return gmax * A * (exp(-Δt / τd) - exp(-Δt / τr))
    end
end

begin # * Set up figure
    f = OnePanel()
    ax = Axis(
        f[1, 1]; xlabel = "Time (ms)", ylabel = "Synaptic strength (a.u.)",
        yticks = [-1, 0, 1]
    )
    t = 0.0:0.1:50.0 # * ms
end

begin # * Excitatory AMPA
    tau_r_e = 0.7
    tau_d_e = 2.0
    x = dual_expon.(t, 0.0; τr = tau_r_e, τd = tau_d_e)
    lines!(
        ax, t, x,
        label = "AMPA", color = cucumber, alpha = 0.8
    )
end

begin # * Excitatory NMDA
    tau_r_e = 4.0
    tau_d_e = 40.0
    x = dual_expon.(t, 0.0; τr = tau_r_e, τd = tau_d_e)
    lines!(
        ax, t, x,
        label = "NMDA", color = cornflowerblue
    )
end

begin
    tau_r_i = 2.0
    tau_d_i = 4.5  # 3.0 for yifan, # 4.5 for shencong
    x = dual_expon.(t, 0.0; τr = tau_r_i, τd = tau_d_i)
    lines!(
        ax, t, .-x,
        label = "GABA", color = crimson, alpha = 0.8
    )
end

axislegend(ax; position = :rb)
display(f)
save(plotsdir("dual_expon_synapse.pdf"), f)
