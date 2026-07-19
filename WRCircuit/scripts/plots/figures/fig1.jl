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

# begin # * Plot inputs heatmap
#     file = files[1]
#     input = load(file, "inputs/distribution")
#     X = input[𝑡 = 20u"s" .. 20.1u"s"]
#     X = mean(X, dims = 𝑡)
#     X = dropdims(X, dims = 𝑡)
#     dx = load(file, "parameters")[:dx]
#     positions, idxs = WRCircuit.infer_geometry(X, dx)
#     N = sqrt(length(X)) |> Int
#     X = reshape(X, N, N)
#     heatmap(X;)
# end
begin # * Run a sample simulation at the working point (default values)
    tmax = 30u"s" # * Bump up
    tmin = 5u"s"
    model = WRCircuit.models.Spatial

    m = model()
    x = bpsolve(
        m, tmax;
        populations = [:E],
        vars = [:spike, :V, :input],
        transient = tmin
    )
end

begin # * Animate
    spikes = x[Population = At(:E), Var = At(:spike)]
    rates = WRCircuit.compute_rates(spikes, 50u"ms")
    dx = WRCircuit.defaults(model)[:parameters][:dx]
    WRCircuit.animate_rates(rates, dx; filename = plotdir("fig1", "animation.mp4"))
end
