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

begin # * Choose example delta
    delta = 4.0

    files = readdir(datadir("critical_sweep"), join = true)
    deltas = map(f -> parse_savename(f; connector = string(connector))[2]["delta"], files)
    deltaidx = findmin(abs.(deltas .- delta))[2]
end

begin
    f = OnePanel()
    gs = subdivide(f, 1, 3)
    # end
    # begin # * Plot input distribution
    inputs = load(files[deltaidx], "inputs/distribution")
    ax = Axis(gs[1][1, 1:2])
    lines!(ax, inputs[𝑡 = 25u"s" .. 28u"s"][:, 1]; linewidth = 2, color = :crimson)

    ax = Axis(gs[1][2, 1])
    bins = range(-1, 1, length = 50)
    hist!(ax, inputs[:]; bins)

    ax = Axis(gs[1][2, 2]; yscale = log10, xscale = Makie.pseudolog10)
    bins = range(-2, 11, length = 50)
    hist!(ax, inputs[:]; bins)
    display(f)
end
