#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRExperiment"

using WRExperiment
import TimeseriesTools: freqs
using Peaks
using FileIO
using CairoMakie
using IntervalSets
using Foresight
using DimensionalData
using TimeseriesTools
using StatsBase
using Statistics
using DataFrames
using HypothesisTests
using MultipleTesting
using Distributed
using Random
using Unitful
using LinearAlgebra
using Clustering
import AllenNeuropixelsBase as AN
using AllenNeuropixels.AllenNeuropixelsBase
import TimeseriesTools.duration
CairoMakie.set_theme!(Foresight.foresight(:physics))

begin
    stimulus = "spontaneous"
    structure = "VISp"
    sessionid = DEFAULT_SESSION_ID
end

begin
    S = AN.Session(sessionid)
    LFP = AN.formatlfp(S; rectify = false, epoch = :longest, structure, stimulus)
    spikes = AN.getspiketimes(S, structure)
end

begin # * Sort spikes by correlation
    _ts = 139 .. 140
    ts = _ts + minimum(times(LFP))
    i = 8
    y = values(spikes)
    y = map(y) do s
        idxs = findall(s .∈ [ts])
        s[idxs]
    end
    y = filter(x -> length(x) > 5, y)
    D = pairwise(stoic, y)
    h = hclust(Symmetric(1.0 ./ D))
    y = y[h.order]
    x = deepcopy(LFP[𝑡 = ts][:, i])
    t0 = minimum(times(x))
    times(x) .= times(x) .- t0

    # * Spike raster and LFP trace
    f = TwoPanel(; scale = 1)
    ax = Axis(f[1, 1], ylabel = "LFP", xtickformat = terseticks,
              yticks = ([], []), xgridvisible = false, limits = ((0, duration(x)), nothing),
              xticks = [])

    lines!(ax, x; linewidth = 2)

    ax = Axis(f[2, 1]; ylabel = "Neuron", xlabel = "Time (s)", xtickformat = terseticks,
              yticks = ([], []), xgridvisible = false, limits = ((0, duration(x)), nothing))
    map(enumerate(y)) do (neuronid, spktimes)
        scatter!(ax, spktimes .- t0, fill(neuronid, length(spktimes));
                 color = :black, markersize = 5)
    end
    rowgap!(f.layout, 1, 0.0)

    display(f)
    wsave(plotdir("traces", "traces.pdf"), f)
end
