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
using Random
using Distributed
@preamble
set_theme!(foresight(:physics))
Random.seed!(32)

stimuli = [r"Natural_Images", "spontaneous", "flash_250ms"]
xtickformat = terseticks
theta = Interval(THETA()...)
gamma = Interval(GAMMA()...)
alpha = 0.8
bandalpha = 0.2
mkpath(plotdir("fig2"))

if !isfile(calcdir("plots", savepath("fig2", Dict(), "jld2")))
    if nprocs() == 1
        if CLUSTER()
            using AcademicClusters
            ourprocs = AcademicClusters.USydPhysics.addprocs(
                30; mem = 6, ncpus = 1,
                project = projectdir(),
                queue = "l40s"
            )
        else
            addprocs(19)
        end
    end
    @everywhere using WRExperiment
    @everywhere WRExperiment.@preamble
end
