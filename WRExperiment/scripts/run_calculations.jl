#! /bin/bash
#=
exec julia +1.12 -t auto "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRExperiment"
using WRExperiment
using MoreMaps
using AcademicClusters
using Distributed
using ProgressLogging
using DimensionalData
using Peaks
using FileIO
using JLD2
using IntervalSets
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
import AllenNeuropixelsBase as AN
import AllenNeuropixelsBase: Depth
import TimeseriesTools: freqs
calcdir = DrWatson.datadir

path = calcdir("madev")
mkpath(path)
stimuli = ["spontaneous", "flash_250ms", r"Natural_Images"]
session_table = load(calcdir("plots", "session_table.jld2"), "session_table")
oursessions = session_table.ecephys_session_id

pstructures = deepcopy(structures)
# Add thalamic regions
push!(pstructures, "LGd")
push!(pstructures, "LGd-sh")
push!(pstructures, "LGd-co")
_params = Iterators.product(oursessions, stimuli, unique(pstructures)) |> collect

Q = calcquality(path)
params = []
map(_params) do p
    sessionid, stimulus, structure = p
    try
        if !Q[
                stimulus = At(stimulus),
                Structure = At(structure),
                SessionID = At(sessionid),
            ]
            push!(params, p)
        end
    catch
        push!(params, p)
    end
end

if !isempty(params)
    if contains(gethostname(), "physics.usyd.edu.au")
        exprs = map(params) do (o, stimulus, structure)
            expr = quote
                using Pkg
                Pkg.instantiate()
                import AllenNeuropixelsBase as AN
                using WRExperiment
                WRExperiment.send_madev($o, $stimulus, $structure)
            end
        end
        batches = batches = 1:ceil(Int, 64):length(exprs)
        batches = [exprs[i:min(i + 63, length(exprs))] for i in batches]
        for batch in batches
            WRExperiment.submit_calculations(
                batch[1:min(length(batch), 32)], mem = 32,
                ncpus = 3, walltime = 8,
                queue = `taiji`
            )
            if length(batch) > 32
                WRExperiment.submit_calculations(
                    batch[32:end], mem = 32, ncpus = 3,
                    walltime = 8,
                    queue = `defaultQ`
                )
            end
        end
    else
        addprocs(7)
        @everywhere import AllenNeuropixelsBase as AN
        @everywhere using WRExperiment

        map(Chart(MoreMaps.Pmap(), LogLogger()), params) do param
            @info "Calculating madev for $(param)"
            WRExperiment.send_madev(param...)
            GC.gc()
        end
    end
end
