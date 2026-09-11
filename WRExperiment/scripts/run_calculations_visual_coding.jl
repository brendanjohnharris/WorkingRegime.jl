#! /bin/bash
#=
exec julia +1.12 -t auto "${BASH_SOURCE[0]}" "$@"
=#
# Per-(session, structure) calculations for the Allen Visual Coding cohorts, the second data source
# behind Figure S1. This is `run_calculations.jl` with a different session list and output
# directory: `send_madev` is the same estimator stack (MAD, power spectrum, per-channel diffusion
# and spectral exponents, per-unit Fano curves), so the two cohorts cannot drift apart.
#
# Usage: `./run_calculations_visual_coding.jl [fc|bo]`, defaulting to the functional-connectivity
# cohort. Output goes to `visual_coding_calcdir(cohort)`; `collect_calculations_visual_coding.jl`
# reads it and `scripts/FigS1_visual_coding.jl` draws it.
#
# Restartable: `calcquality` skips any (session, structure) whose file already loads with a complete
# key set, so a killed run resumes where it stopped. Only "spontaneous" is analysed --- the other
# two pipeline stimuli are behavioural and have no Visual Coding counterpart.
using DrWatson
@quickactivate "WRExperiment"
using WRExperiment
using MoreMaps
using Distributed
using DimensionalData
using FileIO
using JLD2
using TimeseriesTools
using Statistics
import AllenNeuropixelsBase as AN
import TimeseriesTools: FFTW
FFTW.set_num_threads(1)   # FFTW planning is not thread-safe here; see fftw-threads-segfault

const COHORT = isempty(ARGS) ? :functional_connectivity :
    (lowercase(ARGS[1]) in ("bo", "brain_observatory") ? :brain_observatory :
     :functional_connectivity)
const NWORKERS = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 7

path = visual_coding_calcdir(COHORT)
mkpath(path)
sessions = visual_coding_sessions(COHORT)
stimulus = VISUAL_CODING_STIMULUS

@info "Visual Coding calculations" cohort=COHORT nsessions=length(sessions) structures path
@assert contains(AN.datadir, "AllenNeuropixelsBase.jl") "unexpected Allen cache at $(AN.datadir)"

# Only the six visual areas; the thalamic regions `run_calculations.jl` adds are not used by any
# Figure S1 panel and would triple the run for nothing.
_params = Iterators.product(sessions, unique(structures)) |> collect

Q = calcquality(path)
params = []
for p in _params
    sessionid, structure = p
    done = try
        Q[stimulus = At(stimulus), Structure = At(structure), SessionID = At(sessionid)]
    catch
        false   # absent from the quality table (never run, or errored) --- queue it
    end
    done || push!(params, p)
end
@info "queued" todo=length(params) of=length(_params)

if !isempty(params)
    if contains(gethostname(), "physics.usyd.edu.au")
        exprs = map(params) do (sessionid, structure)
            quote
                using Pkg
                Pkg.instantiate()
                import AllenNeuropixelsBase as AN
                using WRExperiment
                import TimeseriesTools: FFTW
                FFTW.set_num_threads(1)
                WRExperiment.send_madev(
                    $sessionid, $stimulus, $structure;
                    outpath = $(path)
                )
            end
        end
        batches = [exprs[i:min(i + 63, length(exprs))] for i in 1:64:length(exprs)]
        for batch in batches
            WRExperiment.submit_calculations(batch, mem = 32, ncpus = 3, walltime = 8,
                queue = `defaultQ`)
        end
    else
        addprocs(NWORKERS)
        @everywhere import AllenNeuropixelsBase as AN
        @everywhere using WRExperiment
        @everywhere using MoreMaps
        @everywhere import TimeseriesTools: FFTW
        @everywhere FFTW.set_num_threads(1)

        map(Chart(MoreMaps.Pmap(), LogLogger()), params) do (sessionid, structure)
            @info "Calculating $(stimulus) in $(structure) for session $(sessionid)"
            WRExperiment.send_madev(sessionid, stimulus, structure; outpath = path)
            GC.gc()
        end
    end
end
@info "done" path
