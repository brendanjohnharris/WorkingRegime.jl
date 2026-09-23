#! /bin/bash
#=
exec julia +1.13 -t auto "${BASH_SOURCE[0]}" "$@"
=#
# Surrogate nulls for the two quoted LFP statistics (per-channel excess kurtosis and diffusion
# exponent), one file per (session, stimulus, structure). Default null is FT, whose increments are
# Gaussian, so the kurtosis null value is exactly 0 and the data value IS the effect size. Scoped to
# the main-text claims: spontaneous, the six cortical areas, with layers selected at collect from the
# saved per-channel labels. Mirrors run_calculations.jl.
using DrWatson
@quickactivate "WRExperiment"
using WRExperiment
using MoreMaps
using Distributed
using DimensionalData
using FileIO
using JLD2
import AllenNeuropixelsBase as AN
import TimeseriesTools: FFTW
FFTW.set_num_threads(1)   # FFTW's own threads segfault under -t auto; see fftw-threads-segfault

method = isempty(ARGS) ? :ft : Symbol(first(ARGS)) # :ft (default) or :iaaft
path = rootdatadir(method === :iaaft ? "surrogates" : "surrogates_$(method)") # also the `outpath` below: one folder for the check and the writes
mkpath(path)
stimuli = ["spontaneous"]
session_table = load(DrWatson.datadir("session_table.jld2"), "session_table")
oursessions = session_table.ecephys_session_id

pstructures = deepcopy(structures) # the six cortical areas, for the hierarchy statistics
_params = Iterators.product(oursessions, stimuli, unique(pstructures)) |> collect

Q = calcquality(path; requirekeys = ["s0", "s", "n", "channels", "layernums"])
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
                import TimeseriesTools: FFTW
                FFTW.set_num_threads(1)
                WRExperiment.send_surrogates(
                    $o, $stimulus, $structure; n = 20, method = $(QuoteNode(method)), outpath = $path
                )
            end
        end
        batches = 1:8:length(exprs)
        batches = [exprs[i:min(i + 7, length(exprs))] for i in batches]
        for (i, batch) in enumerate(batches)
            WRExperiment.submit_calculations(
                batch, mem = 32, ncpus = 4,
                walltime = 4, # array elements are single conditions, a few minutes at 4 threads
                exeflags = `+1.13 -t auto`,
                queue = iseven(i) ? `taiji` : `defaultQ` # alternate queues for ~2x concurrency
            )
        end
    else
        map(Chart(LogLogger()), params) do param # serial over conditions; surrogates + fits thread inside
            @info "Calculating surrogates for $(param)"
            WRExperiment.send_surrogates(param...; n = 20, method, outpath = path)
            GC.gc()
        end
    end
end
