module WRExperiment
using DrWatson
using CairoMakie
using DimensionalData
import DimensionalData: @dim
using Statistics
using TimeseriesTools
import TimeseriesTools: bootstrapmedian  # owned binding so `export bootstrapmedian` re-exports TT's (BCa via BootstrapExt)
import Optim, ForwardDiff  # loaded (not used directly) so TimeseriesTools' OptimExt --- MAPPLE's fit! --- activates
using Preferences # Should load this and the next line for preferences to work properly
using AllenNeuropixelsBase
import AllenNeuropixelsBase: Unit  # owned binding so `export Unit` below actually re-exports it to the scripts
using AcademicClusters
using Random

# Dimensions vendored from SpatiotemporalMotifs (defined via @dim there too). NOTE: these are distinct
# types from SM's same-named dims, so anything crossing the WRExperiment/SM boundary (e.g. indexing
# calcquality's Q) must use WRExperiment's dims consistently.
@dim SessionID ToolsDim "SessionID"
@dim Trial ToolsDim "Trial"
@dim Structure ToolsDim "Structure"


include("Patch.jl")
include("Surrogates.jl")

# Public API for the scripts (replaces what they used to get from `using SpatiotemporalMotifs`).
# `Unit` is re-exported from AllenNeuropixelsBase; the rest are vendored/defined here.
export structures, layers, PTHR, hierarchy_scores, THETA, GAMMA, bootstrapmedian, val_to_string,
    calcquality, calcdir, savepath, SessionID, Trial, Structure, Unit, CLUSTER, DEFAULT_SESSION_ID,
    hierarchicalkendall, mediankendallpvalue, @preamble,
    send_madev, produce_unitdepths, madev, mapple_fit, diffusion_fit, diffusion_line, diffusion_knee,
    confirm_band_before_knee, fano_factor, rates, plotspectrum!,
    parselayernum, commondepths,
    madev_taus, surrogate_null, surrogatep, surrogatez,
    surrogate_diffusion_exponent, surrogate_kurtosis, lfp_surrogate_stats, send_surrogates

function submit_calculations(exprs; queue = ``, mem = 50, ncpus = 8, walltime = 8, exeflags = `+1.12`)
    exprs = deepcopy(exprs)
    return if length(exprs) > 2
        shuffle!(exprs) # ? Shuffle so restarted calcs are more even
        AcademicClusters.USydPhysics.runscripts(
            exprs; ncpus, mem,
            walltime,
            project = Cmd([projectdir()]),
            exeflags,
            queue = queue
        )
    else
        AcademicClusters.USydPhysics.runscript.(
            exprs; ncpus, mem,
            walltime,
            project = Cmd([projectdir()]),
            exeflags,
            queue = queue
        )
    end
end

end
