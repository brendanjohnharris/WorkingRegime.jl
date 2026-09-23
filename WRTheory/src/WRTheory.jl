module WRTheory
const _preamble = quote
    using DelimitedFiles
    using DrWatson
    using Distributions
    using TerminalLoggers
    using ProgressLogging
    using Logging
    using TimeseriesTools
    using CairoMakie
    using TimeseriesMakie
    using Fathom
    using MoreMaps
    using StatsBase
    using FractionalNeuralSampling
    using DiffEqNoiseProcess
    using Optim
    import FractionalNeuralSampling: Density
    set_theme!(fathom())
    global_logger(TerminalLogger(right_justify = 200))
    return nothing
end
Base.eval(WRTheory, _preamble)
macro preamble()
    return _preamble
end

const connector = '&'

"""
    rootdatadir(args...)

Path under the repository's `data/WRTheory/`, which holds every file a figure script reads
(intermediates stay in `datadir()`). Anchored on the package folder, not the active project.
"""
rootdatadir(args...) = joinpath(dirname(pkgdir(WRTheory)), "data", "WRTheory", args...)

export connector, rootdatadir, variability_exponent, variability_model, VARIABILITY_SEED,
    VARIABILITY_PINS, VARIABILITY_WIDTH, VARIABILITY_NOFIT

include("Variability.jl")
include("Utils.jl")
include("Circuit.jl")

end
