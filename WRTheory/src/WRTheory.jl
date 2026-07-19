module WRTheory
const _preamble = quote
    using DelimitedFiles
    using DrWatson
    using Distributions
    using TerminalLoggers
    using ProgressLogging
    using Logging
    using TimeseriesTools
    using ComplexityMeasures
    using CairoMakie
    using TimeseriesMakie
    using Fathom
    using MoreMaps
    using StatsBase
    using FractionalNeuralSampling
    using DiffEqNoiseProcess
    using Optim
    import FractionalNeuralSampling: Density
    set_theme!(foresight(:physics))
    global_logger(TerminalLogger(right_justify = 200))
    return nothing
end
Base.eval(WRTheory, _preamble)
macro preamble()
    return _preamble
end

plotdir(args...) = projectdir("plots", args...)
const connector = '&'
export plotdir, connector

include("Utils.jl")
include("Makie.jl")
include("Circuit.jl")

end
