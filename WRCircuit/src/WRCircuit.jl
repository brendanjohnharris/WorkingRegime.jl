module WRCircuit
using DrWatson
using Dewdrop
using CUDA
using TerminalLoggers: TerminalLogger
import Logging

export convert2, variability_exponent, variability_model, VARIABILITY_SEED,
    VARIABILITY_PINS, VARIABILITY_WIDTH, VARIABILITY_NOFIT

const DEWDROP_BACKEND = Dewdrop.GPU

include("Variability.jl")
include("SpatialNetwork.jl")
include("ModelInterface.jl")
include("Utils.jl")
include("Sweep.jl")

end # module
