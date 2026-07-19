module WRCircuit
using DrWatson
using Dewdrop
using CUDA
using TerminalLoggers: TerminalLogger
import Logging

export convert2

const DEWDROP_BACKEND = Dewdrop.GPU

# Kept for script compatibility: was a PythonCall `pyconvert` wrapper; native results are already Julia
# values, so this is just a typed converter (`convert2(Float32)(x) == convert(Float32, x)`).
convert2(T::Type) = Base.Fix1(convert, T)

include("SpatialNetwork.jl")
include("ModelInterface.jl")
include("Utils.jl")
include("Plots.jl")

const stats = (;
    firing_rate = Dewdrop.firing_rate,
    susceptibility = Dewdrop.susceptibility,
    mua = Dewdrop.mua,
    radial_autocorrelation = Dewdrop.radial_autocorrelation,
    power_spectrum = Dewdrop.power_spectrum,
    cv_isi = Dewdrop.cv_isi,
    temporal_average = Dewdrop.temporal_average,
    coarsegrain = Dewdrop.coarsegrain,
    grand_distribution = Dewdrop.grand_distribution,
    efficiency = Dewdrop.efficiency,
)


end # module
