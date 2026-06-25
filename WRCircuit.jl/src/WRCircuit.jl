module WRCircuit
# Native-Julia backend: the BrainPy/JAX solver has been swapped out for the native Dewdrop engine
# (../../../Dewdrop.jl). The simulation model + run facade live in `ModelInterface.jl`; the public
# surface that scripts use (`models.Spatial`, `bpsolve`, `PRNGKey`) and the shape of the returned /
# saved data (a `Population × Var` ToolsArray of `Timeseries`/`SpikeTrain`s) are unchanged, so existing
# scripts and the saved `.jld2` format are byte-compatible with the old BrainPy outputs.
using DrWatson
using Dewdrop
using CUDA
using TerminalLoggers: TerminalLogger
import Logging

export convert2, terminal_logging!

const DEWDROP_BACKEND = Dewdrop.GPU

# Kept for script compatibility: was a PythonCall `pyconvert` wrapper; native results are already Julia
# values, so this is just a typed converter (`convert2(Float32)(x) == convert(Float32, x)`).
convert2(T::Type) = Base.Fix1(convert, T)

include("SpatialNetwork.jl")    # builder-native spatial FNS model (first-class Dewdrop components)
include("ModelInterface.jl")
include("Utils.jl")
include("Plots.jl")

# Native statistical observables --- now first-class in Dewdrop (ports of the old Python `src/stats.py`).
# Each operates on a raw [`simulate`](@ref) solution rather than the formatted `bpsolve` output, e.g.
# `susceptibility(sol; bin, of=:E)`, `mua(sol; …)`, `radial_autocorrelation(sol; dr, of)`,
# `power_spectrum(sol; n_segments, of)`, `cv_isi(sol; of)`, `firing_rate(sol, :E)`,
# `temporal_average(sol, :V; of)`. Grouped here (qualified --- avoids clashing with the TimeseriesTools
# `coarsegrain`/Utils helpers) to mirror the old `WRCircuit.stats.*` access. Not exported; use
# `WRCircuit.stats.<name>` or, since the engine is `using`-ed, `Dewdrop.<name>` directly.
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

# --- REPL progress logging ----------------------------------------------------------------------
"""
    terminal_logging!()

Route progress bars and logs to the REPL terminal via TerminalLoggers, replacing VSCode's native progress
widget --- so `simulate`/`bpsolve` progress renders inline in the REPL. Installed automatically on load in
interactive sessions; call it manually to re-enable (e.g. after another package resets the global logger).
Restore the default logger with `Logging.global_logger(Logging.ConsoleLogger())` or by restarting the REPL.
"""
terminal_logging!() = (Logging.global_logger(TerminalLogger()); nothing)

# Install the terminal logger on load: interactive sessions only (scripts/CI keep their own logger), and
# never while generating compiled output (`jl_generating_output` --- so we don't mutate global logging state
# during precompilation, of this package or anything downstream). `TerminalLogger()`'s default `min_level`
# is `ProgressLevel` (LogLevel(-1)), matching Dewdrop's progress records, so progress bars render.
function __init__()
    if ccall(:jl_generating_output, Cint, ()) == 0 && isinteractive()
        terminal_logging!()
    end
    return nothing
end

end # module
