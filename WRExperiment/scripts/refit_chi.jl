#! /bin/bash
#=
exec julia +1.12 -t auto --project="$(dirname "${BASH_SOURCE[0]}")/.." "${BASH_SOURCE[0]}" "$@"
=#
# Re-fit the per-channel spectral exponent `chi` in the cached calculation files, in place.
#
# `chi` is fit inside `send_madev`, so a change to `PSD_RANGE` would normally mean re-running
# `run_calculations.jl` --- which re-downloads every session's LFP. But the fit only reads `S`,
# which those same files already cache, so re-fitting is a local read-fit-write over ~1900 small
# files instead of days of downloads. Everything else in each file is round-tripped untouched.
#
# After this: delete `datadir("mad_psd.jld2")` (it has the old `chi` stacked into it), then re-run
# `collect_calculations.jl`.
using DrWatson
@quickactivate :WRExperiment
using MoreMaps
using JLD2
using TimeseriesTools

path = DrWatson.datadir("calculations")
files = filter(f -> endswith(f, ".jld2") && !endswith(f, ".refit.jld2"), readdir(path; join = true))
@info "Re-fitting chi over PSD_RANGE = $(WRExperiment.PSD_RANGE) Hz in $(length(files)) files"

out = map(Chart(LogLogger(), Threaded()), files) do file
    d = load(file)
    haskey(d, "S") || return nothing # error files carry only "error"
    chi = map(s -> last(WRExperiment.mapple_fit(s))[:χ], eachslice(d["S"], dims = 2))
    d["chi"] = chi .|> Float32
    tmp = file * ".refit.jld2" # keep the .jld2 extension: FileIO dispatches the saver on it
    tagsave(tmp, d)
    mv(tmp, file; force = true) # atomic swap: a kill mid-write can't leave a half-written cache
    return file
end

n = count(!isnothing, out)
@info "Re-fit $n files ($(length(files) - n) skipped as error files)"
