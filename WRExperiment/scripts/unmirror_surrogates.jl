#! /bin/bash
#=
exec julia +1.12 -t auto --project="$(dirname "${BASH_SOURCE[0]}")/.." "${BASH_SOURCE[0]}" "$@"
=#
# One-off repair of the surrogate files written before the AllenNeuropixelsBase channel-order fix.
#
# `send_surrogates` computes its statistics from `parent(lfp)` and labels the columns from
# `lookup(LFP, Chan)`. The old `AN.sortbydepth` returned data mirrored against that lookup, and
# nothing in this path cancelled it (unlike `send_madev`, whose `set` call happened to), so every
# per-channel statistic is attached to the channel at the opposite end of the probe.
#
# The mirror is exact, so reversing the per-channel vectors repairs the files without recomputing
# any surrogates. Verified against a fresh computation with the fixed package: the stored values are
# `reverse` of the correct ones to 1e-7. The label vectors (channels, depths, layers, layernums,
# streamlinedepths) are already in the right order and are left alone.
#
# Idempotent: repaired files carry `channel_order_fixed`. Re-run `collect_surrogates.jl` afterwards.
using DrWatson
@quickactivate :WRExperiment
using MoreMaps
using JLD2

# Only per-channel vectors are mirrored. Scalars such as the IAAFT files' `a_med` (the median over
# channels) are order-invariant and must be left alone.
revfield(v, n) = (v isa AbstractVector && length(v) == n) ? reverse(v) : v
rev(nt::NamedTuple, n) = NamedTuple{propertynames(nt)}(map(v -> revfield(v, n), values(nt)))

paths = [DrWatson.datadir("surrogates"), DrWatson.datadir("surrogates_ft")]
for path in paths
    isdir(path) || continue
    files = filter(f -> endswith(f, ".jld2") && !endswith(f, ".fix.jld2"),
                   readdir(path; join = true))
    @info "Un-mirroring $(length(files)) files in $(basename(path))"
    out = map(Chart(LogLogger(), Threaded()), files) do file
        d = load(file)
        haskey(d, "s0") || return :error_file
        haskey(d, "channel_order_fixed") && return :already_fixed
        n = length(d["channels"])
        length(d["s0"].a) == n || return :unexpected_shape
        d["s0"] = rev(d["s0"], n)
        d["s"] = [rev(s, n) for s in d["s"]]
        d["channel_order_fixed"] = true
        tmp = file * ".fix.jld2" # keep the .jld2 extension: FileIO dispatches the saver on it
        tagsave(tmp, d)
        mv(tmp, file; force = true) # atomic swap
        return :fixed
    end
    @info "done" basename(path) counts=Dict(k => count(==(k), out) for k in unique(out))
end
