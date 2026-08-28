#! /bin/bash
#=
exec julia +1.12 -t auto "${BASH_SOURCE[0]}" "$@"
=#
# Collect the surrogate sweep into per-layer group statistics. Per session and layer: the median over
# that layer's channels of each statistic, for the data and for each surrogate draw (the aggregation
# is replicated inside the null), giving a paired per-session effect Δ = data − mean(null). Group
# level: a one-sided paired signed-rank across sessions, the quoted p.
#
# Both statistics are one-sided LARGER than the null: excess kurtosis (heavier-tailed than the
# spectrum-matched Gaussian null, whose own value is exactly 0) and the diffusion exponent (against
# the FT null, Δa ≈ ζ(1) − ζ(2)/2, the first-order intermittency coefficient).
using DrWatson
@quickactivate "WRExperiment"
using WRExperiment
using FileIO
using JLD2
using Statistics
import HypothesisTests: SignedRankTest, pvalue

method = isempty(ARGS) ? :ft : Symbol(first(ARGS))
stimulus = "spontaneous"
path = DrWatson.datadir(method === :iaaft ? "surrogates" : "surrogates_$(method)")
const LAYERS = [1 => "L1", 2 => "L2/3", 3 => "L4", 4 => "L5", 5 => "L6"]
const STATS = [:kurt => "excess kurtosis", :a => "diffusion exponent"]

files = filter(contains("stimulus=$(stimulus)"), readdir(path; join = true))
rows = map(files) do f
    D = load(f)
    haskey(D, "error") && return missing
    s0, s, lnum = D["s0"], D["s"], D["layernums"]
    m = match(r"sessionid=(\d+).*structure=([A-Za-z0-9\-]+)\.jld2", basename(f))
    map(LAYERS) do (code, _)
        sel = findall(lnum .== code)
        length(sel) < 2 && return missing
        agg(v, j) = median(filter(!isnan, getfield(v, j)[sel]))
        (; sessionid = parse(Int, m[1]), structure = String(m[2]), layer = code, nch = length(sel),
            data = Dict(j => agg(s0, j) for j in first.(STATS)),
            eff = Dict(j => agg(s0, j) - mean(agg.(s, j)) for j in first.(STATS)))
    end
end
rows = collect(skipmissing(reduce(vcat, collect.(skipmissing(rows)))))
@info "Collected $(length(unique(getfield.(rows, :sessionid)))) sessions, " *
    "$(length(unique(getfield.(rows, :structure)))) structures"

for (code, name) in LAYERS, (j, label) in STATS
    lr = filter(r -> r.layer == code, rows)
    isempty(lr) && continue
    d = [r.eff[j] for r in lr]
    println(
        rpad(name, 6), rpad(label, 20), " N = ", lpad(length(d), 4),
        "   data median = ", lpad(round(median([r.data[j] for r in lr]), digits = 4), 8),
        "   Δ = ", lpad(round(median(d), digits = 4), 8),
        "   p = ", lpad(round(pvalue(SignedRankTest(d); tail = :right), sigdigits = 2), 9),
        "   consistent ", count(>(0), d), "/", length(d)
    )
end

outfile = DrWatson.datadir("surrogate_stats" * (method === :ft ? "" : "_$(method)") * ".jld2")
tagsave(outfile, Dict("rows" => rows, "stimulus" => stimulus, "method" => string(method)))
@info "Saved" outfile
