#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Diagnostic (not in the paper): test the Section IV claim that the unconfined bFNS diffusion
# exponent follows a ≈ 1 - α/2 + β/2, against the saved flat sweeps
# (data/WRTheory/bFNS_sweep/flat_*.jld2; per-cell 1-comp MAPPLE fits to the 0.1-10 ms MAD, 10 repeats).
# Also scores two alternative surfaces: the noise self-similarity H = 1/2 + 1/α - β/2 and the
# space-time fractional-diffusion similarity β/α. Writes a summary txt + per-cell residual TSV
# to plots/checks/.
using DrWatson
@quickactivate "WRTheory"
import WRTheory: rootdatadir
using JLD2
using TimeseriesTools
using Statistics
using DelimitedFiles

nanmean(x) = (v = filter(!isnan, vec(collect(x))); isempty(v) ? NaN : mean(v))

"Collapse the stored (α, β, γ, η, Obs) diffusion-exponent array to a 2-D (α, β) grid by
NaN-aware averaging over repeats (mean, matching the Fig 2 heatmaps)."
function ab_grid(file)
    A = load(rootdatadir("bFNS_sweep", file), "diffusion_exponent")
    da = parent(A)
    αs = try
        collect(lookup(A, :α))
    catch
        collect(range(1.2, 2.0, length = size(da, 1)))
    end
    βs = try
        collect(lookup(A, :β))
    catch
        collect(range(0.2, 1.0, length = size(da, 2)))
    end
    colons = ntuple(_ -> Colon(), ndims(da) - 2)
    a = [nanmean(@view da[i, j, colons...]) for i in eachindex(αs), j in eachindex(βs)]
    return αs, βs, a
end

const CANDIDATES = (
    claimed = (α, β) -> 1 - α / 2 + β / 2,      # manuscript Section IV
    noise_H = (α, β) -> 1 / 2 + 1 / α - β / 2,  # noise self-similarity
    beta_over_alpha = (α, β) -> β / α,          # fractional-diffusion similarity
    # Exact similarity of the position process: x = I^β ξ with the integrated noise H-sss, so
    # a = H + (β - 1) = 1/α + β/2 - 1/2. The claimed linear relation is this curve's chord in α
    # between α = 1 and α = 2 (they coincide exactly at both endpoints).
    similarity = (α, β) -> 1 / α + β / 2 - 1 / 2,
)

function check(file, io, tsvpath)
    αs, βs, a = ab_grid(file)
    valid = findall(!isnan, a)   # invalid-H cells are NaN at simulation time
    av = [a[I] for I in valid]
    αv = [αs[I[1]] for I in valid]
    βv = [βs[I[2]] for I in valid]

    println(io, "file: ", file)
    println(io, "valid cells (0 < H < 1): ", length(valid), " / ", length(a))

    # Best-fit linear surface, to compare against each candidate's coefficients
    X = hcat(ones(length(av)), αv, βv)
    c = X \ av
    println(io, "OLS a ~ c0 + c_α α + c_β β: c0 = ", round(c[1]; digits = 3),
        ", c_α = ", round(c[2]; digits = 3), ", c_β = ", round(c[3]; digits = 3),
        "   [claimed relation: 1, -0.5, +0.5]")

    preds = Dict{Symbol, Vector{Float64}}()
    for (name, f) in pairs(CANDIDATES)
        pv = f.(αv, βv)
        preds[name] = pv
        r = av .- pv
        println(io, rpad(string(name), 16),
            " mean resid = ", round(mean(r); digits = 4),
            ", median|r| = ", round(median(abs.(r)); digits = 4),
            ", p90|r| = ", round(quantile(abs.(r), 0.9); digits = 4),
            ", max|r| = ", round(maximum(abs.(r)); digits = 4),
            ", pearson = ", round(cor(av, pv); digits = 4))
    end

    # Canonical operating point
    i0 = argmin(abs.(αs .- 1.5))
    j0 = argmin(abs.(βs .- 0.8))
    println(io, "nearest cell to (α, β) = (1.5, 0.8): (", round(αs[i0]; digits = 4), ", ",
        round(βs[j0]; digits = 4), "): a = ", round(a[i0, j0]; digits = 4),
        "; ", join(
            [string(n) * " = " * string(round(f(αs[i0], βs[j0]); digits = 4))
             for (n, f) in pairs(CANDIDATES)], ", "))
    println(io)

    names = collect(keys(CANDIDATES))
    hdr = permutedims(vcat(["alpha", "beta", "a_measured"], ["a_" * string(n) for n in names]))
    rows = hcat(αv, βv, av, [preds[n] for n in names]...)
    writedlm(tsvpath, vcat(hdr, rows), '\t')
    return nothing
end

outdir = plotsdir("checks")
mkpath(outdir)
open(joinpath(outdir, "diffusion_relation_check.txt"), "w") do io
    for file in ("flat_γ=0.03_η=0.01.jld2", "flat_γ=0.0_η=0.01.jld2")
        tsv = joinpath(outdir,
            "diffusion_relation_" * replace(file, ".jld2" => "") * ".tsv")
        check(file, io, tsv)
    end
end
print(read(joinpath(outdir, "diffusion_relation_check.txt"), String))
