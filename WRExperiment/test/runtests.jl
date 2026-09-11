#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 --project="$(dirname "${BASH_SOURCE[0]}")/.." "${BASH_SOURCE[0]}" "$@"
=#
# `sessionkendall` and its two call-site helpers. `src/SessionKendall.jl` is included directly
# rather than through `using WRExperiment`: the estimator depends only on Statistics, Random and
# StatsBase, and loading the package would drag in AllenNeuropixelsBase and a Conda environment for
# no benefit.
using Test
using Statistics
using Random
import StatsBase: corkendall
import MultipleTesting
import HypothesisTests: SignedRankTest, pvalue

include(joinpath(@__DIR__, "..", "src", "SessionKendall.jl"))

const HIER = [-0.357, -0.093, -0.059, 0.152, 0.327, 0.441]   # Siegle 2021, the six visual areas

@testset "sessionmatrix" begin
    # Alignment is by identifier, not position. This is the defect the Figure 4 rework carried:
    # VISp holds 68 sessions against the other areas' 69, and truncating each area's vector to the
    # shortest length paired 40 of the 68 rows across different sessions.
    ids = [[10, 20, 40], [10, 20, 30, 40], [10, 20, 30, 40]]
    vals = [[1.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    Y, sessions = sessionmatrix(ids, vals)
    @test sessions == [10, 20, 40]
    @test Y == [1.0 1.0 1.0; 2.0 2.0 2.0; 4.0 4.0 4.0]
    # Positional truncation would have given row 3 as [4, 3, 3]: the bug this guards against.
    @test Y[3, :] != [4.0, 3.0, 3.0]

    # Order of the incoming ids must not matter.
    Yp, sp = sessionmatrix([[40, 10, 20], ids[2], ids[3]], [[4.0, 1.0, 2.0], vals[2], vals[3]])
    @test sp == sessions && Yp == Y

    @test_throws ArgumentError sessionmatrix(ids, vals[1:2])
    @test_throws ArgumentError sessionmatrix(ids, [vals[1][1:2], vals[2], vals[3]])
end

@testset "monotone design" begin
    rng = MersenneTwister(1)
    Y = [HIER[j] + 0.001 * randn(rng) for i in 1:30, j in 1:6]   # noise far below the area spacing
    r = sessionkendall(HIER, Y; N = 2000)
    @test r.tau == 1.0
    @test all(==(1.0), r.taus)
    @test r.nsessions == 30
    @test r.p == 1 / 2001                       # the add-one floor, never exactly zero
    @test r.ci == (1.0, 1.0)
end

@testset "add-one p-value floor" begin
    rng = MersenneTwister(2)
    Y = [HIER[j] + 0.001 * randn(rng) for i in 1:20, j in 1:6]
    for N in (100, 1000)
        @test sessionkendall(HIER, Y; N).p == 1 / (N + 1)
    end
end

@testset "session offsets do not attenuate" begin
    # The whole reason for the within-session construction: an offset several times the area effect
    # leaves every session's internal ranking intact but destroys a pooled correlation.
    rng = MersenneTwister(3)
    S, β, σs = 60, 1.0, 8.0
    Y = [β * HIER[j] for i in 1:S, j in 1:6] .+ (σs .* randn(rng, S)) .+ 0.01 .* randn(rng, S, 6)
    r = sessionkendall(HIER, Y; N = 2000)
    pooled = corkendall(repeat(HIER, S), vec(permutedims(Y)))

    @test r.tau == 1.0                          # within-session: the area effect is recovered intact
    @test pooled < 0.3                          # pooled: visibly attenuated by the offsets
    @test r.tau > 3 * pooled
end

@testset "ragged coverage" begin
    rng = MersenneTwister(4)
    S = 80
    Y = [HIER[j] + 0.3 * randn(rng) for i in 1:S, j in 1:6]
    for i in 1:S                                # drop 1-2 areas at random from every session
        for j in randperm(rng, 6)[1:rand(rng, 1:2)]
            Y[i, j] = NaN
        end
    end
    r = sessionkendall(HIER, Y; minareas = 4, N = 500)
    @test !any(isnan, r.taus)
    @test !isnan(r.tau) && all(isfinite, r.ci) && isfinite(r.p)
    @test all(>=(4), r.nareas)
    @test r.nsessions == count(i -> count(!isnan, view(Y, i, :)) >= 4, 1:S)

    # `minareas` is a real gate, not decoration: raising it can only drop sessions.
    @test sessionkendall(HIER, Y; minareas = 5, N = 100).nsessions <= r.nsessions
    @test sessionkendall(HIER, Y; minareas = 6, N = 100).nsessions ==
        count(i -> !any(isnan, view(Y, i, :)), 1:S)

    # A session below the gate must not contribute a degenerate tau in {-1, +1}.
    Z = fill(NaN, 3, 6)
    Z[1, 1:2] = [0.0, 1.0]                      # two areas only
    Z[2:3, :] = [HIER'; HIER']
    @test sessionkendall(HIER, Z; minareas = 4, N = 100).nsessions == 2

    # Nothing survives the gate: a defined, non-throwing empty result.
    empty = sessionkendall(HIER, fill(NaN, 4, 6); minareas = 4, N = 100)
    @test empty.nsessions == 0 && isnan(empty.tau) && isnan(empty.p)
end

@testset "reproducibility" begin
    rng = MersenneTwister(5)
    Y = [0.5 * HIER[j] + randn(rng) for i in 1:40, j in 1:6]
    a = sessionkendall(HIER, Y; N = 500)
    b = sessionkendall(HIER, Y; N = 500)
    @test a.tau == b.tau && a.ci == b.ci && a.p == b.p && a.taus == b.taus
    @test sessionkendall(HIER, Y; N = 500, seed = 43).p != a.p    # the seed is actually used
end

@testset "null calibration" begin
    # Exponents independent of hierarchy: the p-values must be roughly uniform. This is the test
    # that fails if the median is used as the test statistic instead of the mean --- with tau on a
    # 1/15 grid the null medians collapse onto a handful of atoms, and the rejection rate falls to
    # ~0.02 with a median p-value near 0.9 rather than 0.5.
    rng = MersenneTwister(6)
    ps = [sessionkendall(HIER, randn(rng, 40, 6); N = 199).p for _ in 1:200]
    @test 0.01 <= mean(ps .< 0.05) <= 0.12          # nominal 0.05; +-3 binomial SE is +-0.046
    @test 0.05 <= mean(ps .< 0.10) <= 0.20          # nominal 0.10
    @test 0.35 <= median(ps) <= 0.65                # nominal 0.5
end

@testset "agrees with a signed-rank cross-check" begin
    # Not the reported statistic, but a real effect should be flagged by both, and a null by
    # neither. Only the ordering is asserted: the two tests are not expected to agree numerically.
    rng = MersenneTwister(7)
    signal = [0.4 * HIER[j] + 0.35 * randn(rng) for i in 1:60, j in 1:6] .+ 2.0 .* randn(rng, 60)
    null = randn(rng, 60, 6)
    for (Y, expect) in ((signal, true), (null, false))
        r = sessionkendall(HIER, Y; N = 2000)
        w = pvalue(SignedRankTest(Float64.(r.taus)))
        @test (r.p < 0.01) == expect
        @test (w < 0.01) == expect
    end
end

@testset "bhadjust" begin
    for p in ([0.001, 0.008, 0.039, 0.041, 0.042, 0.6, 0.9],
            [0.5, 0.01, 0.3, 0.02], collect(range(1.0e-6, 0.9; length = 12)))
        @test bhadjust(p) ≈ MultipleTesting.adjust(collect(p), MultipleTesting.BenjaminiHochberg())
    end
    @test all(bhadjust([0.5, 0.5, 0.5]) .≈ 0.5)     # monotone enforcement, no value above 1
    @test all(<=(1.0), bhadjust([0.9, 0.95, 0.99]))

    q = bhadjust([0.01, NaN, 0.9])                  # non-finite entries pass through, family is 2
    @test isnan(q[2])
    @test q[[1, 3]] ≈ MultipleTesting.adjust([0.01, 0.9], MultipleTesting.BenjaminiHochberg())
    @test all(isnan, bhadjust([NaN, NaN]))
end
