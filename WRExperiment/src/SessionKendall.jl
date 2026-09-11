# The per-session hierarchy correlation behind Figure 4c and Figure S1f, and the two helpers it
# needs at its call sites (session alignment, and Benjamini-Hochberg over the cells a panel draws).
#
# Standalone by design: it uses only Statistics, Random and StatsBase. The figure scripts activate
# the root project, which does not depend on WRExperiment (a workspace member), so they `include`
# this file directly. `Patch.jl` includes it too, so the package exports the same code the figures
# run and there is one implementation rather than one per script.

using Statistics
using Random
import StatsBase: corkendall

"""
    sessionmatrix(ids, vals) -> (Y, sessions)

Assemble the `(session × area)` matrix `Y` from one `(ids, vals)` pair per area, aligning areas on
the session identifiers they share. `ids[j]` are the session identifiers behind `vals[j]`; column
`j` of `Y` is area `j`, row `i` is session `sessions[i]`, and `sessions` is the sorted intersection.

Alignment must be by identifier, never by position: the areas do not carry the same session lists
(in Visual Behaviour, VISp has 68 sessions against the other five areas' 69), so pairing the `i`th
entry of each area's vector silently correlates one session's value in one area against a different
session's value in another.
"""
function sessionmatrix(ids, vals)
    length(ids) == length(vals) ||
        throw(ArgumentError("need one id vector per value vector, got $(length(ids)) and $(length(vals))"))
    all(length.(ids) .== length.(vals)) ||
        throw(ArgumentError("each id vector must match the length of its value vector"))
    sessions = sort(intersect(collect.(ids)...))
    Y = Matrix{Float64}(undef, length(sessions), length(ids))
    for j in eachindex(ids)
        lookup = Dict(zip(ids[j], vals[j]))
        Y[:, j] = [Float64(lookup[s]) for s in sessions]
    end
    return Y, sessions
end

"""
    sessionkendall(x, Y; minareas = 4, N = 10_000, seed = 42)

Per-session Kendall correlation of one exponent against the hierarchy scores `x`, for the
`(session × area)` matrix `Y`. Each session is correlated over its own non-`NaN` areas and
contributes only if it retains at least `minareas` of them, so a ragged cohort (Visual Coding, where
no session records all six areas) is handled without confounding coverage with hierarchy.

Returns `(; tau, ci, p, taus, meantau, nsessions, nareas)`. `tau` is the across-session median and
`ci` its percentile bootstrap interval over sessions; `taus` are the per-session values, kept so
callers can draw the distribution the median came from; `nareas` is the areas each retained session
contributed.

Sessions, not area-points, are the independent unit: the bootstrap resamples sessions and the
permutation shuffles hierarchy labels within each session, among only the areas that session has.
Correlating within session removes the between-session variance that dominates these exponents and
that a pooled correlation spends most of its pairs on.

`p` tests the across-session **mean** τ against a null of no hierarchy ordering, by the add-one
estimator `(1 + #{|τ*| ≥ |τ_obs|}) / (N + 1)`, which cannot return zero. The mean rather than the
median is the test statistic because six areas put each session's τ on a 1/15 grid, so the median is
grid-valued and its permutation null collapses onto a handful of atoms: with 68 sessions, 44% of
null medians can land exactly on the observed value, leaving the p-value to be decided by the
tie convention (0.91 including ties against 0.03 excluding them) and its null distribution grossly
non-uniform (median p ≈ 0.9, rejection rate at α = 0.05 of 0.02). The mean of the same per-session
τ is not grid-locked, is calibrated under the null, and orders the cells identically. The median is
still what is reported and drawn; only the test statistic differs.

`p` is raw. Correct it across the family the caller draws with [`bhadjust`](@ref); this function
does not know the family.

Sessions are weighted equally in the median whatever their coverage, so a session with four areas
counts as much as one with six despite the larger sampling variance of its τ. With coverage
independent of hierarchy this costs a little power and biases nothing.
"""
function sessionkendall(x, Y; minareas = 4, N = 10_000, seed = 42)
    size(Y, 2) == length(x) ||
        throw(DimensionMismatch("Y has $(size(Y, 2)) areas, x has $(length(x)) hierarchy scores"))
    x = collect(float.(x))
    rows = [i for i in axes(Y, 1) if count(!isnan, view(Y, i, :)) >= minareas]
    areas = [findall(!isnan, view(Y, i, :)) for i in rows]           # areas each session actually has
    ys = [Float64.(Y[i, o]) for (i, o) in zip(rows, areas)]
    taus = [corkendall(x[o], y) for (o, y) in zip(areas, ys)]

    isempty(taus) && return (; tau = NaN, ci = (NaN, NaN), p = NaN, taus,
        meantau = NaN, nsessions = 0, nareas = Int[])

    tau, meantau = median(taus), mean(taus)
    # Separate streams so the interval does not depend on whether the test ran, or on `N`.
    ci = _percentileci(MersenneTwister(seed), taus, N)
    prng = MersenneTwister(seed + 1)
    ge = count(1:N) do _
        m = mean(corkendall(x[shuffle(prng, o)], y) for (o, y) in zip(areas, ys))
        abs(m) >= abs(meantau) - 1e-12                              # ties count against rejection
    end
    return (; tau, ci, p = (1 + ge) / (N + 1), taus, meantau,
        nsessions = length(taus), nareas = length.(areas))
end

"Percentile bootstrap interval for the median of `v`, resampling `v` itself."
function _percentileci(rng, v, N, α = 0.05)
    n = length(v)
    meds = [median(v[rand(rng, 1:n, n)]) for _ in 1:N]
    return Tuple(quantile(meds, (α / 2, 1 - α / 2)))
end

"""
    bhadjust(p)

Benjamini-Hochberg adjusted p-values, over the finite entries of `p` (non-finite entries stay
non-finite and are excluded from the family size). Local rather than `MultipleTesting.adjust` so the
figure scripts, which cannot load WRExperiment, correct their own families; `test/runtests.jl` pins
it against `MultipleTesting`.
"""
function bhadjust(p)
    q = fill(NaN, length(p))
    ok = findall(isfinite, p)
    n = length(ok)
    n == 0 && return q
    o = ok[sortperm([p[i] for i in ok])]
    m = 1.0
    for k in n:-1:1                       # step-up, enforcing monotonicity as it goes
        m = min(m, n * p[o[k]] / k)
        q[o[k]] = m
    end
    return q
end
