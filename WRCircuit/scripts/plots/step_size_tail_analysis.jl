#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes --project="$(dirname "${BASH_SOURCE[0]}")/.." "${BASH_SOURCE[0]}" "$@"
=#
# Investigate the input step-size (ΔI) tail: is the "flat/heavy" tail real or a
# binning/display artifact, and is it driven by positive vs negative deviations?
# Compares the current run (demo_run.jld2) against the δ=4 baseline (critical_demo.jld2).
using DrWatson
DrWatson.@quickactivate :WRCircuit
using JLD2
using Statistics
using CairoMakie
WRCircuit.@preamble

const OUT = "/tmp/claude-1785800720/-import-taiji1-bhar9988-code-DDC-WorkingRegime-jl/63a900dd-8731-4e81-a625-e3f6810996fa/scratchpad"

# Load E_input (last 5 s, 𝑡×Neuron), strip units → step×neuron Float matrix.
function load_input(f)
    isfile(f) || return nothing
    d = jldopen(f, "r") do g
        haskey(g, "E_input") || return nothing
        fp = haskey(g, "fixed_params") ? g["fixed_params"] : (;)
        println("  fixed_params keys for ", basename(f), ": ", keys(fp))
        haskey(fp, :delta) && println("    delta = ", fp.delta)
        (; M = parent(ustripall(g["E_input"])), delta = get(fp, :delta, NaN),
            dt = try step(ustripall(times(g["E_input"]))) catch; NaN end)
    end
    return d
end

# Increments per neuron, flattened over a neuron subsample (every `nsub`th).
function increments(M; nsub = 5)
    cols = 1:nsub:size(M, 2)
    dI = diff(@view M[:, cols]; dims = 1)   # (nstep-1) × ncols
    return vec(dI)
end

# CCDF on a fixed log-x grid (resolves the tail evenly, unlike index-downsampling).
function ccdf(x; npts = 300)
    x = sort(filter(v -> v > 0 && isfinite(v), x))
    n = length(x)
    n == 0 && return (Float64[], Float64[])
    xs = exp10.(range(log10(quantile(x, 0.5)), log10(x[end]), length = npts))  # body→tail
    surv = [ (n - searchsortedlast(x, xi)) / n for xi in xs ]                    # P(X > xi)
    keep = surv .> 0
    return (xs[keep], surv[keep])
end

# Hill tail-exponent estimator on the top-k fraction: α such that P(X>x) ~ x^-α.
function hill(x; frac = 0.001)
    x = sort(filter(v -> v > 0 && isfinite(v), x))
    n = length(x); k = max(50, round(Int, frac * n))
    k >= n && return NaN
    xmin = x[n - k]
    return 1 / mean(@view(x[n-k+1:n]) .|> v -> log(v / xmin))
end

# Log-binned pdf (geometric bins) — the honest way to read a power-law tail.
function logpdf(x; nbins = 45)
    x = filter(v -> v > 0 && isfinite(v), x)
    isempty(x) && return (Float64[], Float64[])
    lo, hi = quantile(x, 0.001), maximum(x)
    lo <= 0 && (lo = minimum(x[x .> 0]))
    edges = exp10.(range(log10(lo), log10(hi), length = nbins + 1))
    cnt = zeros(Int, nbins)
    for v in x
        b = searchsortedlast(edges, v)
        1 <= b <= nbins && (cnt[b] += 1)
    end
    w = diff(edges)
    centers = sqrt.(edges[1:end-1] .* edges[2:end])
    dens = cnt ./ (sum(cnt) .* w)
    keep = dens .> 0
    return (centers[keep], dens[keep])
end

# CCDF log-log slope over window [x0, x1] (α of P(X>x) ~ x^-α); operates on the grid CCDF.
function tail_slope(x, s; x0, x1)
    m = (x .>= x0) .& (x .<= x1) .& (s .> 0)
    sum(m) < 3 && return NaN
    lx, ls = log10.(x[m]), log10.(s[m])
    return -(cov(lx, ls) / var(lx))
end

datasets = [
    ("demo_run (δ=3.25, cv=0.2)", datadir("demo_run.jld2"), :current),
    ("baseline (critical_demo)",  datadir("critical_demo.jld2"), :base),
]

results = Dict{Symbol,Any}()
for (label, f, tag) in datasets
    d = load_input(f)
    d === nothing && (@warn "missing/no E_input" file=f; continue)
    dI = increments(d.M)
    pos = dI[dI .> 0]
    neg = -dI[dI .< 0]
    absx = abs.(dI)
    @info label delta=d.delta dt=d.dt n=length(dI) maxabs=maximum(absx) frac_ge4=mean(absx .>= 4.0) frac_pos=mean(dI .> 0)
    println("  |ΔI| quantiles (.9,.99,.999,.9999,max): ",
        round.(quantile(absx, [0.9,0.99,0.999,0.9999]); sigdigits=3), "  ", round(maximum(absx); sigdigits=3))
    println("  mean ΔI = ", round(mean(dI); sigdigits=3), "  skew(ΔI) proxy mean(ΔI^3)/std^3 = ",
        round(mean((dI .- mean(dI)).^3)/std(dI)^3; sigdigits=3))
    cabs, cpos, cneg = ccdf(absx), ccdf(pos), ccdf(neg)
    results[tag] = (; label, delta = d.delta,
        abs = cabs, pos = cpos, neg = cneg,
        pabs = logpdf(absx), ppos = logpdf(pos), pneg = logpdf(neg))
    sl(c) = tail_slope(c...; x0 = 0.5, x1 = 4.0)
    println("  CCDF slope α over [0.5,4] nA  — |ΔI|=$(round(sl(cabs);digits=2))  +ΔI=$(round(sl(cpos);digits=2))  -ΔI=$(round(sl(cneg);digits=2))")
    println("  Hill α (top 0.1%)            — |ΔI|=$(round(hill(absx);digits=2))  +ΔI=$(round(hill(pos);digits=2))  -ΔI=$(round(hill(neg);digits=2))")
    println("  Hill α (top 0.01%)           — |ΔI|=$(round(hill(absx;frac=1e-4);digits=2))  +ΔI=$(round(hill(pos;frac=1e-4);digits=2))  -ΔI=$(round(hill(neg;frac=1e-4);digits=2))\n")
end

# --- Figure: CCDF (top row) and log-binned pdf (bottom row); columns = |ΔI|, +ΔI, -ΔI ---
begin
    fig = Figure(size = (1150, 720))
    kinds = [(:abs, "|ΔI|"), (:pos, "+ΔI (positive)"), (:neg, "-ΔI (negative)")]
    colors = Dict(:current => :firebrick, :base => :steelblue)
    for (col, (k, ttl)) in enumerate(kinds)
        axc = Axis(fig[1, col]; xscale = log10, yscale = log10, title = "CCDF  $ttl",
            xlabel = "|step| (nA)", ylabel = col==1 ? "P(X ≥ x)" : "")
        axp = Axis(fig[2, col]; xscale = log10, yscale = log10, title = "log-binned pdf  $ttl",
            xlabel = "|step| (nA)", ylabel = col==1 ? "density" : "")
        for tag in (:base, :current)
            haskey(results, tag) || continue
            r = results[tag]
            cc = getfield(r, k); pp = getfield(r, Symbol(:p, k))
            isempty(cc[1]) || lines!(axc, cc[1], cc[2]; color = colors[tag], label = r.label)
            isempty(pp[1]) || scatterlines!(axp, pp[1], pp[2]; color = colors[tag], markersize = 5)
        end
        vlines!(axc, [4.0]; color = :gray, linestyle = :dash)  # histogram clip in the demo plot
        vlines!(axp, [4.0]; color = :gray, linestyle = :dash)
        col == 1 && axislegend(axc; position = :lb, framevisible = true)
    end
    Label(fig[0, :], "Input step-size ΔI tail: CCDF & log-binned pdf (dashed = 4 nA histogram clip)";
        fontsize = 16, font = :bold)
    save(joinpath(OUT, "step_size_tail_analysis.png"), fig)
    @info "saved" file=joinpath(OUT, "step_size_tail_analysis.png")
end
