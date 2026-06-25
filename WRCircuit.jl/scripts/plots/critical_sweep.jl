#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Critical sweep — exponent extraction.
#
# Load every (δ, Δg_K) simulation in data/critical_sweep, fit the per-neuron
# diffusion exponent (from the input MAD) and spectral exponent (from the input
# PSD), and save *all* per-neuron exponents (not neuron-averages) to
# data/plots/critical_sweep.jld2.
#
# The fitting recipe follows scripts/plots/_critical_sweep.jl: the diffusion
# exponent is the first component of a 2-component MAPPLE fit to the MAD curve;
# the spectral exponent is the last component of a 1-component fit to the
# 10-1000 Hz PSD.

using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
using LinearAlgebra
using Optim
using MoreMaps
using Statistics
using Distributed
using USydClusters
import ForwardDiff
WRCircuit.@preamble

# ──────────────────────────────────────────────────────────────────────────────
# Distributed workers — one grid cell (file load + neuron sweep) per Pmap task
# ──────────────────────────────────────────────────────────────────────────────

begin # * Add procs and load code everywhere
    USydClusters.Physics.addprocs(
        20; ncpus = 1, mem = "10GB", walltime = "23:00:00",
        queue = `taiji`
    )
    USydClusters.Physics.addprocs(10; ncpus = 1, mem = "10GB", walltime = "23:00:00")
    addprocs(8) # Local

    @everywhere begin
        using WRCircuit
        import WRCircuit
        using DrWatson
        using TimeseriesTools
        using Optim
        import ForwardDiff
    end
end

# Neurons to fit per cell, injected into each grid-cell fit. 1 = every neuron
# ("all exponents"); the prototype used 50 purely for speed. The per-neuron fits
# themselves (per_neuron / diffusion_exponents / spectral_exponents) live in
# WRCircuit, so workers pick them up via `@everywhere using WRCircuit`.
const neuron_step = 10

# ──────────────────────────────────────────────────────────────────────────────
# Assemble the full (δ, Δg_K) grid of files
# ──────────────────────────────────────────────────────────────────────────────

begin # * Load sweep parameters (mirrors _critical_sweep.jl, but the full grid)
    files = readdir(datadir("critical_sweep"), join = true)
    ps = map(files) do f
        fname = parse_savename(f; connector = string(connector))[2]
        if haskey(fname, "key") || !haskey(fname, "delta") || !haskey(fname, "Delta_g_K")
            return nothing
        else
            return fname
        end
    end
    keep = .!isnothing.(ps)
    files = files[keep]
    ps = ps[keep]

    deltas = [p["delta"] for p in ps]
    Delta_g_Ks = [p["Delta_g_K"] for p in ps]

    # Sorted lookups so the saved grid (and any heatmap) is monotonic in both axes
    udelta = sort(unique(deltas))
    ugk = sort(unique(Delta_g_Ks))

    parameter_grid = Iterators.product(
        Dim{:delta}(udelta),
        Dim{:Delta_g_K}(ugk)
    ) |> collect
    parameter_grid = map(parameter_grid) do (d, gk)
        idx = findfirst((deltas .== d) .& (Delta_g_Ks .== gk))
        isnothing(idx) ? nothing : files[idx]
    end
    @info "Found $(count(!isnothing, parameter_grid)) / $(length(parameter_grid)) grid cells"
end

# ──────────────────────────────────────────────────────────────────────────────
# Fit exponents across the grid (one pass; missing cells -> empty vectors)
# ──────────────────────────────────────────────────────────────────────────────

"Fit `exponents` to the array loaded under `key` for every file in `grid`, fitting every `step`-th neuron; missing cells -> empty vectors. Distributes one cell per worker via Pmap."
function fit_grid(grid, key, exponents; step)
    return map(Chart(Pmap(), LogLogger(10)), grid) do f
        isnothing(f) && return Float64[]
        try
            return exponents(load(f, key); step)
        catch err
            @warn "Failed grid-cell fit for $f" err
            return Float64[]
        end
    end
end

begin # * Fit exponents across the grid
    @info "Fitting diffusion exponents (MAPPLE to MAD) across the grid..."
    a = fit_grid(parameter_grid, "inputs/mad", diffusion_exponents; step = neuron_step)

    @info "Fitting spectral exponents (MAPPLE to PSD) across the grid..."
    b = fit_grid(parameter_grid, "inputs/psd", spectral_exponents; step = neuron_step)
end

# ──────────────────────────────────────────────────────────────────────────────
# Save (per-neuron exponents only, plus the parameter lookups)
# ──────────────────────────────────────────────────────────────────────────────

begin # * Save
    mkpath(datadir("plots"))
    out = Dict(
        "a" => a,            # per-neuron diffusion exponents, ToolsArray{Vector} over (δ, Δg_K)
        "b" => b,            # per-neuron spectral exponents
        "delta" => udelta,
        "Delta_g_K" => ugk
    )
    outfile = datadir("plots", "critical_sweep.jld2")
    wsave(outfile, out)
    @info "Saved $outfile"
end
