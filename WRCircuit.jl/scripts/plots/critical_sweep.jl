#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Critical sweep — exponent extraction.
#
# The sweep (scripts/calculations/critical_sweep.jl) is THREE intersecting 2-D
# planes through the default working-regime point, each run over 5 connectome
# seeds:
#   - dg plane: (delta, Delta_g_K) at sigma_ee = sigma_ee_0
#   - ds plane: (delta, sigma_ee)  at Delta_g_K = Delta_g_K_0
#   - gs plane: (Delta_g_K, sigma_ee) at delta = delta_0
# For each plane we fit the per-neuron diffusion exponent (from the input MAD) and
# spectral exponent (from the input PSD) and save *all* per-neuron exponents (not
# neuron-averages) to data/plots/critical_sweep.jld2, KEEPING THE SEED AXIS
# EXPLICIT: every saved grid is (axis1, axis2, seed) of per-neuron exponent
# vectors, so downstream scripts collapse the seed 'Obs' dimension as they wish.
#
# The fitting recipe follows scripts/plots/_critical_sweep.jl: the diffusion
# exponent is the first component of a 2-component MAPPLE fit to the MAD curve;
# the spectral exponent is the last component of a 1-component fit to the
# 10-1000 Hz PSD.

using DrWatson
DrWatson.@quickactivate :WRCircuit
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
try
    begin # * Add procs and load code everywhere
        USydClusters.Physics.distributeprocs(Inf; mem = "8GB", ncpus = 1)
        666
        @everywhere begin
            using WRCircuit
            @info "WRCircuit loaded on worker $(myid())"
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
    neuron_step = 10

    # ──────────────────────────────────────────────────────────────────────────────
    # Index every result file, then build one 3-D (axis1, axis2, seed) grid per plane
    # ──────────────────────────────────────────────────────────────────────────────

    begin # * Index files by (delta, Delta_g_K, sigma_ee, seed) and lay out the three planes
        files = readdir(datadir("critical_sweep"), join = true)
        entries = map(files) do f
            fname = parse_savename(f; connector = string(connector))[2]
            needed = ("delta", "Delta_g_K", "sigma_ee", "seed")
            if haskey(fname, "key") || !all(k -> haskey(fname, k), needed)
                return nothing
            end
            return (;
                file = f, delta = fname["delta"], Delta_g_K = fname["Delta_g_K"],
                sigma_ee = fname["sigma_ee"], seed = Int(fname["seed"]),
            )
        end
        entries = filter(!isnothing, entries)

        # Sorted lookups so each saved grid is monotonic along both swept axes.
        udelta = sort(unique(e.delta for e in entries))
        ugk = sort(unique(e.Delta_g_K for e in entries))
        usigma = sort(unique(e.sigma_ee for e in entries))
        useed = sort(unique(e.seed for e in entries))

        # Plane anchors: the default working-regime point, snapped onto the grids that are
        # actually present so each plane's fixed coordinate matches a parsed filename value.
        defaults = WRCircuit.defaults(WRCircuit.models.Spatial)
        _snap(v, grid) = grid[argmin(abs.(grid .- v))]
        delta_0 = _snap(round(Float64(defaults[:delta]); sigdigits = 3), udelta)
        Delta_g_K_0 = _snap(round(Float64(defaults[:Delta_g_K]); sigdigits = 3), ugk)
        sigma_ee_0 = _snap(round(Float64(defaults[:sigma_ee]); sigdigits = 3), usigma)

        # (delta, Delta_g_K, sigma_ee, seed) -> file lookup, then one 3-D file grid per plane:
        # the two swept axes x the explicit seed axis, at the plane's fixed third coordinate.
        lookup = Dict((e.delta, e.Delta_g_K, e.sigma_ee, e.seed) => e.file for e in entries)
        file_at(d, gk, s, seed) = get(lookup, (d, gk, s, seed), nothing)
        grid_dg = map(
            collect(
                Iterators.product(
                    Dim{:delta}(udelta), Dim{:Delta_g_K}(ugk), Dim{:seed}(useed)
                )
            )
        ) do (d, gk, seed)
            file_at(d, gk, sigma_ee_0, seed)
        end
        grid_ds = map(
            collect(
                Iterators.product(
                    Dim{:delta}(udelta), Dim{:sigma_ee}(usigma), Dim{:seed}(useed)
                )
            )
        ) do (d, s, seed)
            file_at(d, Delta_g_K_0, s, seed)
        end
        grid_gs = map(
            collect(
                Iterators.product(
                    Dim{:Delta_g_K}(ugk), Dim{:sigma_ee}(usigma), Dim{:seed}(useed)
                )
            )
        ) do (gk, s, seed)
            file_at(delta_0, gk, s, seed)
        end
        @info "Plane cells (incl. seed): " *
            "dg $(count(!isnothing, grid_dg))/$(length(grid_dg)), " *
            "ds $(count(!isnothing, grid_ds))/$(length(grid_ds)), " *
            "gs $(count(!isnothing, grid_gs))/$(length(grid_gs)); " *
            "anchors delta_0=$delta_0, Delta_g_K_0=$Delta_g_K_0, sigma_ee_0=$sigma_ee_0"
    end

    # ──────────────────────────────────────────────────────────────────────────────
    # Fit exponents across all three planes (one pass each; missing cells -> empty vectors)
    # ──────────────────────────────────────────────────────────────────────────────

    "Fit `exponents` to the array loaded under `key` for every file in `grid`, fitting every `step`-th neuron; missing cells -> empty vectors. Distributes one cell per worker via Pmap."
    function fit_grid(grid, key, exponents; step)
        return map(Chart(Pmap(), LogLogger(100)), grid) do f
            isnothing(f) && return Float64[]
            try
                return exponents(load(f, key); step)
            catch err
                @warn "Failed grid-cell fit for $f" err
                return Float64[]
            end
        end
    end

    begin # * Fit exponents across all three planes (per-neuron, per-seed)
        @info "Fitting diffusion exponents (MAPPLE to MAD)..."
        @info "a_dg"
        a_dg = fit_grid(grid_dg, "inputs/mad", diffusion_exponents; step = neuron_step)
        @info "a_ds"
        a_ds = fit_grid(grid_ds, "inputs/mad", diffusion_exponents; step = neuron_step)
        @info "a_gs"
        a_gs = fit_grid(grid_gs, "inputs/mad", diffusion_exponents; step = neuron_step)

        @info "Fitting spectral exponents (MAPPLE to PSD)..."
        @info "b_dg"
        b_dg = fit_grid(grid_dg, "inputs/psd", spectral_exponents; step = neuron_step)
        @info "b_ds"
        b_ds = fit_grid(grid_ds, "inputs/psd", spectral_exponents; step = neuron_step)
        @info "b_gs"
        b_gs = fit_grid(grid_gs, "inputs/psd", spectral_exponents; step = neuron_step)
    end

    # ──────────────────────────────────────────────────────────────────────────────
    # Save (per-neuron exponents only; seed kept as the trailing axis of every grid)
    # ──────────────────────────────────────────────────────────────────────────────

    begin # * Save
        mkpath(datadir("plots"))
        out = Dict(
            # Each grid is (axis1, axis2, seed) of per-neuron exponent vectors.
            "a_dg" => a_dg, "b_dg" => b_dg,   # dg plane: (delta, Delta_g_K) at sigma_ee_0
            "a_ds" => a_ds, "b_ds" => b_ds,   # ds plane: (delta, sigma_ee)  at Delta_g_K_0
            "a_gs" => a_gs, "b_gs" => b_gs,   # gs plane: (Delta_g_K, sigma_ee) at delta_0
            # Shared axis lookups + the plane anchors (the default working-regime point).
            "delta" => udelta, "Delta_g_K" => ugk, "sigma_ee" => usigma, "seed" => useed,
            "delta_0" => delta_0, "Delta_g_K_0" => Delta_g_K_0, "sigma_ee_0" => sigma_ee_0,
        )
        outfile = datadir("plots", "critical_sweep.jld2")
        wsave(outfile, out)
        @info "Saved $outfile"
    end
finally
    rmprocs()
end
