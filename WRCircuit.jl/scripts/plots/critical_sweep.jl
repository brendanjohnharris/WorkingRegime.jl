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
using AcademicClusters
import ForwardDiff
WRCircuit.@preamble

try
    begin # * Add procs and load code everywhere
        AcademicClusters.USydPhysics.distributeprocs(Inf; mem = "6GB", ncpus = 1)
        addprocs(16)

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

    # Reject a heatmap parameter point (one (axis1, axis2) cell, across seed) unless at least this many
    # of its seeds produced a sweep file; blanked points become empty vectors -> NaN in every heatmap.
    N_REQUIRED = 5

    # Which planes to (re)compute. ONLY these are fit (the multi-hour cost) and written; every other plane
    # already in critical_sweep.jld2 is preserved
    PLANES = [:dtd] # [:dg, :ds, :gs, :td, :dtd]
    want(p) = p in PLANES

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
        # τ_syn plane: a separate file set keyed by (tau_r_e, tau_d_e, seed) --- the main filter above drops
        # these (they carry no delta/Delta_g_K/sigma_ee keys), so index them here into a 4th grid over
        # (tau_r_e, tau_d_e, seed) at the same default working-regime point.
        tau_entries = filter(
            !isnothing, map(files) do f
                fname = parse_savename(f; connector = string(connector))[2]
                needed = ("tau_r_e", "tau_d_e", "seed")
                (haskey(fname, "key") || !all(k -> haskey(fname, k), needed)) && return nothing
                return (; file = f, tau_r_e = fname["tau_r_e"], tau_d_e = fname["tau_d_e"], seed = Int(fname["seed"]))
            end
        )
        utau_r = sort(unique(e.tau_r_e for e in tau_entries))
        utau_d = sort(unique(e.tau_d_e for e in tau_entries))
        tau_r_e_0 = _snap(round(Float64(defaults[:tau_r_e]); sigdigits = 3), utau_r)
        tau_d_e_0 = _snap(round(Float64(defaults[:tau_d_e]); sigdigits = 3), utau_d)
        tau_lookup = Dict((e.tau_r_e, e.tau_d_e, e.seed) => e.file for e in tau_entries)
        grid_td = map(
            collect(
                Iterators.product(
                    Dim{:tau_r_e}(utau_r), Dim{:tau_d_e}(utau_d), Dim{:seed}(useed)
                )
            )
        ) do (tr, td, seed)
            get(tau_lookup, (tr, td, seed), nothing)
        end
        # δ/τ_d plane: (delta, tau_d_e, seed) --- these files carry delta + tau_d_e but no
        # Delta_g_K/sigma_ee/tau_r_e, so both filters above drop them; index them here into a 5th grid.
        dtd_entries = filter(
            !isnothing, map(files) do f
                fname = parse_savename(f; connector = string(connector))[2]
                needed = ("delta", "tau_d_e", "seed")
                (haskey(fname, "key") || !all(k -> haskey(fname, k), needed) || haskey(fname, "Delta_g_K")) && return nothing
                return (; file = f, delta = fname["delta"], tau_d_e = fname["tau_d_e"], seed = Int(fname["seed"]))
            end
        )
        udtd_delta = sort(unique(e.delta for e in dtd_entries))
        udtd_tau_d = sort(unique(e.tau_d_e for e in dtd_entries))
        dtd_lookup = Dict((e.delta, e.tau_d_e, e.seed) => e.file for e in dtd_entries)
        grid_dtd = map(
            collect(
                Iterators.product(
                    Dim{:delta}(udtd_delta), Dim{:tau_d_e}(udtd_tau_d), Dim{:seed}(useed)
                )
            )
        ) do (d, td, seed)
            get(dtd_lookup, (d, td, seed), nothing)
        end
        # Blank parameter points with too few sweep files: for each (axis1, axis2), if fewer than
        # N_REQUIRED seeds have a file, drop the whole seed slice so the point is missing everywhere.
        function require_samples!(grid, n_required)
            a1, a2, _ = size(grid)
            for i in 1:a1, j in 1:a2
                count(!isnothing, @view grid[i, j, :]) < n_required && (grid[i, j, :] .= nothing)
            end
            return grid
        end
        require_samples!(grid_dg, N_REQUIRED)
        require_samples!(grid_ds, N_REQUIRED)
        require_samples!(grid_gs, N_REQUIRED)
        require_samples!(grid_td, N_REQUIRED)
        require_samples!(grid_dtd, N_REQUIRED)

        @info "Plane cells (incl. seed): " *
            "dg $(count(!isnothing, grid_dg))/$(length(grid_dg)), " *
            "ds $(count(!isnothing, grid_ds))/$(length(grid_ds)), " *
            "gs $(count(!isnothing, grid_gs))/$(length(grid_gs)), " *
            "td $(count(!isnothing, grid_td))/$(length(grid_td)), " *
            "dtd $(count(!isnothing, grid_dtd))/$(length(grid_dtd)); " *
            "anchors delta_0=$delta_0, Delta_g_K_0=$Delta_g_K_0, sigma_ee_0=$sigma_ee_0"
    end

    # ──────────────────────────────────────────────────────────────────────────────
    # Fit exponents across all three planes (one pass each; missing cells -> empty vectors)
    # ──────────────────────────────────────────────────────────────────────────────

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

    begin # * Fit exponents for the SELECTED planes only (per-neuron, per-seed). `a` = diffusion (MAPPLE to
        # MAD), `b` = spectral (MAPPLE to PSD). Each `fit_grid` is the expensive step, so unselected planes
        # are skipped and their existing exponents are kept by the merge on save.
        if want(:dg)
            @info "fitting a_dg/b_dg";   a_dg = fit_grid(grid_dg, "inputs/mad", diffusion_exponents; step = neuron_step); b_dg = fit_grid(grid_dg, "inputs/psd", spectral_exponents; step = neuron_step)
        end
        if want(:ds)
            @info "fitting a_ds/b_ds";   a_ds = fit_grid(grid_ds, "inputs/mad", diffusion_exponents; step = neuron_step); b_ds = fit_grid(grid_ds, "inputs/psd", spectral_exponents; step = neuron_step)
        end
        if want(:gs)
            @info "fitting a_gs/b_gs";   a_gs = fit_grid(grid_gs, "inputs/mad", diffusion_exponents; step = neuron_step); b_gs = fit_grid(grid_gs, "inputs/psd", spectral_exponents; step = neuron_step)
        end
        if want(:td)
            @info "fitting a_td/b_td";   a_td = fit_grid(grid_td, "inputs/mad", diffusion_exponents; step = neuron_step); b_td = fit_grid(grid_td, "inputs/psd", spectral_exponents; step = neuron_step)
        end
        if want(:dtd)
            @info "fitting a_dtd/b_dtd"; a_dtd = fit_grid(grid_dtd, "inputs/mad", diffusion_exponents; step = neuron_step); b_dtd = fit_grid(grid_dtd, "inputs/psd", spectral_exponents; step = neuron_step)
        end
    end

    # ──────────────────────────────────────────────────────────────────────────────
    # Save (per-neuron exponents only; seed kept as the trailing axis of every grid)
    # ──────────────────────────────────────────────────────────────────────────────

    begin # * Save --- MERGE into critical_sweep.jld2: only the planes computed this run are (over)written;
        # every other plane already in the file is loaded and kept. Each grid stays (axis1, axis2, seed) of
        # per-neuron exponent vectors, alongside its own axis lookups (+ anchors for the main δ planes).
        mkpath(datadir("plots"))
        outfile = datadir("plots", "critical_sweep.jld2")
        merged = isfile(outfile) ? load(outfile) : Dict{String, Any}()
        if want(:dg)
            merged["a_dg"] = a_dg; merged["b_dg"] = b_dg
        end
        if want(:ds)
            merged["a_ds"] = a_ds; merged["b_ds"] = b_ds
        end
        if want(:gs)
            merged["a_gs"] = a_gs; merged["b_gs"] = b_gs
        end
        if want(:td)
            merged["a_td"] = a_td; merged["b_td"] = b_td
        end
        if want(:dtd)
            merged["a_dtd"] = a_dtd; merged["b_dtd"] = b_dtd
        end
        # Refresh only the axis lookups / anchors tied to the planes just computed (each plane owns its axes).
        if want(:dg) || want(:ds) || want(:gs)
            merged["delta"] = udelta; merged["Delta_g_K"] = ugk; merged["sigma_ee"] = usigma; merged["seed"] = useed
            merged["delta_0"] = delta_0; merged["Delta_g_K_0"] = Delta_g_K_0; merged["sigma_ee_0"] = sigma_ee_0
        end
        if want(:td)
            merged["tau_r_e"] = utau_r; merged["tau_d_e"] = utau_d; merged["tau_r_e_0"] = tau_r_e_0; merged["tau_d_e_0"] = tau_d_e_0
        end
        if want(:dtd)
            merged["dtd_delta"] = udtd_delta; merged["dtd_tau_d_e"] = udtd_tau_d
        end
        wsave(outfile, merged)
        @info "Saved $outfile (planes computed: $PLANES)"
    end
finally
    rmprocs()
end
