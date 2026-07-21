#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Circuit sweep — exponent extraction.
#
# The sweep (scripts/circuit_sweep.jl) writes one result file per (plane cell, seed), each named for exactly
# its two swept axes + seed. This script groups files back into FIVE 2-D planes through the working-regime
# point --- a file's non-seed key set IS its plane:
#   - dg  plane: (delta, Delta_g_K)   - ds plane: (delta, sigma_ee)   - gs plane: (Delta_g_K, sigma_ee)
#   - td  plane: (tau_r_e, tau_d_e)   - dtd plane: (delta, tau_d_e)
# For each plane we fit the per-neuron diffusion exponent (from the input MAD) and spectral exponent (from the
# input PSD) and save *all* per-neuron exponents (not neuron-averages) to data/circuit_exponents.jld2, KEEPING
# THE SEED AXIS EXPLICIT: every saved grid is (axis1, axis2, seed) of per-neuron exponent vectors, so
# downstream scripts collapse the seed 'Obs' dimension as they wish.

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
        if contains(gethostname(), "physics.usyd.edu.au")
            AcademicClusters.USydPhysics.distributeprocs(Inf; mem = "6GB", ncpus = 1)
            addprocs(16)
        elseif contains(gethostname(), "gadi") && haskey(ENV, "PBS_NCPUS")
            AcademicClusters.NCIGadi.distributeprocs() # Defaults to as many single-cpu workers as available
        end
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
    # already in circuit_sweep.jld2 is preserved
    PLANES = [:dtd] # [:dg, :ds, :gs, :td, :dtd]
    want(p) = p in PLANES

    begin # * Index sweep files by plane. Each file is named for exactly its two swept axes + seed, so a
        # file's non-seed key set IS its plane --- no negative filters, no shared 3-key pool.
        files = readdir(datadir("circuit_sweep"), join = true)
        parsed = filter(
            !isnothing, map(files) do f
                d = try
                    parse_savename(f; connector = string(connector))[2]
                catch
                    return nothing
                end
                (haskey(d, "key") || !haskey(d, "seed")) && return nothing
                return (; file = f, keyset = Set(keys(d)), d)
            end
        )

        # Entries (file, v1, v2, seed) for the plane whose two swept axes are (ax1, ax2): files whose varied
        # keys are EXACTLY {ax1, ax2, seed}.
        function plane_entries(ax1, ax2)
            wantkeys = Set((string(ax1), string(ax2), "seed"))   # NOT `want`: that name is the `want(p)` plane filter, which try-scope makes a shared local this nested fn would clobber
            return [
                (; file = p.file, v1 = p.d[string(ax1)], v2 = p.d[string(ax2)], seed = Int(p.d["seed"]))
                    for p in parsed if p.keyset == wantkeys
            ]
        end
        # 3-D file grid (ax1 × ax2 × seed) over the supplied axis lookups; `nothing` where a cell has no file.
        function plane_grid(entries, ax1, u1, ax2, u2, useed)
            lk = Dict((e.v1, e.v2, e.seed) => e.file for e in entries)
            return map(collect(Iterators.product(Dim{ax1}(u1), Dim{ax2}(u2), Dim{:seed}(useed)))) do (a, b, s)
                get(lk, (a, b, s), nothing)
            end
        end

        useed = sort(unique(Int(p.d["seed"]) for p in parsed))
        defaults = WRCircuit.defaults(WRCircuit.models.Spatial)
        _snap(v, grid) = isempty(grid) ? v : grid[argmin(abs.(grid .- v))]

        # δ/Δg_K/σ_ee main planes: three 2-axis families sharing the δ, Δg_K, σ_ee axes. Union each shared axis
        # so grid_dg/ds/gs sit on common (udelta, ugk, usigma) grids --- the layout circuit_exponents.jld2 expects.
        e_dg = plane_entries(:delta, :Delta_g_K)
        e_ds = plane_entries(:delta, :sigma_ee)
        e_gs = plane_entries(:Delta_g_K, :sigma_ee)
        udelta = sort(unique([e.v1 for e in vcat(e_dg, e_ds)]))
        ugk = sort(unique(vcat([e.v2 for e in e_dg], [e.v1 for e in e_gs])))
        usigma = sort(unique(vcat([e.v2 for e in e_ds], [e.v2 for e in e_gs])))
        delta_0 = _snap(round(Float64(defaults[:delta]); sigdigits = 3), udelta)
        Delta_g_K_0 = _snap(round(Float64(defaults[:Delta_g_K]); sigdigits = 3), ugk)
        sigma_ee_0 = _snap(round(Float64(defaults[:sigma_ee]); sigdigits = 3), usigma)
        grid_dg = plane_grid(e_dg, :delta, udelta, :Delta_g_K, ugk, useed)
        grid_ds = plane_grid(e_ds, :delta, udelta, :sigma_ee, usigma, useed)
        grid_gs = plane_grid(e_gs, :Delta_g_K, ugk, :sigma_ee, usigma, useed)

        # τ_syn plane (tau_r_e, tau_d_e) and δ/τ_d plane (delta, tau_d_e): each its own 2-axis family.
        e_td = plane_entries(:tau_r_e, :tau_d_e)
        utau_r = sort(unique(e.v1 for e in e_td))
        utau_d = sort(unique(e.v2 for e in e_td))
        tau_r_e_0 = _snap(round(Float64(defaults[:tau_r_e]); sigdigits = 3), utau_r)
        tau_d_e_0 = _snap(round(Float64(defaults[:tau_d_e]); sigdigits = 3), utau_d)
        grid_td = plane_grid(e_td, :tau_r_e, utau_r, :tau_d_e, utau_d, useed)

        e_dtd = plane_entries(:delta, :tau_d_e)
        udtd_delta = sort(unique(e.v1 for e in e_dtd))
        udtd_tau_d = sort(unique(e.v2 for e in e_dtd))
        grid_dtd = plane_grid(e_dtd, :delta, udtd_delta, :tau_d_e, udtd_tau_d, useed)
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

    begin # * Save --- MERGE into circuit_exponents.jld2: only the planes computed this run are (over)written;
        # every other plane already in the file is loaded and kept. Each grid stays (axis1, axis2, seed) of
        # per-neuron exponent vectors, alongside its own axis lookups (+ anchors for the main δ planes).
        mkpath(datadir("plots"))
        outfile = datadir("circuit_exponents.jld2")
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
