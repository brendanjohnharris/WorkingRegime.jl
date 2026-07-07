#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Hierarchical variation figure.
#
# Layout (4 rows × 2 cols):
#   Row 1    — the (a, b) plane at L2/3 (left) and L6 (right). Each carries the
#              neuropixels scatter coloured by anatomical hierarchy score, plus
#              five mean-direction arrows anchored at the scatter centroid that
#              show the local Jacobian directions of two forward maps at their
#              canonical operating points:
#                · circuit (δ, Δg_K, σ_ee) → (a, b) [green/purple/orange], from critical_sweep.jld2;
#                · bFNS theory (α, β) → (a, b)       [red/blue], from the flat bFNS sweep.
#   Rows 2-4 — heatmaps for the three circuit forward-map planes, a (left) and b
#              (right): (δ, Δg_K), (δ, σ_ee) and (Δg_K, σ_ee). Each cell is a
#              per-neuron-exponent median pooled across the connectome seeds.
#
# Sources: WRExperiment.jl/data/plots/WRExperiment.jld2 (scatter; produced by
# WRExperiment.jl/scripts/collect_calculations.jl),
# WRCircuit.jl/data/plots/critical_sweep.jld2 (the three circuit planes + δ/Δg_K/σ_ee
# arrows), and WRTheory.jl/data/bFNS_sweep/flat_γ=0.03_η=0.01.jld2 (α/β arrows).

using DrWatson
@quickactivate "WorkingRegime"
using JLD2
using CairoMakie
using Fathom
using Statistics
using Random

set_theme!(fathom())

# ──────────────────────────────────────────────────────────────────────────────
# JLD2 typemap: SpatiotemporalMotifs' dim types (SessionID, Structure, …) are
# not in WorkingRegime's env, so on load we upgrade every saved ToolsArray to a
# lightweight NamedArray that keeps numeric data + (dim-name => lookup) pairs.
# The upgrade is recursive, so nested ToolsArrays (e.g. coeffs_median) are
# upgraded too. Pattern copied from scripts/plots/combined_curves.jl.
# ──────────────────────────────────────────────────────────────────────────────

struct NamedArray{T, N}
    data::Array{T, N}
    dims::Vector{Pair{Symbol, Vector}}
end
Base.collect(x::NamedArray) = x.data

function _dim_name(d)
    tname = string(typeof(d).parameters[1])
    head = first(split(tname, ('{', ',', ' ')))
    if head == "Dim"
        inner = split(tname, '{'; limit = 2)[2]
        return Symbol(first(split(inner, (',', '}'))))
    end
    return Symbol(head)
end
_dim_lookup(d) = collect(d.val.data)
function _try_dim_pair(d)
    try
        return _dim_name(d) => _dim_lookup(d)
    catch
        return nothing
    end
end
function JLD2.rconvert(::Type{<:NamedArray}, x)
    outer = Pair{Symbol, Vector}[]
    for d in x.dims
        p = _try_dim_pair(d)
        p === nothing || push!(outer, p)
    end
    return NamedArray(collect(x.data), outer)
end
const _toolsarray_typemap = Dict(
    # WRExperiment.jld2 was saved when ToolsArray lived under TimeseriesTools;
    # critical_sweep.jld2 was saved after the refactor, so its inner cells use
    # the new TimeseriesBase.ToolsArrays path. Its outer grid is a plain
    # DimensionalData.DimArray (built from Iterators.product). Upgrade all three
    # to the same NamedArray shim.
    "TimeseriesTools.ToolsArray" => JLD2.Upgrade(NamedArray),
    "TimeseriesBase.ToolsArrays.ToolsArray" => JLD2.Upgrade(NamedArray),
    "DimensionalData.DimArray" => JLD2.Upgrade(NamedArray)
)

"Index a NamedArray by named values along one or more dims (returns the slice)."
function _select(x::NamedArray, selectors::Pair...)
    idxs = Any[Colon() for _ in x.dims]
    for (name, val) in selectors
        i = findfirst(d -> d.first == Symbol(name), x.dims)
        i === nothing && error("Dim $name not in $(first.(x.dims))")
        j = findfirst(==(val), x.dims[i].second)
        j === nothing && error("Value $val not in $(x.dims[i].first) lookup")
        idxs[i] = j
    end
    return x.data[idxs...]
end

"Pick one element of a NamedArray whose elements are themselves NamedArrays."
function _select_outer(x::NamedArray, dimname::Symbol, val)
    i = findfirst(d -> d.first == dimname, x.dims)
    i === nothing && error("Dim $dimname not in $(first.(x.dims))")
    j = findfirst(==(val), x.dims[i].second)
    j === nothing && error("Value $val not in $(x.dims[i].first) lookup")
    return x.data[j]
end

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap median + 95% CI across sessions
# ──────────────────────────────────────────────────────────────────────────────

function bootstrapmedian(x; N = 10_000, α = 0.05)
    x = collect(skipmissing(x))
    x = filter(!isnan, x)
    isempty(x) && return (NaN, (NaN, NaN))
    rng = Random.MersenneTwister(42)
    n = length(x)
    meds = [median(x[rand(rng, 1:n, n)]) for _ in 1:N]
    return median(x), Tuple(quantile(meds, (α / 2, 1 - α / 2)))
end

# ──────────────────────────────────────────────────────────────────────────────
# Constants — Siegle 2021 anatomical hierarchy scores (mouse visual cortex)
# (transcribed from SpatiotemporalMotifs/src/Plots.jl)
# ──────────────────────────────────────────────────────────────────────────────

const structures = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
const hierarchy_scores = Dict(
    "VISp" => -0.357, "VISl" => -0.093, "VISrl" => -0.059,
    "VISal" => 0.152, "VISpm" => 0.327, "VISam" => 0.441
)
const stim = "spontaneous"

# Layer integer codes in the saved data: 2 = L2/3, 3 = L4, 4 = L5, 5 = L6.
const layer_names = Dict(2 => "L2/3", 3 => "L4", 4 => "L5", 5 => "L6")

const DELTA_MIN = 4
const DGK_MIN = 0.001
const SIGMA_MIN = 0.06
const SIGMA_MAX = 0.075

const outdir = plotsdir("hierarchical_variation")
mkpath(outdir)

# ──────────────────────────────────────────────────────────────────────────────
# Load aggregated experiment data
# ──────────────────────────────────────────────────────────────────────────────

const inpath = projectdir("WRExperiment.jl", "data", "plots", "WRExperiment.jld2")
plot_data = jldopen(
    f -> Dict(k => f[k] for k in keys(f)), inpath;
    typemap = _toolsarray_typemap
)

# Circuit per-neuron exponent grids — three planes through the working-regime
# point, each saved as (axis₁, axis₂, seed) of per-neuron exponent vectors. We
# pool seed + neuron to a per-cell median for the heatmaps and the arrows.
const circuit_path = projectdir("WRCircuit.jl", "data", "plots", "critical_sweep.jld2")
circuit = jldopen(
    f -> Dict(k => f[k] for k in keys(f)), circuit_path;
    typemap = _toolsarray_typemap
)

"NaN-aware median of one cell's per-neuron exponents POOLED across all seeds (the
trailing grid axis). Each (axis₁, axis₂, seed) cell may be a plain `Vector{Float64}`
(the empty-cell sentinel `Float64[]`) or a `NamedArray` (the typemap-upgraded
per-neuron `ToolsArray`); `collect` normalises both. Empty cells stay NaN."
function seed_pooled_median(cells)
    n1, n2 = size(cells, 1), size(cells, 2)
    out = Matrix{Float64}(undef, n1, n2)
    for i in 1:n1, j in 1:n2
        vals = Float64[]
        for k in axes(cells, 3)
            append!(vals, filter(!isnan, collect(cells[i, j, k])))
        end
        out[i, j] = isempty(vals) ? NaN : median(vals)
    end
    return out
end
# Wrap the immediate-use map calls in `invokelatest` to satisfy Julia 1.12's
# stricter world-age rules for global bindings defined at top-level.
const _δ_lookup_full = circuit["delta"]
const _gk_lookup_full = circuit["Delta_g_K"]
const _σ_lookup_full = circuit["sigma_ee"]
# Full-resolution per-cell median grids, one per plane (rows × cols):
#   dg: (δ × Δg_K), ds: (δ × σ_ee), gs: (Δg_K × σ_ee).
const _A_dg_full = Base.invokelatest(seed_pooled_median, circuit["a_dg"].data)
const _B_dg_full = Base.invokelatest(seed_pooled_median, circuit["b_dg"].data)
const _A_ds_full = Base.invokelatest(seed_pooled_median, circuit["a_ds"].data)
const _B_ds_full = Base.invokelatest(seed_pooled_median, circuit["b_ds"].data)
const _A_gs_full = Base.invokelatest(seed_pooled_median, circuit["a_gs"].data)
const _B_gs_full = Base.invokelatest(seed_pooled_median, circuit["b_gs"].data)
# td plane (τ_r_e × τ_d_e): the excitatory-synapse time-constant plane from the batched sweep. Plotted at
# full 8×8 resolution --- no ROI slice or coarsening (it is already small, and the full plane shows how a
# rises with τ_r and the fit blows up in the fast-rise/slow-decay corner).
const _A_td_full = Base.invokelatest(seed_pooled_median, circuit["a_td"].data)
const _B_td_full = Base.invokelatest(seed_pooled_median, circuit["b_td"].data)
const τr_lookup = collect(circuit["tau_r_e"])
const τd_lookup = collect(circuit["tau_d_e"])

# Slice each axis to its region of interest: δ > DELTA_MIN, Δg_K > DGK_MIN and
# σ_ee ∈ [SIGMA_MIN, SIGMA_MAX]. Each plane uses the slices of its own two axes.
const δ_keep = findall(>(DELTA_MIN), _δ_lookup_full)
const gk_keep = findall(>(DGK_MIN), _gk_lookup_full)
const σ_keep = findall(v -> SIGMA_MIN <= v <= SIGMA_MAX, _σ_lookup_full)
const δ_lookup = _δ_lookup_full[δ_keep]
const gk_lookup = _gk_lookup_full[gk_keep]
const σ_lookup = _σ_lookup_full[σ_keep]
const A_grid = _A_dg_full[δ_keep, gk_keep]   # dg plane (δ × Δg_K)
const B_grid = _B_dg_full[δ_keep, gk_keep]
const A_ds = _A_ds_full[δ_keep, σ_keep]      # ds plane (δ × σ_ee)
const B_ds = _B_ds_full[δ_keep, σ_keep]
const A_gs = _A_gs_full[gk_keep, σ_keep]     # gs plane (Δg_K × σ_ee)
const B_gs = _B_gs_full[gk_keep, σ_keep]

# Coarse-grain for display: average each COARSEN×COARSEN block of cells into one
# pixel (NaN-aware). Only the heatmaps are coarsened; the mean-direction arrows
# use the full-resolution grids.
const COARSEN = 2

"Average `M` into non-overlapping `b×b` blocks; partial edge blocks average over
whatever they contain. NaN cells are ignored; an all-NaN block stays NaN."
function block_average(M::AbstractMatrix, b)
    R, C = size(M)
    out = Matrix{Float64}(undef, cld(R, b), cld(C, b))
    for i in axes(out, 1), j in axes(out, 2)
        rows = ((i - 1) * b + 1):min(i * b, R)
        cols = ((j - 1) * b + 1):min(j * b, C)
        block = filter(!isnan, vec(M[rows, cols]))
        out[i, j] = isempty(block) ? NaN : mean(block)
    end

    return out
end

"Average a lookup vector into non-overlapping blocks of size `b` (block centres)."
function block_average(v::AbstractVector, b)
    n = length(v)
    return [mean(v[((i - 1) * b + 1):min(i * b, n)]) for i in 1:cld(n, b)]
end

# Coarsened axes (shared where planes share an axis) and per-plane grids.
const δ_coarse = Base.invokelatest(block_average, δ_lookup, COARSEN)
const gk_coarse = Base.invokelatest(block_average, gk_lookup, COARSEN)
const σ_coarse = Base.invokelatest(block_average, σ_lookup, COARSEN)
const A_coarse = Base.invokelatest(block_average, A_grid, COARSEN)
const B_coarse = Base.invokelatest(block_average, B_grid, COARSEN)
const A_ds_coarse = Base.invokelatest(block_average, A_ds, COARSEN)
const B_ds_coarse = Base.invokelatest(block_average, B_ds, COARSEN)
const A_gs_coarse = Base.invokelatest(block_average, A_gs, COARSEN)
const B_gs_coarse = Base.invokelatest(block_average, B_gs, COARSEN)

# Mean direction vectors of the circuit forward map in (a, b). Rather than
# drawing full isolines, we summarise each knob's effect as a single net
# displacement F(param_max) − F(param_min), averaged over the other parameter
# to marginalise out the operating point.
#
# The circuit arrows read the τ_syn (τ_r_e, τ_d_e) plane --- the excitatory synaptic time constants, the
# knobs that actually move the exponents (δ / Δg_K / σ_ee barely do; those planes are in the supplement).
#   τ_r arrow: τ_r_e swept over the clean low-rise band, averaged over τ_d_e   (td plane)
#   τ_d arrow: τ_d_e swept over its full range, averaged over the low-τ_r band (td plane)
# The fast-rise/slow-decay corner drives the 2-component MAD fit's first component past 1 (an artefact, not a
# regime), so mask a > 1 to NaN (mean_direction skips NaN corners) and keep the τ_r sweep in the clean band.
const _td_bad = _A_td_full .> 1
const _A_td_arrow = ifelse.(_td_bad, NaN, _A_td_full)
const _B_td_arrow = ifelse.(_td_bad, NaN, _B_td_full)
const τr_arrow_range = (minimum(τr_lookup), 1.2)   # low-rise band, near the neuropixels operating point (low a)
const τd_arrow_range = (minimum(τd_lookup), 5.5)    # low-decay band --- the scatter sits at low a ⇒ short τ_d

"Nearest grid index to a target value in a lookup vector."
_nearest(lookup, v) = argmin(abs.(lookup .- v))

"""
    mean_direction(grid_a, grid_b, sweep_lookup, sweep_range, other_lookup, other_range; dim)

Net (Δa, Δb) displacement as the swept parameter goes from `sweep_range[1]` to
`sweep_range[2]`, averaged over the `other` parameter restricted to `other_range`.
`dim = 1` sweeps rows (δ), `dim = 2` sweeps columns (Δg_K).
"""
function mean_direction(
        grid_a, grid_b, sweep_lookup, sweep_range,
        other_lookup, other_range; dim
    )
    s_lo = _nearest(sweep_lookup, sweep_range[1])
    s_hi = _nearest(sweep_lookup, sweep_range[2])
    other_keep = findall(v -> other_range[1] <= v <= other_range[2], other_lookup)
    da = Float64[]
    db = Float64[]
    for k in other_keep
        a_lo, a_hi, b_lo, b_hi = if dim == 1
            grid_a[s_lo, k], grid_a[s_hi, k], grid_b[s_lo, k], grid_b[s_hi, k]
        else
            grid_a[k, s_lo], grid_a[k, s_hi], grid_b[k, s_lo], grid_b[k, s_hi]
        end
        if all(!isnan, (a_lo, a_hi, b_lo, b_hi))
            push!(da, a_hi - a_lo)
            push!(db, b_hi - b_lo)
        end
    end
    return (mean(da), mean(db))
end

const τr_dir = mean_direction(
    _A_td_arrow, _B_td_arrow, τr_lookup, τr_arrow_range,
    τd_lookup, τd_arrow_range; dim = 1
)
const τd_dir = mean_direction(
    _A_td_arrow, _B_td_arrow, τd_lookup, τd_arrow_range,
    τr_lookup, τr_arrow_range; dim = 2
)

# ──────────────────────────────────────────────────────────────────────────────
# bFNS theory sweep — (α, β) → (a, b) direction arrows
#
# The flat (unconfined) sweep is the same grid behind the theory figure's
# (α, β) → (a, b) heatmaps: WRTheory.jl/data/bFNS_sweep/flat_γ=0.03_η=0.01.jld2
# holds `diffusion_exponent` and `spectral_exponent` as ToolsArrays over
# (α, β, γ, η, Obs). We NaN-aware average over the Obs seeds (and the singleton
# γ, η axes) to get 2-D (α, β) exponent grids, then reuse `mean_direction` to read
# the local Jacobian directions at the canonical operating point (α = 1.5, β = 0.85).
# ──────────────────────────────────────────────────────────────────────────────

const bfns_path = projectdir(
    "WRTheory.jl", "data", "bFNS_sweep", "flat_γ=0.03_η=0.01.jld2"
)
bfns = jldopen(
    f -> Dict(k => f[k] for k in keys(f)), bfns_path;
    typemap = _toolsarray_typemap
)

"NaN-aware mean over a collection; empty / all-NaN → NaN."
function _nanmean(x)
    v = filter(!isnan, vec(collect(x)))
    return isempty(v) ? NaN : mean(v)
end

"Collapse a (α, β, γ, η, Obs) sweep NamedArray to a 2-D (α, β) grid by NaN-aware
averaging over every axis after the first two (γ, η singletons + Obs seeds)."
function _ab_grid(na::NamedArray)
    da = na.data
    colons = ntuple(_ -> Colon(), ndims(da) - 2)
    return [_nanmean(@view da[i, j, colons...]) for i in axes(da, 1), j in axes(da, 2)]
end

"Lookup vector for dim `name`; falls back to `default` if the typemap dropped the
dim name (e.g. a custom Obs dim whose name fails to parse)."
function _dimlookup(na::NamedArray, name::Symbol, default)
    i = findfirst(d -> d.first == name, na.dims)
    return collect(i === nothing ? default : na.dims[i].second)
end

const _α_lookup = _dimlookup(bfns["diffusion_exponent"], :α, range(1.2, 2.0, length = 32))
const _β_lookup = _dimlookup(bfns["diffusion_exponent"], :β, range(0.2, 1.0, length = 32))
const A_ab = Base.invokelatest(_ab_grid, bfns["diffusion_exponent"])
const B_ab = Base.invokelatest(_ab_grid, bfns["spectral_exponent"])

# Local Jacobian directions at the canonical operating point (α = 1.5, β = 0.85):
#   α arrow: α swept 1.35 → 1.65, averaged over β ∈ [0.75, 0.95]
#   β arrow: β swept 0.70 → 1.00, averaged over α ∈ [1.35, 1.65]
const α_arrow_range = (1.35, 1.65)
const β_arrow_range = (0.7, 1.0)
const α_dir = mean_direction(
    A_ab, B_ab, _α_lookup, α_arrow_range,
    _β_lookup, (0.75, 0.95); dim = 1
)
const β_dir = mean_direction(
    A_ab, B_ab, _β_lookup, β_arrow_range,
    _α_lookup, (1.35, 1.65); dim = 2
)

# ──────────────────────────────────────────────────────────────────────────────
# Per-region (a, b) at a chosen cortical layer — bootstrap median over sessions
# ──────────────────────────────────────────────────────────────────────────────

# coeffs_median: NamedArray{Structure} of NamedArray{layer, SessionID}
function region_a(structure, layer_idx)
    cm = plot_data["madev_data"][stim]["coeffs_median"]
    inner = _select_outer(cm, :Structure, structure)
    return collect(_select(inner, :layer => layer_idx))
end
# spectral_exponents: NamedArray{Structure, layer, SessionID}
region_b(s, layer_idx) = collect(
    _select(
        plot_data["spectral_exponents"][stim],
        :Structure => s, :layer => layer_idx
    )
)

"Per-region (a, b) bootstrap medians + 95% CI at one layer, sorted low → high hierarchy."
function compute_points(layer_idx)
    pts = filter(
        !isnothing, map(structures) do s
            try
                am, (alo, ahi) = bootstrapmedian(region_a(s, layer_idx))
                bm, (blo, bhi) = bootstrapmedian(region_b(s, layer_idx))
                (
                    structure = s, h = hierarchy_scores[s],
                    a = am, a_lo = alo, a_hi = ahi,
                    b = bm, b_lo = blo, b_hi = bhi,
                )
            catch err
                @warn "Skipping $s at layer $layer_idx" exception = err
                nothing
            end
        end
    )
    sort!(pts; by = p -> p.h)
    return pts
end

const points_l23 = compute_points(2)   # L2/3
const points_l6 = compute_points(5)    # L6
const hrange = extrema(values(hierarchy_scores))

# ──────────────────────────────────────────────────────────────────────────────
# Hero-panel drawer. The circuit sweep curves and operating-point marker are
# layer-independent (the circuit doesn't have cortical layers), so they appear
# identically in both bottom panels; only the data scatter + trajectory change.
# ──────────────────────────────────────────────────────────────────────────────

"Place `label` just past an arrow tip (`anchor + vec`), nudged ~14 px further
along the arrow in pixel space and coloured to match."
function label_tip!(ax, anchor, vec, label, color)
    tip = anchor .+ vec
    n = hypot(vec...)
    off = n == 0 ? (0.0, 0.0) : (vec ./ n) .* 14
    return text!(
        ax, tip[1], tip[2]; text = label, color,
        align = (:center, :center), offset = Vec2f(off...), fontsize = 14
    )
end

function plot_hero!(
        ax, points; arrow_offset = (0.0, 0.0),
        label_align = (:left, :bottom), label_offset = (6, 6)
    )
    xs = [p.a for p in points]
    ys = [p.b for p in points]

    vlines!(ax, [0.5]; color = :gray, linestyle = :dash, linewidth = 1)

    # Mean-direction arrows, all anchored at the centroid of the experimental
    # scatter (plus an optional per-panel `arrow_offset` in data units to keep the
    # glyphs clear of the scatter). Three read the circuit forward map (δ, Δg_K,
    # σ_ee), two read the bFNS theory map (α, β). The raw displacement vectors span
    # very different magnitudes, so we rescale each to a fixed on-panel length — these
    # glyphs convey *direction*, not magnitude. Length is set to a fraction of the
    # data's diagonal spread so it adapts to whatever the autoscaled axis ends up
    # being.
    cx, cy = mean(xs) + arrow_offset[1], mean(ys) + arrow_offset[2]
    span = hypot(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys))
    arrow_len = 0.6 * span
    scaled(v) = (n = hypot(v...); n == 0 ? v : (v .* (arrow_len / n)))

    # Two arrow fans, each at its own anchor so they don't overlap: the circuit τ_syn pair (τ_r, τ_d) sits
    # left of the centroid, the bFNS theory (α, β) pair to the right. The split is a fraction of the data span.
    # (vector, label, colour): synaptic time constants in green/orange, bFNS orders in red/blue.
    split = 0.3 * span
    pairs = (
        (
            (cx - split, cy), (
                (scaled(τr_dir), "τ_r", qinghai),
                (scaled(τd_dir), "τ_d", seohae),
            ),
        ),
        (
            (cx + split, cy), (
                (scaled(α_dir), "α", bermejo),
                (scaled(β_dir), "β", baikal),
            ),
        ),
    )
    for (anchor, group) in pairs
        for (v, label, color) in group
            arrows2d!(
                ax, [Point2f(anchor...)], [Vec2f(v...)];
                color, tipwidth = 12, tiplength = 12, shaftwidth = 2.5
            )
            # Label each arrow directly at its tip (no legend), nudged a few pixels
            # further along the arrow and coloured to match.
            label_tip!(ax, anchor, v, label, color)
        end
    end


    # Session-bootstrap 95 % CI as error bars
    errorbars!(
        ax, xs, ys, xs .- [p.a_lo for p in points],
        [p.a_hi for p in points] .- xs;
        direction = :x, color = :gray70, whiskerwidth = 6
    )
    errorbars!(
        ax, xs, ys, ys .- [p.b_lo for p in points],
        [p.b_hi for p in points] .- ys;
        direction = :y, color = :gray70, whiskerwidth = 6
    )

    scatter!(
        ax, xs, ys; color = [p.h for p in points], colormap = binarysunset,
        colorrange = hrange, markersize = 18,
        strokecolor = :black, strokewidth = 0.8
    )

    for p in points
        text!(
            ax, p.a, p.b; text = p.structure,
            align = label_align, offset = label_offset, fontsize = 12
        )
    end
    return ax
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure: (a, b) scatters on top, then one heatmap row per circuit plane below
# ──────────────────────────────────────────────────────────────────────────────

f = FourPanel()   # main figure: (a, b) scatters (top) + the τ_syn heatmaps (bottom), 2×2 at 360×270 panel scale

# One sub-grid per panel; each holds [axis | colorbar] in its own columns. Row-major:
# gs[1,2] = heroes (L2/3, L6); gs[3,4] = td (a, b). The δ/Δg_K/σ_ee planes move to the supplementary figure below.
gs = subdivide(f, 2, 2)

begin # * Top row — (a, b) plane at L2/3 (left) and L6 (right)
    ax_l23 = Axis(
        gs[1][1, 1]; xlabel = "Diffusion exponent  a",
        ylabel = "Spectral exponent  b",
        title = "$stim, $(layer_names[2])"
    )
    plot_hero!(ax_l23, points_l23; arrow_offset = (0.0, -0.05))

    ax_l6 = Axis(
        gs[2][1, 1]; xlabel = "Diffusion exponent  a",
        ylabel = "Spectral exponent  b",
        title = "$stim, $(layer_names[5])"
    )
    plot_hero!(
        ax_l6, points_l6; arrow_offset = (-0.05, 0.0),
        label_align = (:right, :center), label_offset = (-10, 0)
    )

    # Share axis limits so the L2/3 ↔ L6 comparison is visually fair.
    linkaxes!(ax_l23, ax_l6)
end

"Draw one circuit forward-map heatmap (axis + colorbar) into sub-grid `pos`."
function circuit_heatmap!(pos, x, y, z; xlabel, ylabel, title, clabel, colorrange = automatic, highclip = automatic)
    ax = Axis(pos[1, 1]; xlabel = xlabel, ylabel = ylabel, title = title)
    p = heatmap!(ax, x, y, z; colormap = binarysunset, colorrange, highclip)
    Colorbar(pos[1, 2], p; label = clabel, width = 12)
    return ax
end

begin # * τ_syn plane heatmaps — the (τ_r_e × τ_d_e) synaptic-filter plane, a (left) and b (right)
    τrlab = "τ_r_e  (E rise, ms)"
    τdlab = "τ_d_e  (E decay, ms)"
    circuit_heatmap!(
        gs[3], τr_lookup, τd_lookup, _A_td_full;
        xlabel = τrlab, ylabel = τdlab, title = "Circuit:  a", clabel = "a",
        colorrange = (minimum(filter(isfinite, _A_td_full)), 1.0), highclip = binarysunset[end]
    )
    circuit_heatmap!(
        gs[4], τr_lookup, τd_lookup, _B_td_full;
        xlabel = τrlab, ylabel = τdlab, title = "Circuit:  b", clabel = "b"
    )
end

# A matching hierarchy colorbar on each top panel keeps the two axes the
# same pixel width — same data scale + same box geometry → genuinely fair
# visual comparison between L2/3 and L6.
for j in (1, 2)
    Colorbar(
        gs[j][1, 2]; colormap = binarysunset, limits = hrange,
        label = "Hierarchy score", width = 12
    )
end

addlabels!(f)
display(f)
outfile = joinpath(outdir, "hierarchical_variation.pdf")
wsave(outfile, f)
@info "Saved $outfile"

# ──────────────────────────────────────────────────────────────────────────────
# Supplementary figure: the three operating-point planes (δ, Δg_K, σ_ee) --- the knobs that barely move the
# exponents (contrast the τ_syn plane in the main figure). One plane per row, a (left) and b (right).
# ──────────────────────────────────────────────────────────────────────────────
fs = SixPanel()   # 3 rows × 2 cols at 360×270 panel scale
gss = subdivide(fs, 3, 2)
begin
    δlab = "δ  (I:E ratio)"
    gklab = "Δg_K  (adaptation)"
    σlab = "σ_ee  (E→E spread)"
    # dg plane (δ × Δg_K)
    circuit_heatmap!(
        gss[1], δ_coarse, gk_coarse, A_coarse;
        xlabel = δlab, ylabel = gklab, title = "Circuit:  a", clabel = "a"
    )
    circuit_heatmap!(
        gss[2], δ_coarse, gk_coarse, B_coarse;
        xlabel = δlab, ylabel = gklab, title = "Circuit:  b", clabel = "b"
    )
    # ds plane (δ × σ_ee)
    circuit_heatmap!(
        gss[3], δ_coarse, σ_coarse, A_ds_coarse;
        xlabel = δlab, ylabel = σlab, title = "Circuit:  a", clabel = "a"
    )
    circuit_heatmap!(
        gss[4], δ_coarse, σ_coarse, B_ds_coarse;
        xlabel = δlab, ylabel = σlab, title = "Circuit:  b", clabel = "b"
    )
    # gs plane (Δg_K × σ_ee)
    circuit_heatmap!(
        gss[5], gk_coarse, σ_coarse, A_gs_coarse;
        xlabel = gklab, ylabel = σlab, title = "Circuit:  a", clabel = "a"
    )
    circuit_heatmap!(
        gss[6], gk_coarse, σ_coarse, B_gs_coarse;
        xlabel = gklab, ylabel = σlab, title = "Circuit:  b", clabel = "b"
    )
end
addlabels!(fs)
suppfile = joinpath(outdir, "hierarchical_variation_supp.pdf")
wsave(suppfile, fs)
@info "Saved $suppfile"
