#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.13 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate :WorkingRegime
using JLD2
import JSON
import Colors
using Rsvg # activates FathomRsvgExt, so `svgimage!` draws true vector graphics
using TimeseriesTools
using CairoMakie
using Fathom
using Statistics
using Random
import StatsBase: corkendall
using DelimitedFiles

# The per-session hierarchy correlation, shared verbatim with `WRExperiment` and Figure S1.
# `include`d rather than imported: WRExperiment is a workspace member, not a dependency of the
# root project this script activates, so `using WRExperiment` is not available here.
include(projectdir("WRExperiment", "src", "SessionKendall.jl"))

set_theme!(fathom())

const structures = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
const hierarchy_scores = Dict(
    "VISp" => -0.357, "VISl" => -0.093, "VISrl" => -0.059,
    "VISal" => 0.152, "VISpm" => 0.327, "VISam" => 0.441
)
const stim = "spontaneous"
const PTHR = 1.0e-2   # matches WRExperiment.PTHR; the τ p-values are already BH-adjusted across depths

# Layer integer codes in the saved data: 2 = L2/3, 3 = L4, 4 = L5, 5 = L6.
const layer_names = Dict(2 => "L2/3", 3 => "L4", 4 => "L5", 5 => "L6")

# How many Δg_K curves to draw in the bottom row, evenly spaced across the swept range. The sweep
# has 21 Δg_K values, which bundle into an unreadable band; a handful shows the same ordering and
# spread while staying legible. Five lands the working point Δg_K₀ = 0.002 exactly on a drawn line.
const N_GK_LINES = 5

# Span of the local window over which the circuit arrow directions are measured, centred on
# (DELTA_CENTER, GK_CENTER). The Δg_K window is centred on the working point.
const DELTA_DELTA = 0.5
const DELTA_GK = 0.002
const DELTA_CENTER = 3.2
const GK_CENTER = 0.002

# Draw the circuit δ / Δg_K direction arrows on the (a, b) panels. Set false to show only the bFNS α/β arrows.
const SHOW_CIRCUIT_ARROWS = true

# Shared origin of the direction arrows, in (a, b) data coordinates.
const ARROW_ORIGIN = (0.55, -1.75)

NAME = "Fig4_hierarchical_variation"
outdir = plotsdir(NAME)
mkpath(outdir)

"Pick one element of an array whose elements are themselves arrays."
_select_outer(x, dimname::Symbol, val) = _select(x, dimname => val)


# ──────────────────────────────────────────────────────────────────────────────
# Load aggregated experiment data
# ──────────────────────────────────────────────────────────────────────────────

const inpath = datadir("WRExperiment", "WRExperiment.jld2")
plot_data = jldopen(
    f -> Dict(k => f[k] for k in keys(f)), inpath;
    typemap = toolsarray_typemap
)

# Circuit per-neuron exponent grids — three planes through the working-regime
# point, each saved as (axis₁, axis₂, seed) of per-neuron exponent vectors. We
# pool seed + neuron to a per-cell median for the heatmaps and the arrows.
const circuit_path = datadir("WRCircuit", "circuit_exponents.jld2")
circuit = jldopen(
    f -> Dict(k => f[k] for k in keys(f)), circuit_path;
    typemap = toolsarray_typemap
)

"NaN-aware median of one cell's per-neuron exponents POOLED across all seeds (the
trailing grid axis). Each (axis₁, axis₂, seed) cell may be a plain `Vector{Float64}`
(the empty-cell sentinel `Float64[]`) or a `ToolsArray` (the typemap-upgraded
per-neuron array); `collect` normalises both. Empty cells stay NaN."
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

"""
    seed_ci(cells)

Per-cell 95% CI of the exponent ACROSS SEEDS, as `(lower, upper)` matrices laid out like
[`seed_pooled_median`](@ref). Each seed is first reduced to its own median over that seed's
neurons, then the median of those (at most 10) per-seed values is bootstrapped.

The interval therefore measures how much a cell's exponent depends on the network realisation,
and deliberately not the neuron-to-neuron spread, which the pooled median averages away. Note its
centre is the median of per-seed medians, close to but not identical to the pooled median the
panels draw: pooling implicitly weights each seed by how many of its neurons were fit. Cells with
fewer than two surviving seeds stay `NaN`.
"""
function seed_ci(cells)
    n1, n2 = size(cells, 1), size(cells, 2)
    lo = fill(NaN, n1, n2)
    hi = fill(NaN, n1, n2)
    for i in 1:n1, j in 1:n2
        per = Float64[]
        for k in axes(cells, 3)
            v = filter(!isnan, collect(cells[i, j, k]))
            isempty(v) || push!(per, median(v))
        end
        length(per) < 2 && continue
        _, (l, h) = percentilebootmedian(per)
        lo[i, j] = l
        hi[i, j] = h
    end
    return lo, hi
end
# δ/Δg_K plane (delta × Delta_g_K): the joint E/I-ratio × K-adaptation-conductance plane. Reads its
# own lookups (delta, Delta_g_K). Both the full swept δ and the full swept Δg_K range are drawn, so
# nothing is cropped here.
# The map calls are wrapped in `invokelatest` for Julia 1.12's stricter world-age rules on top-level globals.
const _A_dg_full = Base.invokelatest(seed_pooled_median, parent(circuit["a_dg"]))
const _B_dg_full = Base.invokelatest(seed_pooled_median, parent(circuit["b_dg"]))
# Across-seed 95% CIs for the same grids; exported alongside the panel data, not drawn.
const _A_dg_ci = Base.invokelatest(seed_ci, parent(circuit["a_dg"]))
const _B_dg_ci = Base.invokelatest(seed_ci, parent(circuit["b_dg"]))
const δ_dg_lookup = Float64.(collect(circuit["delta"]))
const gk_dg_lookup = Float64.(collect(circuit["Delta_g_K"]))

# Mean direction vectors of the circuit forward map in (a, b) on the δ/Δg_K plane. Rather than
# drawing full isolines, we summarise each knob's effect as a single net displacement
# F(param_max) − F(param_min), averaged over the other parameter to marginalise out the operating point.
#   δ arrow:    δ swept over the DELTA_DELTA window, averaged over the Δg_K window
#   Δg_K arrow: Δg_K swept over the DELTA_GK window, averaged over the δ window
# The a > 1 mask the δ/τ_d plane needed (its fast-rise/slow-decay corner drove the MAD fit past 1) is
# kept as a guard, but this plane tops out near 0.64 so it is inert.
const _dg_bad = _A_dg_full .> 1
const _A_dg_arrow = ifelse.(_dg_bad, NaN, _A_dg_full)
const _B_dg_arrow = ifelse.(_dg_bad, NaN, _B_dg_full)
const δ_arrow_range = (DELTA_CENTER - DELTA_DELTA / 2, DELTA_CENTER + DELTA_DELTA / 2)   # window centered on DELTA_CENTER
const gk_arrow_range = (GK_CENTER - DELTA_GK / 2, GK_CENTER + DELTA_GK / 2)

"Nearest grid index to a target value in a lookup vector."
_nearest(lookup, v) = argmin(abs.(lookup .- v))

"""
    mean_direction(grid_a, grid_b, sweep_lookup, sweep_range, other_lookup, other_range; dim)

Net (Δa, Δb) displacement as the swept parameter goes from `sweep_range[1]` to
`sweep_range[2]`, averaged over the `other` parameter restricted to `other_range`.
`dim = 1` sweeps rows, `dim = 2` sweeps columns.
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

const δ_dir = mean_direction(
    _A_dg_arrow, _B_dg_arrow, δ_dg_lookup, δ_arrow_range,
    gk_dg_lookup, gk_arrow_range; dim = 1
)
const gk_dir = mean_direction(
    _A_dg_arrow, _B_dg_arrow, gk_dg_lookup, gk_arrow_range,
    δ_dg_lookup, δ_arrow_range; dim = 2
)

# ──────────────────────────────────────────────────────────────────────────────
# bFNS theory sweep — (α, β) → (a, b) direction arrows
#
# The flat (unconfined) sweep is the same grid behind the theory figure's
# (α, β) → (a, b) heatmaps: data/WRTheory/bFNS_sweep/flat_γ=0.03_η=0.01.jld2
# holds `diffusion_exponent` and `spectral_exponent` as ToolsArrays over
# (α, β, γ, η, Obs). We NaN-aware average over the Obs seeds (and the singleton
# γ, η axes) to get 2-D (α, β) exponent grids, then reuse `mean_direction` to read
# the local Jacobian directions at the canonical operating point (α = 1.5, β = 0.85).
# ──────────────────────────────────────────────────────────────────────────────

const bfns_path = datadir("WRTheory", "bFNS_sweep", "flat_γ=0.03_η=0.01.jld2")
bfns = jldopen(
    f -> Dict(k => f[k] for k in keys(f)), bfns_path;
    typemap = toolsarray_typemap
)

"NaN-aware mean over a collection; empty / all-NaN → NaN."
function _nanmean(x)
    v = filter(!isnan, vec(collect(x)))
    return isempty(v) ? NaN : mean(v)
end

"Collapse a (α, β, γ, η, Obs) sweep array to a 2-D (α, β) grid by NaN-aware
averaging over every axis after the first two (γ, η singletons + Obs seeds)."
function _ab_grid(na)
    da = parent(na)
    colons = ntuple(_ -> Colon(), ndims(da) - 2)
    return [_nanmean(@view da[i, j, colons...]) for i in axes(da, 1), j in axes(da, 2)]
end

"Lookup vector for dim `name`; falls back to `default` if the typemap dropped the
dim name (e.g. a custom Obs dim whose name fails to parse)."
function _dimlookup(na, name::Symbol, default)
    return collect(hasdim(na, name) ? lookup(na, name) : default)
end

const _α_lookup = _dimlookup(bfns["diffusion_exponent"], :α, range(1.2, 2.0, length = 32))
const _β_lookup = _dimlookup(bfns["diffusion_exponent"], :β, range(0.2, 1.0, length = 32))
const A_ab = Base.invokelatest(_ab_grid, bfns["diffusion_exponent"])
const B_ab = Base.invokelatest(_ab_grid, bfns["spectral_exponent"])

# Local Jacobian directions at the canonical operating point (α = 1.5, β = 0.85):
#   α arrow: α swept 1.35 → 1.65, averaged over β ∈ [0.75, 0.95]
#   β arrow: β swept 0.75 → 0.95, averaged over α ∈ [1.35, 1.65]
# β stops short of 1.0: that boundary cell of the bFNS sweep is corrupted (b snaps -1.94 → -1.58),
# and mean_direction's endpoint-only difference would otherwise cancel Δb and point the arrow sideways.
const α_arrow_range = (1.35, 1.65)
const β_arrow_range = (0.75, 0.95)
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

# The `_raw` forms keep the SessionID lookup, which `exponent_matrix` needs to align the areas by
# session identifier. The areas do NOT share a session list --- VISp has 68 sessions against the
# other five areas' 69 --- so anything that pairs them positionally is silently wrong.
# coeffs_median: ToolsArray{Structure} of ToolsArray{layer, SessionID}
function region_a_raw(structure, layer_idx)
    cm = plot_data["madev_data"][stim]["coeffs_median"]
    return _select(_select_outer(cm, :Structure, structure), :layer => layer_idx)
end
# spectral_exponents: ToolsArray{SessionID, Structure, layer}
region_b_raw(s, layer_idx) = _select(
    plot_data["spectral_exponents"][stim],
    :Structure => s, :layer => layer_idx
)
region_a(structure, layer_idx) = collect(region_a_raw(structure, layer_idx))
region_b(s, layer_idx) = collect(region_b_raw(s, layer_idx))

"Per-region (a, b) bootstrap medians + 95% CI at one layer, sorted low → high hierarchy."
function compute_points(layer_idx)
    pts = filter(
        !isnothing, map(structures) do s
            try
                am, (alo, ahi) = percentilebootmedian(region_a(s, layer_idx))
                bm, (blo, bhi) = percentilebootmedian(region_b(s, layer_idx))
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

# Variability exponent c (the unified BIC-selected MAPPLE fit to each session's unit-median Fano
# curve; see WRExperiment/scripts/variability_variation.jl, which derives and
# caches it): `exponents[layer]` is a (SessionID × Structure) matrix in `structures` order, so c
# bootstraps over sessions exactly as a and b do. Read rather than refit --- the fit needs the
# unit-level Fano tables, which are slow to load.
const variability_path = datadir(
    "WRExperiment", "variability_variation", "variability_exponents.jld2"
)
isfile(variability_path) ||
    error("No $variability_path --- run WRExperiment/scripts/variability_variation.jl for the c column")
const c_exponents = jldopen(f -> f["exponents"], variability_path, "r")

"One region's variability exponents across sessions at `layer_idx`; empty if the layer is absent."
function region_c(structure, layer_idx)
    haskey(c_exponents, layer_idx) || return Float64[]
    return filter(isfinite, c_exponents[layer_idx][:, findfirst(==(structure), structures)]) # sessions lacking this area are NaN
end

# Categorical colour per region, ordered low → high hierarchy (`structures` is
# already in ascending-score order). Scatter points and the colorbar draw from
# this same discrete gradient so they stay consistent.
const region_colors = cgrad(binarysunset, length(structures); categorical = true)
const structure_color = Dict(s => region_colors[i] for (i, s) in enumerate(structures))
const HEATMAP = sunrise
# One rich form per symbol, so the axis label, the colorbar and the arrow tip cannot disagree.
const δsym = mit("δ")
const gksym = rich(mit("Δg"), subscript(mit("K")))
const asym = mit("a")
const bsym = mit("b")
const δlab = unitlabel(δsym, "I:E ratio")
# µS, not mS/cm²: the model is a point neuron, `C dV/dt = -gL(V - VL) - gK(V - VK) + I` with V in mV
# and I in nA, so conductances are nA/mV = µS (C = 0.25 nF, gL = 0.0167 µS give τm = 15 ms).
const gklab = unitlabel(gksym, "µS")

# `(key, label, direction, colour)`: the key names the knob in the source data, the label draws it.
const ARROWS = SHOW_CIRCUIT_ARROWS ? (
        ("δ", δsym, δ_dir, qinghai),
        ("Δg_K", gksym, gk_dir, seohae),
        ("α", mit("α"), α_dir, baikal),
        ("β", mit("β"), β_dir, bermejo),
    ) : (
        ("α", mit("α"), α_dir, baikal),
        ("β", mit("β"), β_dir, bermejo),
    )

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

function plot_ab_plane!(ax, points; arrow_offset = (0.0, 0.0), axis_ranges = (1.0, 1.0))
    xs = [p.a for p in points]
    ys = [p.b for p in points]

    vlines!(ax, [0.5]; color = :gray, linestyle = :dash, linewidth = 1)

    # Mean-direction arrows, all sharing one origin fixed at ARROW_ORIGIN in (a, b)
    # data coordinates (plus an optional per-panel `arrow_offset` in data units).
    # Two read the circuit forward map (δ, Δg_K), two read the bFNS theory map (α, β).
    # We only care about the *angle*: each vector is normalised to display units (each
    # component ÷ its axis range) so it points along the direction read off the panel,
    # then drawn at a fixed length. Magnitude is discarded.
    ox = ARROW_ORIGIN[1] + arrow_offset[1]
    oy = ARROW_ORIGIN[2] + arrow_offset[2]
    ra, rb = axis_ranges                             # display ranges (a, b)
    arrow_len = 0.25                                  # fraction of each axis range
    function scaled(v)
        u = (v[1] / ra, v[2] / rb)                   # display-fraction direction
        n = hypot(u...)
        n == 0 && return (0.0, 0.0)
        return (u[1] / n * ra, u[2] / n * rb) .* arrow_len
    end

    # All the arrows share the origin: circuit knobs (δ, Δg_K) in green/orange, bFNS orders in
    # blue/red. `ARROWS` holds the raw displacements; `scaled` is display normalisation only.
    for (_, label, raw, color) in ARROWS
        v = scaled(raw)
        arrows2d!(
            ax, [Point2f(ox, oy)], [Vec2f(v...)];
            color, tipwidth = 12, tiplength = 12, shaftwidth = 2.5
        )
        # Label each arrow directly at its tip (no legend), nudged a few pixels
        # further along the arrow and coloured to match.
        label_tip!(ax, (ox, oy), v, label, color)
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
        ax, xs, ys; color = [structure_color[p.structure] for p in points],
        markersize = 18, strokecolor = :black, strokewidth = 0.8
    )
    return ax
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure: the L2/3 (a, b) plane and the exponents' hierarchy correlation across
# depth on top; the circuit's δ-dependence below
# ──────────────────────────────────────────────────────────────────────────────

f = FourPanel()

# Top row: [cortex map | (a, b) scatter | hierarchy τ] in one nested grid; bottom row keeps the
# root cells. `gs` preserves the panel order the rest of the script indexes:
# gs[1] = L2/3 (a, b); gs[2] = hierarchy τ vs depth; gs[3,4] = circuit a, b against δ.
# The bottom row's colorbar labels protrude 64 px right of the root content area, so a plainly
# nested top grid ends 64 px short of the figure's drawn edge. The negative right Outside reaches
# over that band; the other terms cancel this grid's own protrusions, which Outside folds inside
# (measured via layouttree: left 26 = panel letter, bottom 52 = xlabels, top 31 = titles+letters).
gtop = f[1, 1:2] = GridLayout(; alignmode = Outside(-51, -60, 0, 0))
# The map column has no x-label, so the band (b) and (c) reserve for theirs is dead space here:
# reach down into it. The top term cancels the map title's protrusion, which Outside folds inside.
g_cortex = gtop[1, 1] = GridLayout(; alignmode = Outside(0, 0, -44, -20))
gs = [gtop[1, 2], gtop[1, 3], f[2, 1], f[2, 2]]

begin # * Top left — the visual cortical areas, coloured by hierarchy position
    # Geometry from SpatiotemporalMotifs.jl: the svg is embedded as vector art (its dense area
    # paths tessellate badly through `poly!`), recoloured in place to this figure's hierarchy
    # palette; the json supplies the label centroids.
    svg = read(projectdir("assets", "visual_cortex.svg"), String)
    svg = replace(
        svg, r"<path\b[^>]*>" => function (tag)
            m = match(r"id=\"(VIS\w+)\"", tag) # fill paths; lowercase ids are the outlines
            isnothing(m) && return tag
            i = findfirst(==(m[1]), structures)
            isnothing(i) && return tag
            hexcol = "#" * Colors.hex(Colors.RGB(region_colors[i]))
            tag = replace(tag, r" opacity=\"[^\"]*\"" => " opacity=\"1\"") # SM draws them washed at 0.42
            return replace(tag, r"fill=\"[^\"]*\"" => "fill=\"$hexcol\"")
        end
    )
    W, H = Fathom.svgsize(svg)

    ax_map = Axis(g_cortex[1, 1]; aspect = DataAspect(), title = "Mouse visual cortex",
                  halign = :left) # letterboxes in its cell; hug the page edge
    hidedecorations!(ax_map)
    hidespines!(ax_map)
    svgimage!(ax_map, svg)

    cortex = JSON.parsefile(projectdir("assets", "visual_cortex.json")) # image coordinates, y down
    # The svg's areas overlap (VISp paints over its neighbours' centroids), so the small/hidden
    # areas get hand-placed anchors on their visible parts; the rest use their path centroid.
    label_pos = Dict("VISpm" => (232.0, 120.0), "VISam" => (205.0, 45.0), "VISal" => (28.0, 78.0))
    for s in structures # white with a dark halo: readable over any fill, and across boundaries
        pts = cortex["fill"][s]
        cx, cy = get(label_pos, s, (mean(first.(pts)), mean(last.(pts))))
        text!(
            ax_map, Point2f(cx, H - cy); text = s, # y flipped: svg draws y-up
            align = (s == "VISpm" ? :right : :center, :center),
            fontsize = 10, color = :white, glowcolor = chernoe, glowwidth = 6
        )
    end
    # Horizontal hierarchy scale beneath the map. The in-map labels name the areas, so the bar
    # carries only the ordering; it colour-keys the scatter panel too.
    Colorbar(
        g_cortex[2, 1]; colormap = region_colors, limits = (0, length(structures)),
        ticksvisible = false, ticklabelsvisible = false, vertical = false, height = 10
    )
    ax_key = Axis(g_cortex[3, 1]; height = 14, limits = ((0, 1), (0, 1))) # Lower ⟶ Higher, under the bar
    hidedecorations!(ax_key)
    hidespines!(ax_key)
    text!(ax_key, 0, 0.5; text = "Lower", align = (:left, :center), fontsize = 12)
    text!(ax_key, 1, 0.5; text = "Higher", align = (:right, :center), fontsize = 12)
    arrows2d!(
        ax_key, [Point2f(0.3, 0.5)], [Vec2f(0.4, 0)];
        color = chernoe, shaftwidth = 1.5, tipwidth = 8, tiplength = 8
    )
    rowgap!(g_cortex, 3)
    colsize!(gtop, 1, Fixed(185)) # pin the map column; the rest of the row goes to (b) and (c)
end

begin # * Top left — (a, b) plane at L2/3
    scatterlimits = (
        (0.42, 0.65),   # a
        (-1.9, -1.5),    # b
    )
    # Display ranges (a, b) — arrows normalise their direction by these.
    aranges = (
        scatterlimits[1][2] - scatterlimits[1][1],
        scatterlimits[2][2] - scatterlimits[2][1],
    )
    ax_l23 = Axis(
        gs[1][1, 1]; xlabel = rich("Diffusion exponent ", asym),
        ylabel = rich("Spectral exponent ", bsym),
        # The layer comes from `layer_names` rather than being spelled out, so the title cannot
        # drift from the data. The stimulus ($stim) lives in the caption and in `statistics.tsv`.
        # "Hierarchical" overflowed the ~190-unit panel and ran into (c)'s letter; "Hierarchy" fits.
        title = "Hierarchy exponents ($(layer_names[2]))", limits = scatterlimits
    )
    plot_ab_plane!(ax_l23, points_l23; axis_ranges = aranges)
end

begin # * Top right — hierarchy correlation of each exponent, by layer
    # One Kendall 𝜏 per SESSION: rank that session's own areas against their hierarchy scores, then
    # show the distribution across sessions. Ranking within a session removes between-session
    # variance, which dominates these exponents --- across the twelve (exponent, layer) cells it is
    # 22-61% of the total variance and 64-95% of the variance the session and area effects share
    # (b at L2/3: 38% and 65%) --- and which the previous pooled-across-sessions 𝜏 spent most of its
    # pairs on. Pooling attenuates accordingly: b at L2/3 gives 0.38 pooled against 0.60 here.
    #
    # Layers rather than depths: the variability exponent c is only defined per layer --- its unit
    # Fano curves carry no finer binning --- so this is the finest grid all three exponents share.
    τ_layers = 2:5
    # Colours match the bFNS arrows in panel b: a ↔ α blue, b ↔ β red. c has no arrow, so it takes glas.
    τ_series = [("a", baikal), ("b", bermejo), ("c", glas)]
    τ_offsets = Dict("a" => -0.18, "b" => 0.0, "c" => 0.18)
    τ_hier = [hierarchy_scores[s] for s in structures]
    const MIN_AREAS = 4      # areas a session must retain before its 𝜏 is used

    """
    The `(session × area)` matrix of one exponent at one layer, on the session set the areas share.

    `a` and `b` are stored per area with their own SessionID lookups, so they are aligned by
    identifier through `sessionmatrix`; `c` is already stored as one `(session × area)` matrix.
    """
    function exponent_matrix(sym, layer_idx)
        if sym == "c"
            haskey(c_exponents, layer_idx) || return nothing
            return Float64.(c_exponents[layer_idx])
        end
        raw = sym == "a" ? region_a_raw : region_b_raw
        cols = [raw(s, layer_idx) for s in structures]
        any(isempty, cols) && return nothing
        return first(sessionmatrix([lookup(c, :SessionID) for c in cols], [collect(c) for c in cols]))
    end

    # One `sessionkendall` per (exponent, layer) cell, then Benjamini-Hochberg across the 12 cells
    # this panel draws. The correction belongs here rather than inside the estimator, which cannot
    # know what family it is part of.
    τ_cells = NamedTuple[]
    for (sym, _) in τ_series, l in τ_layers
        Y = exponent_matrix(sym, l)
        isnothing(Y) && continue
        r = sessionkendall(τ_hier, Y; minareas = MIN_AREAS)
        r.nsessions < 5 && continue
        push!(τ_cells, (; sym, layer = l, r...))
    end
    τ_cells = [(; c..., padj = q) for (c, q) in zip(τ_cells, bhadjust([c.p for c in τ_cells]))]
    for c in τ_cells
        @info "τ $(c.sym) $(layer_names[c.layer]): median $(round(c.tau; digits = 3)) " *
            "[$(round(c.ci[1]; digits = 3)), $(round(c.ci[2]; digits = 3))], " *
            "mean $(round(c.meantau; digits = 3)), p = $(round(c.p; sigdigits = 2)), " *
            "p_adj = $(round(c.padj; sigdigits = 2)), n = $(c.nsessions)"
    end

    ax_τ = Axis(
        gs[2][1, 1]; xlabel = rich("Kendall's ", mit("τ")), ylabel = "Cortical layer",
        yticks = (collect(τ_layers), [layer_names[l] for l in τ_layers]),
        title = "Hierarchical correlation", yreversed = true,
        # The full range of 𝜏, so the per-session strip is not clipped: a six-area session can
        # reach ±1, and 34 of the 772 drawn values sit beyond ±0.85.
        xticks = -1:0.5:1,
        limits = ((-1.08, 1.08), (first(τ_layers) - 0.62, last(τ_layers) + 0.62))
    )
    hlines!(ax_τ, τ_layers[1:(end - 1)] .+ 0.5; color = (:gray, 0.4), linewidth = 0.5) # layer separators
    vlines!(ax_τ, 0; color = :gray, linestyle = :dash, linewidth = 1)

    # The per-session 𝜏 behind each median, as a jittered strip under its interval. Six areas put
    # every session's 𝜏 on a 1/15 grid, so the median lands on a grid point and a bootstrap endpoint
    # can coincide with it; drawing the sample makes that granularity explicit rather than leaving a
    # zero-width whisker looking like a typo. Seeded, so the jitter is the same on every rerun.
    τ_jitter = Random.MersenneTwister(7)
    for (sym, color) in τ_series
        symlab = mit(sym) # one object, reused: `merge = true` pairs the legend entries by equality
        cells = [c for c in τ_cells if c.sym == sym]
        isempty(cells) && continue
        for c in cells
            y0 = c.layer + τ_offsets[sym]
            scatter!(
                ax_τ, c.taus, y0 .+ (rand(τ_jitter, length(c.taus)) .- 0.5) .* 0.24;
                color = (color, 0.22), markersize = 3, strokewidth = 0
            )
        end
        # Light connector so each exponent reads as a profile down the layers; the marker is the
        # median and the whisker its percentile bootstrap CI over sessions. Filled where the
        # BH-adjusted permutation p clears PTHR, open where it does not. The CI is drawn either way:
        # it describes the median's precision, and is no longer what decides significance.
        ys = [c.layer + τ_offsets[sym] for c in cells]
        ms = [c.tau for c in cells]
        los, his = [c.ci[1] for c in cells], [c.ci[2] for c in cells]
        sig = [c.padj < PTHR for c in cells]
        lines!(ax_τ, ms, ys; color = (color, 0.4), linewidth = 1.5)
        rangebars!(ax_τ, ys, los, his; direction = :x, color, linewidth = 1.5, whiskerwidth = 6)
        scatter!(ax_τ, ms[sig], ys[sig]; color, markersize = 10, label = symlab)
        any(.!sig) && scatter!(
            ax_τ, ms[.!sig], ys[.!sig]; color = :transparent, strokecolor = color,
            strokewidth = 1, markersize = 10, label = symlab
        )
    end
    axislegend(ax_τ; position = :rb, framevisible = false, merge = true, patchsize = (10, 10))
end

"Indices into `gk_dg_lookup` of the Δg_K values the bottom row draws: `nlines` evenly spaced across
the swept range. Shared by the drawing and the saved source data so the two cannot disagree."
function drawn_gks(nlines = N_GK_LINES)
    return unique(round.(Int, range(1, length(gk_dg_lookup); length = nlines)))
end

"""
    circuit_lines!(pos, grid; ylabel, title)

`N_GK_LINES` lines, evenly spaced across the swept Δg_K range, plotted against δ over the full swept
range. Lines rather than a heatmap: they show the δ-dependence and the spread across Δg_K more
directly than a colour scale, and they make the saturation at high δ legible. The colorbar is
categorical and ticked with the Δg_K values actually drawn, so it is a legend for the lines rather
than a continuous scale over values that are not shown.
"""
function circuit_lines!(pos, grid; ylabel, title, nlines = N_GK_LINES)
    ax = Axis(pos[1, 1]; xlabel = δlab, ylabel = ylabel, title = title)
    sel = drawn_gks(nlines)
    cols = cgrad(HEATMAP, max(2, length(sel)); categorical = true)
    for (i, j) in enumerate(sel)
        v = grid[:, j]
        k = findall(!isnan, v)
        isempty(k) && continue
        lines!(ax, δ_dg_lookup[k], v[k]; color = cols[i], linewidth = 2.5)
    end
    Colorbar(
        pos[1, 2]; colormap = cols, limits = (0, length(sel)),
        ticks = ((1:length(sel)) .- 0.5, string.(gk_dg_lookup[sel])),
        label = gklab, width = 12
    )
    return ax
end

begin # * Bottom row — circuit exponents against δ, one line per Δg_K
    ax_dg_a = circuit_lines!(
        gs[3], _A_dg_full; ylabel = rich("Diffusion exponent ", asym),
        title = "Circuit diffusion variation"
    )
    ax_dg_b = circuit_lines!(
        gs[4], _B_dg_full; ylabel = rich("Spectral exponent ", bsym),
        title = "Circuit spectral variation"
    )
end

"""
    save_source_data()

One tab-separated file per panel in `outdir`, holding exactly the values that panel draws. Display
encodings are left out (colours, marker sizes, filled-vs-open significance, and the arrows'
display-normalised length, which is discarded by `plot_ab_plane!` anyway). The arrows' raw `(Δa, Δb)`
displacements ARE kept: they are the panel's quantitative claim and are recoverable from no other
file. Reuses the objects the figure was drawn from, so the files cannot drift from the panels.
"""
function save_source_data()
    # b --- the region scatter with its session-bootstrap 95% CIs. Panel a is the cortex map,
    # whose geometry is assets/visual_cortex.json and whose colours are the hierarchy scores above.
    writedlm(
        joinpath(outdir, "panelB.tsv"),
        vcat(
            ["structure" "hierarchy" "a" "a_lo" "a_hi" "b" "b_lo" "b_hi"],
            reduce(
                vcat,
                [
                    permutedims([p.structure, p.h, p.a, p.a_lo, p.a_hi, p.b, p.b_lo, p.b_hi])
                        for p in points_l23
                ]
            )
        ), '\t'
    )
    # b --- the direction arrows: net (Δa, Δb) displacement per knob, plus their shared origin.
    writedlm(
        joinpath(outdir, "panelB_arrows.tsv"),
        vcat(
            ["knob" "origin_a" "origin_b" "da" "db"],
            reduce(
                vcat,
                [
                    permutedims([key, ARROW_ORIGIN[1], ARROW_ORIGIN[2], v[1], v[2]])
                        for (key, _, v, _) in ARROWS
                ]
            )
        ), '\t'
    )
    # c --- per-session Kendall 𝜏 by layer. Long format: one row per (exponent, layer, session),
    # so the boxes are reconstructible and the distribution is not reduced to a summary.
    rows = Any[]
    for c in τ_cells, (t, na) in zip(c.taus, c.nareas)
        push!(rows, permutedims([c.sym, layer_names[c.layer], t, na]))
    end
    writedlm(
        joinpath(outdir, "panelC.tsv"),
        vcat(["exponent" "layer" "session_tau" "n_areas"], reduce(vcat, rows)), '\t'
    )
    # c --- the summary actually drawn, so the prose has a file to cite. `tau` is the across-session
    # median and `ci` its percentile bootstrap interval; `p` is the within-session label-permutation
    # test on the mean 𝜏, `p_adj` its Benjamini-Hochberg value across these 12 cells, and
    # `significant` the filled/open rule the panel uses (p_adj < PTHR).
    writedlm(
        joinpath(outdir, "panelC_stats.tsv"),
        vcat(
            ["exponent" "layer" "n_sessions" "tau" "ci_lo" "ci_hi" "mean_tau" "p" "p_adj" "significant"],
            reduce(
                vcat,
                [
                    permutedims(
                        [
                            c.sym, layer_names[c.layer], c.nsessions, c.tau, c.ci[1], c.ci[2],
                            c.meantau, c.p, c.padj, c.padj < PTHR,
                        ]
                    ) for c in τ_cells
                ]
            )
        ), '\t'
    )
    # d, e --- the circuit lines: δ against the exponent, one column per Δg_K drawn. `NaN` marks
    # the grid cells the panel skips. Each panel also gets `_lower`/`_upper` files holding the
    # across-seed 95% CI (see `seed_ci`) on the identical grid, so a band can be reconstructed
    # column-by-column without re-reading the sweep.
    sel = drawn_gks()
    hdr = hcat("delta", permutedims(["Delta_g_K=$(gk_dg_lookup[j])" for j in sel]))
    for (name, grid, ci) in
        (("panelD", _A_dg_full, _A_dg_ci), ("panelE", _B_dg_full, _B_dg_ci))
        for (suffix, g) in (("", grid), ("_lower", first(ci)), ("_upper", last(ci)))
            writedlm(
                joinpath(outdir, "$name$suffix.tsv"),
                vcat(hdr, hcat(δ_dg_lookup, g[:, sel])), '\t'
            )
        end
    end
    return @info "Saved source data to $outdir"
end

"""
    save_statistics(layer_idx = 2)

`statistics.tsv`: the three exponents at one layer --- diffusion `a`, spectral `b` and variability
`c` --- each as a session-bootstrap median with its 95% CI, per region and pooled over every
(session, region) pair. These are the numbers quoted in the text; the panels themselves draw only
`a` and `b`, so `save_source_data` does not carry `c`. `a` and `b` are taken from `points_l23` so
the file cannot disagree with panel a.
"""
function save_statistics(layer_idx = 2)
    rows = map(points_l23) do p
        cm, (clo, chi) = percentilebootmedian(region_c(p.structure, layer_idx))
        permutedims([p.structure, p.h, p.a, p.a_lo, p.a_hi, p.b, p.b_lo, p.b_hi, cm, clo, chi])
    end
    pool(f) = reduce(vcat, [collect(f(p.structure, layer_idx)) for p in points_l23])
    am, (alo, ahi) = percentilebootmedian(pool(region_a))
    bm, (blo, bhi) = percentilebootmedian(pool(region_b))
    cm, (clo, chi) = percentilebootmedian(pool(region_c))
    push!(rows, permutedims(["pooled", NaN, am, alo, ahi, bm, blo, bhi, cm, clo, chi]))
    writedlm(
        joinpath(outdir, "statistics.tsv"),
        vcat(
            ["structure" "hierarchy" "a" "a_lo" "a_hi" "b" "b_lo" "b_hi" "c" "c_lo" "c_hi"],
            reduce(vcat, rows)
        ), '\t'
    )
    @info "$(layer_names[layer_idx]) pooled across regions: " *
        "a = $(round(am, digits = 3)) [$(round(alo, digits = 3)), $(round(ahi, digits = 3))], " *
        "b = $(round(bm, digits = 3)) [$(round(blo, digits = 3)), $(round(bhi, digits = 3))], " *
        "c = $(round(cm, digits = 3)) [$(round(clo, digits = 3)), $(round(chi, digits = 3))]"
    return @info "Saved statistics to $(joinpath(outdir, "statistics.tsv"))"
end

"""
    save_surrogate_statistics(layer_idx = 2)

`surrogate_statistics.tsv`: the two surrogate-controlled statistics of the experimental LFP at one
layer --- excess kurtosis of the single-sample increments and the diffusion exponent `a` --- each as
the session median of the data value, of the surrogate null, and of their paired difference, per
region and pooled.

The null is FT (phase randomisation), i.e. a linear Gaussian process with the data's power spectrum,
so both statistics are one-sided LARGER than the null. Excess kurtosis has null value exactly 0 by
construction, making the data value itself the effect size; for `a` the difference is
`ζ(1) − ζ(2)/2`, the first-order intermittency coefficient, which vanishes for any monofractal
process. `p` is an exact sign test across sessions (a signed-rank would need HypothesisTests, which
this project does not carry); with the observed consistency it is far from the deciding factor.

Reads `data/WRExperiment/surrogates_ft`, written by `WRExperiment/scripts/run_surrogates.jl`.
"""
function save_surrogate_statistics(layer_idx = 2)
    dir = datadir("WRExperiment", "surrogates_ft")
    files = isdir(dir) ? filter(contains("stimulus=$stim"), readdir(dir; join = true)) : String[]
    isempty(files) && error("No surrogate sweep in $dir --- run WRExperiment/scripts/run_surrogates.jl")

    # One row per (session, region): the median over this layer's channels, for the data and for the
    # mean of the surrogate draws. Aggregation is replicated inside the null so the two are comparable.
    rows = filter(
        !isnothing, map(files) do f
            D = jldopen(g -> Dict(k => g[k] for k in keys(g)), f)
            haskey(D, "error") && return nothing
            sel = findall(D["layernums"] .== layer_idx)
            length(sel) < 2 && return nothing
            m = match(r"sessionid=(\d+).*structure=([A-Za-z0-9\-]+)\.jld2", basename(f))
            agg(v, j) = median(filter(!isnan, getfield(v, j)[sel]))
            (
                structure = String(m[2]),
                kurt = agg(D["s0"], :kurt), kurt_null = mean(agg.(D["s"], :kurt)),
                a = agg(D["s0"], :a), a_null = mean(agg.(D["s"], :a)),
            )
        end
    )

    "Exact one-sided sign test: P(at least k of n positive | fair coin). BigInt keeps it exact."
    signtest(d) = (
        n = length(d); k = count(>(0), d);
        Float64(sum(binomial(big(n), big(i)) for i in k:n) / big(2)^n)
    )

    function summarise(label, rs)
        isempty(rs) && return nothing
        dk = [r.kurt - r.kurt_null for r in rs]
        da = [r.a - r.a_null for r in rs]
        return permutedims(
            [
                label, length(rs),
                median(getfield.(rs, :kurt)), median(getfield.(rs, :kurt_null)), median(dk),
                signtest(dk), count(>(0), dk),
                median(getfield.(rs, :a)), median(getfield.(rs, :a_null)), median(da),
                signtest(da), count(>(0), da),
            ]
        )
    end

    out = filter(
        !isnothing, [
            [summarise(s, filter(r -> r.structure == s, rows)) for s in structures]...,
            summarise("pooled", rows),
        ]
    )
    writedlm(
        joinpath(outdir, "surrogate_statistics.tsv"),
        vcat(
            [
            "structure" "n_sessions" "kurtosis" "kurtosis_null" "kurtosis_delta" "kurtosis_p" "kurtosis_consistent" "a" "a_null" "a_delta" "a_p" "a_consistent"
            ],
            reduce(vcat, out)
        ), '\t'
    )
    pooled = only(filter(r -> first(r) == "pooled", out))
    @info "$(layer_names[layer_idx]) pooled: excess kurtosis = $(round(pooled[3], digits = 3)) " *
        "(null $(round(pooled[4], digits = 3))), a = $(round(pooled[8], digits = 3)) " *
        "(null $(round(pooled[9], digits = 3)))"
    return @info "Saved statistics to $(joinpath(outdir, "surrogate_statistics.tsv"))"
end

# Taller top row: the map is aspect-locked, so its size is set by the row height, and the extra
# height also squares up (b) and (c), which were landscape at the default even split.
rowsize!(f.layout, 1, Relative(0.55))
# (b)'s title overhangs its panel by 12 each side, leaving (c)'s letter only 5 units of clearance
# while (c)'s own title has 37 to spare; split the difference.
addlabels!([gtop[1, 1], gtop[1, 2], gtop[1, 3], f[2, 1], f[2, 2]], f; dx = [0, 0, 8, 0, 0])
display(f)
outfile = joinpath(outdir, "$NAME.pdf")
wsave(outfile, f)
wsave(joinpath(outdir, "$NAME.png"), f)   # raster preview
@info "Saved $outfile"
save_source_data()
save_statistics()
save_surrogate_statistics()
