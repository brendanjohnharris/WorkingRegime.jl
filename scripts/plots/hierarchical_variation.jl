#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WorkingRegime"
using JLD2
using TimeseriesTools
using CairoMakie
using Fathom
using Statistics
using Random
using DelimitedFiles

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

const outdir = plotsdir("hierarchical_variation")
mkpath(outdir)

# Stored ToolsArrays/DimArrays load as real DimArrays via TimeseriesBase's `toolsarray_typemap`; custom
# dims absent from this project (Structure, SessionID, α, ...) come back as generic `Dim{:name}` with
# lookups intact, so we index them by name.
"Index by named values along one or more dims (returns the slice)."
_select(x, selectors::Pair...) = getindex(x; (Symbol(n) => At(v) for (n, v) in selectors)...)

"Pick one element of an array whose elements are themselves arrays."
_select_outer(x, dimname::Symbol, val) = _select(x, dimname => val)

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
# Load aggregated experiment data
# ──────────────────────────────────────────────────────────────────────────────

const inpath = projectdir("WRExperiment", "data", "WRExperiment.jld2")
plot_data = jldopen(
    f -> Dict(k => f[k] for k in keys(f)), inpath;
    typemap = toolsarray_typemap
)

# Circuit per-neuron exponent grids — three planes through the working-regime
# point, each saved as (axis₁, axis₂, seed) of per-neuron exponent vectors. We
# pool seed + neuron to a per-cell median for the heatmaps and the arrows.
const circuit_path = projectdir("WRCircuit", "data", "circuit_exponents.jld2")
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
        _, (l, h) = bootstrapmedian(per)
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
# (α, β) → (a, b) heatmaps: WRTheory/data/bFNS_sweep/flat_γ=0.03_η=0.01.jld2
# holds `diffusion_exponent` and `spectral_exponent` as ToolsArrays over
# (α, β, γ, η, Obs). We NaN-aware average over the Obs seeds (and the singleton
# γ, η axes) to get 2-D (α, β) exponent grids, then reuse `mean_direction` to read
# the local Jacobian directions at the canonical operating point (α = 1.5, β = 0.85).
# ──────────────────────────────────────────────────────────────────────────────

const bfns_path = projectdir(
    "WRTheory", "data", "bFNS_sweep", "flat_γ=0.03_η=0.01.jld2"
)
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

# coeffs_median: ToolsArray{Structure} of ToolsArray{layer, SessionID}
function region_a(structure, layer_idx)
    cm = plot_data["madev_data"][stim]["coeffs_median"]
    inner = _select_outer(cm, :Structure, structure)
    return collect(_select(inner, :layer => layer_idx))
end
# spectral_exponents: ToolsArray{Structure, layer, SessionID}
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

# Variability exponent c (log-log Fano slope over 10^1.5-10^3 ms), derived and cached by
# scripts/plots/variability_variation.jl: `exponents[layer]` is a (SessionID × Structure) matrix in
# `structures` order, so c bootstraps over sessions exactly as a and b do. Read rather than refit ---
# the fit needs the unit-level Fano tables, which are slow to load.
const variability_path = datadir("variability_variation", "variability_exponents.jld2")
const c_exponents = if isfile(variability_path)
    jldopen(f -> f["exponents"], variability_path, "r")
else
    @warn "No $variability_path --- run scripts/plots/variability_variation.jl for the c column"
    nothing
end

"One region's variability exponents across sessions at `layer_idx`; empty if the cache is absent."
function region_c(structure, layer_idx)
    (isnothing(c_exponents) || !haskey(c_exponents, layer_idx)) && return Float64[]
    return c_exponents[layer_idx][:, findfirst(==(structure), structures)]
end

# Categorical colour per region, ordered low → high hierarchy (`structures` is
# already in ascending-score order). Scatter points and the colorbar draw from
# this same discrete gradient so they stay consistent.
const region_colors = cgrad(binarysunset, length(structures); categorical = true)
const structure_color = Dict(s => region_colors[i] for (i, s) in enumerate(structures))
const HEATMAP = sunrise
const δlab = "δ  (I:E ratio)"
const gklab = "Δg_K  (mS/cm²)"

const ARROWS = SHOW_CIRCUIT_ARROWS ? (
        ("δ", δ_dir, qinghai),
        ("Δg_K", gk_dir, seohae),
        ("α", α_dir, baikal),
        ("β", β_dir, bermejo),
    ) : (
        ("α", α_dir, baikal),
        ("β", β_dir, bermejo),
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

function plot_hero!(ax, points; arrow_offset = (0.0, 0.0), axis_ranges = (1.0, 1.0))
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
    for (label, raw, color) in ARROWS
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

# One sub-grid per panel; each holds [axis | colorbar] in its own columns. Row-major:
# gs[1] = L2/3 (a, b); gs[2] = hierarchy τ vs depth (a and b together);
# gs[3,4] = circuit a, b against δ.
gs = subdivide(f, 2, 2)

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
        gs[1][1, 1]; xlabel = "Diffusion exponent  a",
        ylabel = "Spectral exponent  b",
        title = "$stim, $(layer_names[2])", limits = scatterlimits
    )
    plot_hero!(ax_l23, points_l23; axis_ranges = aranges)
end

begin # * Top right — hierarchy correlation of each exponent across cortical depth
    # The continuous version of the scatter: rather than sampling one layer, `hierarchicalkendall`
    # correlates each area's exponent against its hierarchy score at every common depth. Band is the
    # BCa bootstrap CI; filled markers are significant after BH correction, open markers are not.
    # `a` and `b` share one axis (both are Kendall's τ on the same scale) so the row stays 2-wide.
    τ_panels = [("diffusion_hierarchical", "a", mesopelagic)]
    if haskey(plot_data, "spectral_hierarchical")
        push!(τ_panels, ("spectral_hierarchical", "b", ianthina))
    else
        @warn "No `spectral_hierarchical` in $inpath --- re-run collect_calculations.jl for the b series"
    end

    ax_τ = Axis(
        gs[2][1, 1]; xlabel = "Kendall's 𝜏", ylabel = "Cortical depth (%)",
        ytickformat = xs -> string.(round.(Int, 100 .* xs)),
        title = "Hierarchy correlation", yreversed = true
    )
    vlines!(ax_τ, 0; color = :gray, linestyle = :dash, linewidth = 1)
    for (key, sym, color) in τ_panels
        d = plot_data[key][stim]
        τ, 𝑝 = collect(d.μ), collect(d.𝑝)
        depths = collect(d.unidepths)
        σ = collect(d.σ)
        sig = 𝑝 .< PTHR

        band!(
            ax_τ, Point2f.(first.(σ), depths), Point2f.(last.(σ), depths);
            color = (color, 0.25)
        )
        scatter!(ax_τ, τ[sig], depths[sig]; color, markersize = 10, label = sym)
        scatter!(
            ax_τ, τ[.!sig], depths[.!sig]; color = :transparent,
            strokecolor = color, strokewidth = 1, markersize = 10
        )
        @info "$key: $(count(sig))/$(length(sig)) depths significant at p < $PTHR"
    end
    axislegend(ax_τ; position = :rb, framevisible = false, merge = true)
    # Invisible stand-in for the colorbar the other panels carry, so columns share a width.
    Box(gs[2][1, 2]; visible = false, width = 12)
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
        gs[3], _A_dg_full; ylabel = "Diffusion exponent  a", title = "Circuit:  a vs δ"
    )
    ax_dg_b = circuit_lines!(
        gs[4], _B_dg_full; ylabel = "Spectral exponent  b", title = "Circuit:  b vs δ"
    )
end

# The categorical colorbar on the scatter panel doubles as the region legend: one band per
# structure, ordered low → high hierarchy, with "Higher"/"Lower" ends.
const _nreg = length(structures)
Colorbar(
    gs[1][1, 2]; colormap = region_colors, limits = (0, _nreg),
    ticks = ((1:_nreg) .- 0.5, structures), width = 12
)
Label(gs[1][1, 2, Top()], "Higher"; fontsize = 12, padding = (0, 0, -6, 0))
Label(gs[1][1, 2, Bottom()], "Lower"; fontsize = 12, padding = (0, 0, 0, 4))

"""
    save_source_data()

One tab-separated file per panel in `outdir`, holding exactly the values that panel draws. Display
encodings are left out (colours, marker sizes, filled-vs-open significance, and the arrows'
display-normalised length, which is discarded by `plot_hero!` anyway). The arrows' raw `(Δa, Δb)`
displacements ARE kept: they are the panel's quantitative claim and are recoverable from no other
file. Reuses the objects the figure was drawn from, so the files cannot drift from the panels.
"""
function save_source_data()
    # a --- the region scatter with its session-bootstrap 95% CIs.
    writedlm(
        joinpath(outdir, "panelA.tsv"),
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
    # a --- the direction arrows: net (Δa, Δb) displacement per knob, plus their shared origin.
    writedlm(
        joinpath(outdir, "panelA_arrows.tsv"),
        vcat(
            ["knob" "origin_a" "origin_b" "da" "db"],
            reduce(
                vcat,
                [
                    permutedims([lab, ARROW_ORIGIN[1], ARROW_ORIGIN[2], v[1], v[2]])
                        for (lab, v, _) in ARROWS
                ]
            )
        ), '\t'
    )
    # b --- Kendall 𝜏 against hierarchy at each depth. Long format: the two exponents are separate
    # series and need not share a depth grid.
    rows = Any[]
    for (key, sym, _) in τ_panels
        d = plot_data[key][stim]
        for (dep, t, s, 𝑝) in zip(collect(d.unidepths), collect(d.μ), collect(d.σ), collect(d.𝑝))
            push!(rows, permutedims([sym, dep, t, first(s), last(s), 𝑝]))
        end
    end
    writedlm(
        joinpath(outdir, "panelB.tsv"),
        vcat(["exponent" "depth" "tau" "lo" "hi" "p"], reduce(vcat, rows)), '\t'
    )
    # c, d --- the circuit lines: δ against the exponent, one column per Δg_K drawn. `NaN` marks
    # the grid cells the panel skips. Each panel also gets `_lower`/`_upper` files holding the
    # across-seed 95% CI (see `seed_ci`) on the identical grid, so a band can be reconstructed
    # column-by-column without re-reading the sweep.
    sel = drawn_gks()
    hdr = hcat("delta", permutedims(["Delta_g_K=$(gk_dg_lookup[j])" for j in sel]))
    for (name, grid, ci) in
        (("panelC", _A_dg_full, _A_dg_ci), ("panelD", _B_dg_full, _B_dg_ci))
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
        cm, (clo, chi) = bootstrapmedian(region_c(p.structure, layer_idx))
        permutedims([p.structure, p.h, p.a, p.a_lo, p.a_hi, p.b, p.b_lo, p.b_hi, cm, clo, chi])
    end
    pool(f) = reduce(vcat, [collect(f(p.structure, layer_idx)) for p in points_l23])
    am, (alo, ahi) = bootstrapmedian(pool(region_a))
    bm, (blo, bhi) = bootstrapmedian(pool(region_b))
    cm, (clo, chi) = bootstrapmedian(pool(region_c))
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

Reads `WRExperiment/data/surrogates_ft`, written by `WRExperiment/scripts/run_surrogates.jl`. Skips
with a warning if that sweep has not been run.
"""
function save_surrogate_statistics(layer_idx = 2)
    dir = projectdir("WRExperiment", "data", "surrogates_ft")
    files = isdir(dir) ? filter(contains("stimulus=$stim"), readdir(dir; join = true)) : String[]
    if isempty(files)
        @warn "No surrogate sweep in $dir --- run WRExperiment/scripts/run_surrogates.jl"
        return nothing
    end

    # One row per (session, region): the median over this layer's channels, for the data and for the
    # mean of the surrogate draws. Aggregation is replicated inside the null so the two are comparable.
    rows = filter(!isnothing, map(files) do f
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
    end)

    "Exact one-sided sign test: P(at least k of n positive | fair coin). BigInt keeps it exact."
    signtest(d) = (n = length(d); k = count(>(0), d);
        Float64(sum(binomial(big(n), big(i)) for i in k:n) / big(2)^n))

    function summarise(label, rs)
        isempty(rs) && return nothing
        dk = [r.kurt - r.kurt_null for r in rs]
        da = [r.a - r.a_null for r in rs]
        return permutedims([
            label, length(rs),
            median(getfield.(rs, :kurt)), median(getfield.(rs, :kurt_null)), median(dk),
            signtest(dk), count(>(0), dk),
            median(getfield.(rs, :a)), median(getfield.(rs, :a_null)), median(da),
            signtest(da), count(>(0), da),
        ])
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

addlabels!(f)
display(f)
outfile = joinpath(outdir, "hierarchical_variation.pdf")
wsave(outfile, f)
wsave(joinpath(outdir, "hierarchical_variation.png"), f)   # raster preview
@info "Saved $outfile"
save_source_data()
save_statistics()
save_surrogate_statistics()
