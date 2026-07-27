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

set_theme!(fathom())

const structures = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
const hierarchy_scores = Dict(
    "VISp" => -0.357, "VISl" => -0.093, "VISrl" => -0.059,
    "VISal" => 0.152, "VISpm" => 0.327, "VISam" => 0.441
)
const stim = "spontaneous"

# Layer integer codes in the saved data: 2 = L2/3, 3 = L4, 4 = L5, 5 = L6.
const layer_names = Dict(2 => "L2/3", 3 => "L4", 4 => "L5", 5 => "L6")

# δ/τ_d plane ROI: framed on the L2/3 hierarchy path plus the working point (δ_0 = 4, τd_0 = 5). The data a/b
# now use the same MAPPLE estimator as the circuit (WRExperiment.diffusion_fit/mapple_fit), so no offset
# correction is applied. Re-tune these bounds if the regenerated data lands elsewhere.
const DELTA_MIN = 2.75
const DELTA_MAX = 3.4
const TAU_D_MIN = 4.5
const TAU_D_MAX = 5.5

# Span of the local window over which the circuit arrow directions are measured
const DELTA_DELTA = 0.5
const DELTA_TAU_D = 0.4
const DELTA_CENTER = 3.2
const TAU_D_CENTER = 4.75

# Draw the circuit δ / τ_d direction arrows on the (a, b) panels. Set false to show only the bFNS α/β arrows.
const SHOW_CIRCUIT_ARROWS = true

# Overlay the L2/3 hierarchy path (region scatter + connecting line) on the δ/τ_d heatmaps.
const SHOW_DTD_REGIONS = false

# Shared origin of the direction arrows, in (a, b) data coordinates.
const ARROW_ORIGIN = (0.55, -1.75)

# Middle-row curve panels: the δ values at which the full input MAD and spectrum are drawn, decreasing from
# the working point (δ_0 = 4), with τ_d_e held there. Each request is snapped to the nearest swept cell, and
# the curves are pooled over neurons and over `CURVE_SEEDS` connectome seeds.
const CURVE_DELTAS = [3.5, 3.25, 3.0]
const CURVE_TAU_D = 5.0
const CURVE_SEEDS = 1:10

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
# δ/τ_d plane (delta × tau_d_e): the joint E/I-ratio × excitatory-decay plane, sliced to the δ ROI
# (critical region through the working point) × the τ_d ROI. Reads its own lookups (dtd_delta, dtd_tau_d_e).
# The map calls are wrapped in `invokelatest` for Julia 1.12's stricter world-age rules on top-level globals.
const _A_dtd_full = Base.invokelatest(seed_pooled_median, parent(circuit["a_dtd"]))
const _B_dtd_full = Base.invokelatest(seed_pooled_median, parent(circuit["b_dtd"]))
const δ_dtd_lookup = collect(circuit["dtd_delta"])
const τd_dtd_lookup = collect(circuit["dtd_tau_d_e"])
const δ_dtd_keep = findall(v -> DELTA_MIN <= v <= DELTA_MAX, δ_dtd_lookup)
const τd_dtd_keep = findall(v -> TAU_D_MIN <= v <= TAU_D_MAX, τd_dtd_lookup)
const δ_dtd = δ_dtd_lookup[δ_dtd_keep]
const τd_dtd = τd_dtd_lookup[τd_dtd_keep]
# Default working-regime point (δ_0, τd_0) --- Spatial model defaults snapped to the grid;
# falls back to the known defaults if the sweep predates saving the anchor.
const δ_0 = haskey(circuit, "delta_0") ? Float64(circuit["delta_0"]) : 4.0
const τd_0 = haskey(circuit, "tau_d_e_0") ? Float64(circuit["tau_d_e_0"]) : 5.0
const A_dtd = _A_dtd_full[δ_dtd_keep, τd_dtd_keep]   # (δ × τ_d)
const B_dtd = _B_dtd_full[δ_dtd_keep, τd_dtd_keep]

# Mean direction vectors of the circuit forward map in (a, b) on the δ/τ_d plane. Rather than
# drawing full isolines, we summarise each knob's effect as a single net displacement
# F(param_max) − F(param_min), averaged over the other parameter to marginalise out the operating point.
#   δ arrow:   δ swept over the DELTA_DELTA window, averaged over the τ_d window
#   τ_d arrow: τ_d swept over the DELTA_TAU_D window, averaged over the δ window
# The fast-rise/slow-decay corner drives the 2-component MAD fit's first component past 1 (an artefact, not a
# regime), so mask a > 1 to NaN; mean_direction skips those endpoint cells even though the sweep spans the ROI.
const _dtd_bad = _A_dtd_full .> 1
const _A_dtd_arrow = ifelse.(_dtd_bad, NaN, _A_dtd_full)
const _B_dtd_arrow = ifelse.(_dtd_bad, NaN, _B_dtd_full)
const δ_arrow_range = (DELTA_CENTER - DELTA_DELTA / 2, DELTA_CENTER + DELTA_DELTA / 2)   # window centered on DELTA_CENTER
const τd_arrow_range = (TAU_D_CENTER - DELTA_TAU_D / 2, TAU_D_CENTER + DELTA_TAU_D / 2)

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
    _A_dtd_arrow, _B_dtd_arrow, δ_dtd_lookup, δ_arrow_range,
    τd_dtd_lookup, τd_arrow_range; dim = 1
)
const τd_dir = mean_direction(
    _A_dtd_arrow, _B_dtd_arrow, τd_dtd_lookup, τd_arrow_range,
    δ_dtd_lookup, δ_arrow_range; dim = 2
)

# ──────────────────────────────────────────────────────────────────────────────
# Full input MAD / spectrum curves along the δ ladder
#
# The sweep writes one file per (plane cell, seed) into WRCircuit/data/circuit_sweep, each holding the
# per-neuron input MAD (lag × Neuron) and PSD (frequency × Neuron) behind the a/b fits above. A file's key
# SET is its plane, so the δ/τ_d files are those varying exactly {delta, tau_d_e, seed}; the δ × Δg_K and
# δ × σ_ee planes live in the same directory and also carry a `delta` key.
# ──────────────────────────────────────────────────────────────────────────────

const sweepdir = projectdir("WRCircuit", "data", "circuit_sweep")

const dtd_sweep_files = let wanted = Set(["delta", "tau_d_e", "seed"]),
        out = Dict{NTuple{3, Float64}, String}()
    for f in readdir(sweepdir; join = true)
        p = try
            parse_savename(f; connector = "&")[2]
        catch
            continue
        end
        Set(keys(p)) == wanted || continue
        out[(p["delta"], p["tau_d_e"], Float64(p["seed"]))] = f
    end
    out
end

# Snap the requested curve parameters onto the swept grid.
const _δ_sweep_grid = sort(unique(k[1] for k in keys(dtd_sweep_files)))
const _τd_sweep_grid = sort(unique(k[2] for k in keys(dtd_sweep_files)))
const δ_curves = [_δ_sweep_grid[_nearest(_δ_sweep_grid, v)] for v in CURVE_DELTAS]
const τd_curve = _τd_sweep_grid[_nearest(_τd_sweep_grid, CURVE_TAU_D)]

"""
    pooled_curve(key, δ, τd)

Median `inputs/mad` or `inputs/psd` curve at one sweep cell: median over neurons within each seed, then
over `CURVE_SEEDS`. Returns `(x, y)` with the unit-stripped lookup (ms for MAD, ms⁻¹ for PSD) as `x`.
"""
function pooled_curve(key, δ, τd)
    files = [
        dtd_sweep_files[k] for k in ((δ, τd, Float64(s)) for s in CURVE_SEEDS)
            if haskey(dtd_sweep_files, k)
    ]
    isempty(files) && error("No $key sweep files at δ = $δ, τ_d_e = $τd")
    curves = map(files) do f
        x = ustripall(jldopen(g -> g[key], f; typemap = toolsarray_typemap))
        dropdims(median(x; dims = 2); dims = 2)
    end
    y = vec(median(reduce(hcat, map(collect, curves)); dims = 2))
    return collect(lookup(first(curves), 1)), y
end

@info "Loading circuit curves at δ = $δ_curves, τ_d_e = $τd_curve ($(length(CURVE_SEEDS)) seeds each)"
const mad_curves = [Base.invokelatest(pooled_curve, "inputs/mad", δ, τd_curve) for δ in δ_curves]
const psd_curves = [Base.invokelatest(pooled_curve, "inputs/psd", δ, τd_curve) for δ in δ_curves]

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
const points_l6 = compute_points(5)    # L6

# Categorical colour per region, ordered low → high hierarchy (`structures` is
# already in ascending-score order). Scatter points and the colorbar draw from
# this same discrete gradient so they stay consistent.
const region_colors = cgrad(binarysunset, length(structures); categorical = true)
const structure_color = Dict(s => region_colors[i] for (i, s) in enumerate(structures))
const HEATMAP = sunrise

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

function plot_hero!(ax, points; arrow_offset = (0.0, 0.0), axis_ranges = (1.0, 1.0))
    xs = [p.a for p in points]
    ys = [p.b for p in points]

    vlines!(ax, [0.5]; color = :gray, linestyle = :dash, linewidth = 1)

    # Mean-direction arrows, all sharing one origin fixed at ARROW_ORIGIN in (a, b)
    # data coordinates (plus an optional per-panel `arrow_offset` in data units).
    # Two read the circuit forward map (δ, τ_d), two read the bFNS theory map (α, β).
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

    # All four arrows share the origin: circuit knobs (δ, τ_d) in green/orange,
    # bFNS orders in blue/red. (vector, label, colour).
    arrows = SHOW_CIRCUIT_ARROWS ? (
            (scaled(δ_dir), "δ", qinghai),
            (scaled(τd_dir), "τ_d", seohae),
            (scaled(α_dir), "α", baikal),
            (scaled(β_dir), "β", bermejo),
        ) : (
            (scaled(α_dir), "α", baikal),
            (scaled(β_dir), "β", bermejo),
        )
    for (v, label, color) in arrows
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
# Figure: (a, b) scatters on top, then one heatmap row per circuit plane below
# ──────────────────────────────────────────────────────────────────────────────

f = SixPanel()   # (a, b) scatters (top), MAD/spectrum curves vs δ (middle), δ/τ_d heatmaps (bottom)

# One sub-grid per panel; each holds [axis | colorbar] in its own columns. Row-major:
# gs[1,2] = heroes (L2/3, L6); gs[3,4] = curves along the δ ladder; gs[5,6] = δ/τ_d plane (a, b).
gs = subdivide(f, 3, 2)

begin # * Top row — (a, b) plane at L2/3 (left) and L6 (right)
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

    ax_l6 = Axis(
        gs[2][1, 1]; xlabel = "Diffusion exponent  a",
        ylabel = "Spectral exponent  b",
        title = "$stim, $(layer_names[5])", limits = scatterlimits
    )
    plot_hero!(ax_l6, points_l6; axis_ranges = aranges)

    # Share axis limits so the L2/3 ↔ L6 comparison is visually fair.
    linkaxes!(ax_l23, ax_l6)
end

begin # * Middle row — the full input MAD (left) and spectrum (right) as δ decreases from the working point
    # Colour by ASCENDING δ rank, not by the order CURVE_DELTAS happens to list, so the colorbar below reads
    # low → high bottom-to-top however the config is written.
    curve_rank = invperm(sortperm(δ_curves))
    curve_colors = cgrad(pelagic, length(δ_curves); categorical = true)
    curve_widths = [δ ≈ δ_0 ? 3 : 1.75 for δ in δ_curves]   # thicken the working-point curve, if sampled

    ax_mad = Axis(
        gs[3][1, 1]; xlabel = "Lag (ms)", ylabel = "MAD (arb. units)",
        title = "Circuit:  input MAD", xscale = log10, yscale = log10
    )
    for (i, (t, y)) in enumerate(mad_curves)
        lines!(ax_mad, t, y; color = curve_colors[curve_rank[i]], linewidth = curve_widths[i])
    end

    ax_psd = Axis(
        gs[4][1, 1]; xlabel = "Frequency (Hz)", ylabel = "PSD (arb. units)",
        title = "Circuit:  input spectrum", xscale = log10, yscale = log10
    )
    for (i, (fr, y)) in enumerate(psd_curves)
        lines!(ax_psd, 1.0e3 .* fr, y; color = curve_colors[curve_rank[i]], linewidth = curve_widths[i])  # 𝑓 is in ms⁻¹
    end

    # δ legend: one band per sampled value, ascending. Also keeps these axes the same pixel width as the
    # colorbar-bearing panels above and below.
    δ_sorted = sort(δ_curves)
    for j in (3, 4)
        Colorbar(
            gs[j][1, 2]; colormap = curve_colors, limits = (0, length(δ_sorted)),
            ticks = ((1:length(δ_sorted)) .- 0.5, string.(δ_sorted)),
            label = "δ  (I:E ratio)", width = 12
        )
    end
end

"Draw one circuit forward-map heatmap (axis + colorbar) into sub-grid `pos`."
function circuit_heatmap!(pos, x, y, z; xlabel, ylabel, title, clabel, colorrange = CairoMakie.Makie.automatic, highclip = CairoMakie.Makie.automatic, lowclip = CairoMakie.Makie.automatic)
    ax = Axis(pos[1, 1]; xlabel = xlabel, ylabel = ylabel, title = title)
    p = heatmap!(ax, x, y, z; colormap = HEATMAP, colorrange, highclip, lowclip)
    Colorbar(pos[1, 2], p; label = clabel, width = 12)
    return ax
end

begin # * δ/τ_d plane heatmaps — the joint (δ × τ_d_e) plane, a (left) and b (right)
    δlab = "δ  (I:E ratio)"
    τdlab = "τ_d_e  (E decay, ms)"
    ax_dtd_a = circuit_heatmap!(
        gs[5], δ_dtd, τd_dtd, A_dtd;
        xlabel = δlab, ylabel = τdlab, title = "Circuit:  a", clabel = "a"
    )
    ax_dtd_b = circuit_heatmap!(
        gs[6], δ_dtd, τd_dtd, B_dtd;
        xlabel = δlab, ylabel = τdlab, title = "Circuit:  b", clabel = "b"
    )
    # Overlay the L2/3 hierarchy path: each area mapped to the (δ, τ_d) cell whose circuit (a, b) is closest
    # to its measured (a, b) (data now uses the same MAPPLE estimator as the circuit), distance normalised
    # by the data spans. Areas are joined in hierarchy order (low -> high); fill = area colour.
    SHOW_DTD_REGIONS && let aspan = scatterlimits[1][2] - scatterlimits[1][1], bspan = scatterlimits[2][2] - scatterlimits[2][1]
        closest_dt = function (a_d, b_d)
            best = (1, 1); bd = Inf
            for i in δ_dtd_keep, j in τd_dtd_keep # restrict to DELTA/TAU_D min-max ranges
                ac = _A_dtd_full[i, j]; bc = _B_dtd_full[i, j]
                (isnan(ac) || isnan(bc)) && continue
                dd = ((ac - a_d) / aspan)^2 + ((bc - b_d) / bspan)^2
                dd < bd && (bd = dd; best = (i, j))
            end
            return (δ_dtd_lookup[best[1]], τd_dtd_lookup[best[2]])
        end
        mapped = [closest_dt(p.a, p.b) for p in points_l23]
        δpath = first.(mapped); τpath = last.(mapped)
        cols = [structure_color[p.structure] for p in points_l23]
        for ax in (ax_dtd_a, ax_dtd_b)
            lines!(ax, δpath, τpath; color = (:black, 0.55), linewidth = 2)
            scatter!(
                ax, δpath, τpath; color = cols, marker = :circle,
                markersize = 15, strokecolor = :black, strokewidth = 1
            )
        end
    end
    # Default working point (δ_0, τd_0) --- the circuit's nominal operating regime, for reference.
    # for ax in (ax_dtd_a, ax_dtd_b)
    #     scatter!(
    #         ax, [δ_0], [τd_0]; color = :white, marker = :star5,
    #         markersize = 22, strokecolor = :black, strokewidth = 1.5
    #     )
    # end
end

# A matching categorical colorbar on each top panel keeps the two axes the same
# pixel width (fair L2/3 ↔ L6 comparison) and doubles as the region legend: one
# band per structure, ordered low → high hierarchy, with "Higher"/"Lower" ends.
const _nreg = length(structures)
for j in (1, 2)
    Colorbar(
        gs[j][1, 2]; colormap = region_colors, limits = (0, _nreg),
        ticks = ((1:_nreg) .- 0.5, structures), width = 12
    )
    Label(gs[j][1, 2, Top()], "Higher"; fontsize = 12, padding = (0, 0, -6, 0))
    Label(gs[j][1, 2, Bottom()], "Lower"; fontsize = 12, padding = (0, 0, 0, 4))
end

addlabels!(f)
display(f)
outfile = joinpath(outdir, "hierarchical_variation.pdf")
wsave(outfile, f)
wsave(joinpath(outdir, "hierarchical_variation.png"), f)   # raster preview
@info "Saved $outfile"
