#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WorkingRegime"
using JLD2
using CairoMakie
using Fathom
using TimeseriesTools
using TimeseriesBase
using Optim
using ForwardDiff # MAPPLE's `fit!` lives in TimeseriesTools' OptimExt; without both it silently degrades
using Statistics
using Random
using LinearAlgebra
using DelimitedFiles
using Printf
import ImageMagick # rasterises brain.pdf in pdfpanel!

set_theme!(Fathom.fathom())
include(joinpath(@__DIR__, "mathlabels.jl")) # mit/mrm/unitlabel: symbols in the equation face

const NAME = "Fig1_combined_curves"
const outdir = plotsdir(NAME)
const STIM = "spontaneous"
const experiment_color = :cornflowerblue
# (d)'s title is a `Label`, not an `Axis` title; match whatever the theme sets for titles.
const TITLESIZE = to_value(Makie.current_default_theme()[:Axis][:titlesize])
# Where the blue probe of `brain.pdf` meets the cortical surface, in the coordinates the panel
# plots. Found by isolating strongly saturated blue pixels (HSV hue 200-250, s > 0.45, v > 0.35)
# and walking down to the first row adjacent to the TEAL slabs, which are the visual areas: (row
# 227, col 448) of a 1375x1767 image. Testing against the brain silhouette instead lands ~90 rows
# too high, because the probes are bundled closely enough up there to shade each other.
# `pdfpanel!` draws `rotr90(img)` mirrored in x, which sends (r, c) to (W + 1 - c, H + 1 - r).
const BRAIN_PROBE = (1767 + 1 - 448, 1375 + 1 - 227)
const circuit_color = :crimson

mkpath(outdir)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"Bootstrap median + 95% CI. Local copy so this script needs no Bootstrap.jl (as combined_curves.jl does)."
function percentilebootmedian(x; N = 10_000, α = 0.05)
    x = filter(!isnan, collect(skipmissing(x)))
    isempty(x) && return (NaN, (NaN, NaN))
    rng = Random.MersenneTwister(42)
    n = length(x)
    meds = [median(x[rand(rng, 1:n, n)]) for _ in 1:N]
    return median(x), Tuple(quantile(meds, (α / 2, 1 - α / 2)))
end

_select(x, sels::Pair...) = getindex(x; (Symbol(n) => At(v) for (n, v) in sels)...)

"""
    pdfpanel!(gp, pdf; dpi = 600)

`pdf` rendered at `dpi` into a decoration-free axis, cached as a png in `data/` and re-rendered
whenever the PDF is newer. Rasterised by ImageMagick; the resolution must be set on the wand
BEFORE reading, since a bare `FileIO.load` renders PDFs at 72 dpi. MakieTeX's `PDFDocument` would
keep the artwork vector, but its newest release (0.4.3) pins Makie 0.21 against this project's
0.24, so it cannot resolve here; the brain render is embedded raster anyway, so nothing is lost.
"""
function pdfpanel!(gp, pdf; dpi = 600, flipx = false, kwargs...)
    png = datadir(first(splitext(basename(pdf))) * ".png")
    if !isfile(png) || mtime(png) < mtime(pdf)
        wand = ImageMagick.MagickWand()
        ccall(
            (:MagickSetResolution, ImageMagick.libwand), Cint,
            (Ptr{Cvoid}, Cdouble, Cdouble), wand, dpi, dpi
        )
        ImageMagick.readimage(wand, pdf)
        ImageMagick.writeimage(wand, png)
    end
    ax = Axis(gp; aspect = DataAspect(), kwargs...)
    hidedecorations!(ax)   # leaves the title, which is not a decoration
    hidespines!(ax)
    # The PDF rasterises onto an opaque white page; clear that background, or the panel reads as
    # a white slab inside the tinted group box. The cut is clean: 27% of pixels are pure white
    # and almost nothing falls between 0.98 and 1, so the mesh survives untouched.
    img = map(wload(png)) do c
        min(c.r, c.g, c.b) >= 0.999 ? RGBAf(0, 0, 0, 0) : RGBAf(c.r, c.g, c.b, c.alpha)
    end
    m = rotr90(img)
    image!(ax, flipx ? m[end:-1:1, :] : m)
    return ax
end

"Fathom-style filled histogram with a step outline, from a precomputed density (bin centre -> pdf)."
function zigg!(ax, h; color = baikal, logy = false, dropfirst = logy)
    centers = collect(lookup(h, 1))
    pdf = collect(h)
    w = centers[2] - centers[1]
    edges = [centers .- w / 2; centers[end] + w / 2]
    e = dropfirst ? edges[2:end] : edges
    c = dropfirst ? centers[2:end] : centers
    p = dropfirst ? pdf[2:end] : pdf
    if dropfirst                     # renormalise over the shown bins
        s = sum(p) * w
        s > 0 && (p = p ./ s)
    end
    barplot!(ax, c, p; width = w, gap = 0, color = (color, 0.5), strokewidth = 0)
    ys = Float64.([p; last(p)])
    logy && (ys[ys .<= 0] .= NaN)
    return stairs!(ax, e, ys; step = :post, color)
end

"""
    track_com(field)

Centre of mass of a (time × x × y) field on a torus, by circular mean, so the track does not jump
when the bump crosses the periodic boundary. Copied from `plot_demo_run.jl`.
"""
function track_com(field)
    nt, nx, ny = size(field)
    com_x, com_y = zeros(nt), zeros(nt)
    θx = 2π .* (0:(nx - 1))' ./ nx
    θy = 2π .* (0:(ny - 1)) ./ ny
    for t in 1:nt
        w = abs.(field[t, :, :])
        tw = sum(w)
        if tw < 1.0e-10
            com_x[t], com_y[t] = nx / 2, ny / 2
            continue
        end
        ξx, ζx = sum(w .* cos.(θx)) / tw, sum(w .* sin.(θx)) / tw
        ξy, ζy = sum(w .* cos.(θy)) / tw, sum(w .* sin.(θy)) / tw
        com_x[t] = mod(atan(ζx, ξx) * nx / (2π), nx) + 1
        com_y[t] = mod(atan(ζy, ξy) * ny / (2π), ny) + 1
    end
    return com_x, com_y
end

gaussian(x) = exp(-x^2 / 2) / sqrt(2π)
"""
    logshift(y, ref)

`y` translated in log space so that `ref`'s minimum sits at 1. A pure translation: every log-log
slope survives it, so curves from different sources overlay in one frame and can be compared by eye.

This replaces a min-max normalisation, which divided each curve's log values by that curve's OWN log
range and so scaled its slopes by a different factor per curve. Under it the two MAD fits, both
a = 0.61, drew at 1.11 and 0.83 (log ranges 0.55 and 0.74), and neither drew at 0.61. Passing a bare
power law `f^b` through it was worse still: the divisor is then `|b| * log10(fmax/fmin)`, `b`
cancels, and every guide draws at `-1/log10(fmax/fmin)` whatever its exponent.

Pass the same anchor for a curve and its error band, or the band shifts relative to the curve.
"""
logshift(y, at::Real) = exp10.(log10.(y) .- at)
logshift(y, ref::AbstractVector) = logshift(y, minimum(log10.(ref)))   # anchor on `ref`'s minimum

"`log10` of the curve `(x, y)` at `x0`, as a `logshift` anchor: shifts the curve through 1 at `x0`."
logat(x, y, x0) = log10(y[argmin(abs.(log10.(x) .- log10(x0)))])

"""
    slopeguide!(ax, fx, fy, band, b, lift; kwargs...)

A straight log-log guide of slope exactly `b`, spanning `band`, anchored to the drawn curve
`(fx, fy)` at the band's LOWER edge and offset by the factor `lift` (`< 1` to sit below the curve).

The anchor is the low edge rather than the band centre because these spectra are shallower over
their fitted band than the fitted exponent itself (the experiment averages -1.46 over 3-500 Hz
against a quoted -1.75, the peaks and the low arm pulling it up). A guide pinned at the centre
therefore rises above the curve at low frequency; pinned at the low edge it stays below throughout.
"""
function slopeguide!(ax, fx, fy, band, b, lift; kwargs...)
    x0 = first(band)
    y0 = fy[argmin(abs.(log10.(collect(fx)) .- log10(x0)))]
    x = exp10.(range(log10(first(band)), log10(last(band)); length = 64))
    return lines!(ax, x, lift .* y0 .* (x ./ x0) .^ b; kwargs...)
end

# MAD exponents are re-fit here from the DISPLAYED (median) curves, mirroring each side's
# calculation script (`diffusion_line`/`fit_mad`: 1 component, 0 peaks, log-weighted over the shared
# 0-8 ms band). The PSD exponents are NOT re-fit: they are read from the calculation outputs, whose
# fits hold both MAPPLE knots (`WRExperiment.mapple_fit`). A free-knot refit is not identifiable on
# these curves --- it returns a rising low-frequency arm against an over-steep high-frequency one.

"1-component, 0-peak MAPPLE fit of a MAD curve over `band` (seconds)."
function mad_mapple(t, vals; band = 0.0 .. 8.0e-3) # WRExperiment.MAD_BAND == WRCircuit.MAD_BAND_MS/1000
    y = ToolsArray(vals, (𝑡(t),))[𝑡 = band]
    m = fit(MAPPLE, y; components = 1, peaks = 0)
    fit!(m, y; w = true)
    return m
end

# The variability exponent of a (median) Fano curve, `t` in ms. The convention shared by the
# circuit's `circuit.fano.exponent` and Fig 4's per-session exponents; the drawn experiment median
# is refit here exactly as the MAD panel refits its displayed curves.


# ──────────────────────────────────────────────────────────────────────────────
# Data (everything is read; nothing is computed here)
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading experiment data"
traces = jldopen(f -> Dict(k => f[k] for k in keys(f)), projectdir("WRExperiment", "data", "traces.jld2"))
incr = jldopen(f -> Dict(k => f[k] for k in keys(f)), projectdir("WRExperiment", "data", "increment_histograms.jld2"))
plot_data = jldopen(
    f -> Dict(k => f[k] for k in keys(f)),
    projectdir("WRExperiment", "data", "WRExperiment.jld2"); typemap = toolsarray_typemap
)

@info "Loading circuit data"
circuit = loadtoolsarray(projectdir("WRCircuit", "data", "circuit_curves.jld2"), "circuit_curves")
cstats = jldopen(
    f -> Dict(k => f[k] for k in keys(f)),
    projectdir("WRCircuit", "data", "demo_run_stats.jld2"); typemap = toolsarray_typemap
)
craw = jldopen(
    projectdir("WRCircuit", "data", "demo_run.jld2"); typemap = toolsarray_typemap
) do f                                     # 2.2 GB on disk: take only what the panels draw
    Dict(
        "fixed_params" => f["fixed_params"], "N" => f["N"],
        "mean_V" => f["mean_V"], "nu" => f["nu"],
        "E_V" => f["E_V"], "E_input" => f["E_input"],
        "E_spike" => f["E_spike"], "epositions" => f["epositions"]
    )
end

const N_GRID = craw["N"]
const DX = craw["fixed_params"].dx
# Raw traces carry a time axis in milliseconds; panels convert to seconds where they draw it.
V = craw["E_V"]
INPUT = craw["E_input"]
const MS_TO_S = 1 / 1000
const TIME_MS = ustripall(collect(times(V)))    # the shared time base, in milliseconds
const DT_MS = TIME_MS[2] - TIME_MS[1]           # ~0.1 ms; sample counts are NOT milliseconds
# Tick-label widths reserved on the stacked axes of (e) and on (f)/(g), so their ylabels line up
# rather than tracking each axis' own tick width.
const YSPACE = 26.0
const YSPACE_D = 30.0

# The field window drawn in (d), chosen by searching the whole saved trace for a stretch that (i)
# never crosses the periodic boundary and stays clear of the edges, and (ii) whose central neuron's
# input current shows the characteristic large excursions without dominating the trace: four crossings
# of 1.5 nA peaking at 2.6 nA (a burstier window reached 4.2 nA and clipped the axis), with a path
# length of 290 grid units. Computed here rather than inside the panel because the patch, the traced
# neuron and the trace window are all defined from it.
#
# Re-picked by sweeping every window in the trace against both criteria at once, scoring the bump
# against the patch centre each window implies (the patch follows the trajectory end, so a fixed
# centre scores candidates wrongly). Only 21 windows are seam-free with >= 8 grid units of edge
# clearance and a wide sweep; this one has the strongest bump among them. Against the stretch the
# original search returned, the track is longer (extent 66 against 52 grid units, path 422 against
# 289) and the bump is no longer invisible: mean input inside the patch 1.28 nA against 0.04
# outside, where before it was 0.33 against 0.10. Landing on the field's rare flares instead gives
# a far brighter bump but a short, tangled track --- the two maxima are anticorrelated.
const FIELD_TINDEX = 41_921
const FIELD_WINDOW = 1170
const FIELD_STRIDE = 3
const FIELD_FRAME, TRAJECTORY = let
    grid = reshape(parent(ustripall(INPUT)), (size(INPUT, 1), N_GRID, N_GRID))
    win = grid[(FIELD_TINDEX - FIELD_WINDOW):FIELD_TINDEX, :, :]
    cx, cy = track_com(win)
    (
        win[end, :, :],
        (;
            xs = DX .* cx[1:FIELD_STRIDE:end] ./ N_GRID,
            ys = DX .* cy[1:FIELD_STRIDE:end] ./ N_GRID,
            # Sample offsets are converted with the real sampling period; dividing the sample count
            # by 1000 (as plot_demo_run.jl does) overstates the elapsed time tenfold at dt = 0.1 ms.
            t = collect(0:FIELD_WINDOW)[1:FIELD_STRIDE:end] .* DT_MS .* MS_TO_S,
        ),
    )
end

# The patch is centred where the trajectory ENDS, so (d)'s circle marks where the activity arrives
# and (e)'s raster is the population there. The traced neuron is then just the cell nearest that
# centre; it does not have to sit exactly on it.
const PATCH_RADIUS = 0.1
const PATCH_ORIGIN = [TRAJECTORY.xs[end], TRAJECTORY.ys[end]]
"Distance from `pos` to the patch centre, respecting the periodic (torus) boundary."
patchdistance(pos) = (dp = abs.(pos .- PATCH_ORIGIN); norm(min.(dp, DX .- dp)))
const PATCH_DISTANCES = patchdistance.(craw["epositions"])
const TRACED_NEURON = argmin(PATCH_DISTANCES)

# ──────────────────────────────────────────────────────────────────────────────
# Panel functions
# ──────────────────────────────────────────────────────────────────────────────

"LFP trace above its spike raster, sharing a time axis (Fig 1b)."
function traces_panel!(gl, d)
    axl = Axis(
        # Negative `ylabelpadding`: these axes have no tick labels, so the default padding leaves
        # the ylabel floating well left of the spine, right where the (a) bracket lines arrive.
        gl[1, 1]; ylabel = "LFP", ylabelpadding = -8, yticks = ([], []), xticks = ([], []),
        xgridvisible = false, limits = ((0, maximum(d["t"])), nothing),
        title = "Neuropixels recordings"
    )
    lines!(axl, d["t"], d["lfp"]; color = experiment_color, linewidth = 1.2)

    axr = Axis(
        gl[2, 1]; ylabel = "Neuron", ylabelpadding = -8, xlabel = unitlabel("Time", "s"),
        yticks = ([], []),
        xgridvisible = false, limits = ((0, maximum(d["t"])), nothing)
    )
    for (i, s) in enumerate(d["spikes"])
        isempty(s) && continue
        scatter!(axr, s, fill(i, length(s)); color = :black, markersize = 2)
    end
    rowgap!(gl, 1, 0.0)
    rowsize!(gl, 1, Relative(0.42))
    return axl, axr
end

"""
Excess kurtosis annotated on the panel: the median across sessions of each session's median over its
own L2/3 channels --- the two-stage aggregation `collect_surrogates.jl` uses, so the number printed
here is the one the main text quotes. Falls back to the flat per-channel median for caches written
before `kurtosis_session` existed; that pooled form weights sessions by channel count and reads
higher, which is what made the panel and the text disagree.
"""
increment_kurtosis(d) = median(haskey(d, "kurtosis_session") ? d["kurtosis_session"] : d["kurtosis"])

"""
    increment_panel!(ax, d; showsurrogate = true)

Pooled L2/3 increment density on a log axis against its Gaussian null. A Gaussian is an inverted
parabola in these axes, so heavier tails read directly as the data lifting off the dashed curve. The
dashed curve IS the null quoted in the text: FT surrogates preserve the spectrum but leave Gaussian
increments, and the thin surrogate line shows that empirically.
"""
function increment_panel!(ax, d; showsurrogate = true, xlim = 7)
    x, y, ys = d["centres"], d["density"], d["density_surrogate"]
    if showsurrogate
        ks = (ys .> 0) .& (abs.(x) .<= xlim)
        lines!(ax, x[ks], ys[ks]; color = (abyad, 0.9), label = "Surrogate")
    end
    xg = range(-xlim, xlim, length = 400)
    lines!(ax, xg, gaussian.(xg); color = chernoe, linestyle = :dash, label = "Gaussian")
    keep = (y .> 0) .& (abs.(x) .<= xlim)
    lines!(ax, x[keep], y[keep]; color = experiment_color, label = "Data")
    text!(
        ax, 0.96, 0.92; text = rich(mit("κ"), @sprintf(" = %.2f", increment_kurtosis(d))), space = :relative,
        align = (:right, :top), fontsize = 11, color = experiment_color
    )
    # Compact, and the top limit lifted from 1.5: at the default size the first row sits on the
    # peak, and the corners above a peaked density are the only free space in this panel.
    axislegend(
        ax; position = :lt, framevisible = false, labelsize = 11,
        patchsize = (13, 8), rowgap = -2, padding = (2, 2, 0, 0)
    )
    ylims!(ax, 1.0e-6, 12)
    xlims!(ax, -xlim, xlim)
    return ax
end

"Input field snapshot with the centre-of-mass trajectory over the preceding window (Fig 1e)."
# `tindex` ends a window chosen by searching the whole saved trace for the stretch that never crosses
# the periodic boundary while still travelling far (path length 598 grid units, zero seam crossings).
# With no crossings the cosmetic re-centring `shift` is unnecessary, so it is off.
function circuit_field_panel!(gl)
    frame = FIELD_FRAME
    xs, ys, colour = TRAJECTORY.xs, TRAJECTORY.ys, TRAJECTORY.t
    xx = range(0, DX, length = N_GRID)
    # Insurance for any window that does wrap: blank the wrapping point so the line breaks there
    # rather than being drawn straight across the field.
    wrapped = [false; (abs.(diff(xs)) .> DX / 2) .| (abs.(diff(ys)) .> DX / 2)]
    xs = [wrapped[i] ? NaN : xs[i] for i in eachindex(xs)]
    ys = [wrapped[i] ? NaN : ys[i] for i in eachindex(ys)]

    ax = Axis(
        gl[1, 1]; xlabel = unitlabel(mit("X"), "mm"), ylabel = unitlabel(mit("Y"), "mm"),
        limits = ((0, DX), (0, DX)),
        xticks = 0:0.25:0.5, yticks = 0:0.25:0.5, xtickformat = terseticks,
        ytickformat = terseticks, aspect = 1          # the patch is a disc, so keep the field square
    )
    # Clipped at 0 rather than autoscaled: the negative tail spent a third of the colour range on
    # values the panel is not about, washing out the positive pops.
    h = heatmap!(
        ax, xx, xx, frame'; colormap = seethrough(reverse(sunrise)),
        colorrange = (0, maximum(frame)), rasterize = 10
    )
    lines!(ax, xs, ys; color = :white, linewidth = 2.5)
    p = lines!(ax, xs, ys; color = colour, colormap = reverse(cgrad(:turbo)), linewidth = 1.5)
    θ = range(0, 2π, length = 200)                 # the patch rastered in (e)
    lines!(
        ax, PATCH_ORIGIN[1] .+ PATCH_RADIUS .* cos.(θ),
        PATCH_ORIGIN[2] .+ PATCH_RADIUS .* sin.(θ);
        color = chernoe, linestyle = :dash, linewidth = 1.5
    )
    scatter!(                                       # the neuron traced in (e)
        ax, [craw["epositions"][TRACED_NEURON][1]], [craw["epositions"][TRACED_NEURON][2]];
        color = chernoe, markersize = 7, strokecolor = :white, strokewidth = 1
    )
    Colorbar(gl[1, 2], h; label = unitlabel("Input", "nA"), width = 8) # fills the field's height
    cbt = Colorbar(
        gl[0, 1], p; vertical = false, label = unitlabel("Time", "s"), tickformat = terseticks,
        height = 8,
        labelpadding = 1, ticklabelpad = 2
    )
    # The title cannot be an `Axis` title now (that would land between the colorbar and the field),
    # nor a row of its own: a row above the colorbar pulls the colorbar's own label and tick
    # protrusion inside the block's box, costing the field ~70 pt of height. It goes in the colorbar
    # cell's TOP PROTRUSION, stacked outside the box. The padding is read off the colorbar's own
    # reported protrusion rather than guessed, since both anchor at the cell edge and would
    # otherwise overlap: that is what puts the title above "Time (s)" instead of on it.
    lab = Label(
        gl[0, 1, Top()], "Biophysical circuit"; font = :bold, fontsize = TITLESIZE,
        padding = lift(
            d -> (0.0f0, 0.0f0, d.outer.top + 3.0f0, 0.0f0),
            cbt.layoutobservables.reporteddimensions
        )
    )
    rowgap!(gl, 1, 4.0)
    colgap!(gl, 1, Relative(0.02))
    # The field is aspect-locked and width-bound, so it letterboxes inside a taller cell and the
    # colorbars, which fill that cell, overhang it top and bottom. `Aspect` ties the row's height to
    # column 1's width, making the cell square: the axis then fills it exactly and the bars end
    # flush with the heatmap's own top and bottom edges.
    rowsize!(gl, 1, Aspect(1, 1.0))
    return ax, lab
end

"""
    circuit_trace_panels!(gl; ...)

The membrane potential of the neuron at the centre of the patch, the spike raster of the patch around
it, and that neuron's input current, stacked flush on a shared time axis (mirroring the LFP-over-raster panel of the
experimental block), with the V and |ΔI| densities in a second column.

The stack lives in its own nested layout. Sharing grid rows with the density column would let those
panels' titles and x-labels add protrusions to the same rows, which forces the traces apart no matter
what `rowgap!` is set to; nesting makes the stack's row heights independent so the rows sit flush.
"""
function circuit_trace_panels!(gl; ts = (FIELD_TINDEX - 5000):FIELD_TINDEX, stride = 4)
    neuron = TRACED_NEURON
    gtr = gl[1, 1] = GridLayout()          # the flush three-row stack
    gd = gl[1, 2] = GridLayout()           # the two densities
    tsec = (TIME_MS[ts] .- TIME_MS[first(ts)]) .* MS_TO_S
    xlim = (0, last(tsec))

    axv = Axis(
        gtr[1, 1]; title = "Circuit recordings", ylabel = unitlabel(mit("V"), "mV"),
        yticklabelspace = YSPACE,
        yticks = WilkinsonTicks(3; k_max = 3), xticks = ([], []), xgridvisible = false,
        limits = (xlim, nothing)
    )
    hlines!(axv, [-50]; color = bermejo)
    hlines!(axv, [-70]; color = bermejo, linestyle = :dash)
    hlines!(axv, [craw["mean_V"]]; color = :gray, linestyle = :dash)
    lines!(axv, tsec, collect(ustripall(V[ts, neuron])); linewidth = 1.2, color = experiment_color)
    # Glowed: it sits over the trace, which spikes through it. Fathom's smaller default label size
    # (labelsize 1.1x rather than 1.25x the base) gave this axis more room and let the trace reach
    # further down into the corner the annotation occupies.
    text!(
        axv, 0.97, 0.06; text = rich(mit("ν"), @sprintf(" ≈ %.1f Hz", craw["nu"])), space = :relative,
        align = (:right, :bottom), fontsize = 16, glowcolor = :white, glowwidth = 12
    )

    # Every `stride`-th excitatory neuron across the whole patch: taking a contiguous block instead
    # would sample one strip of the disc, since neuron index maps to grid position. At the 0.1 mm
    # patch radius the disc holds ~640 neurons, so a stride of 4 fills the band without crowding it.
    local_idx = findall(<(PATCH_RADIUS), PATCH_DISTANCES)[1:stride:end]
    axr = Axis(
        # One blank tick rather than none: with an empty tick-label list Makie collapses the
        # tick-label space to zero, so `yticklabelspace` has nothing to widen and the ylabel creeps
        # inwards away from "V (mV)" and "I (nA)".
        gtr[2, 1]; ylabel = "Neuron", yticks = ([1.0], [" "]), yticksvisible = false,
        xticks = ([], []), xgridvisible = false, yticklabelspace = YSPACE,
        limits = (xlim, nothing)
    )
    S = parent(craw["E_spike"])
    for (i, j) in enumerate(local_idx)
        idx = findall(view(S, ts, j))
        isempty(idx) && continue
        scatter!(axr, tsec[idx], fill(i, length(idx)); color = :black, markersize = 2)
    end

    axi = Axis(
        gtr[3, 1]; ylabel = unitlabel(mit("I"), "nA"), xlabel = unitlabel("Time", "s"),
        yticklabelspace = YSPACE,
        yticks = WilkinsonTicks(3; k_max = 3), limits = (xlim, (-1, 3))
    )
    lines!(axi, tsec, collect(ustripall(INPUT[ts, neuron])); linewidth = 1.2, color = experiment_color)

    axvd = Axis(
        gd[1, 1]; title = "Potential", ylabel = "Density", xlabel = unitlabel(mit("V"), "mV"),
        xticks = WilkinsonTicks(3; k_max = 3),
        yticks = WilkinsonTicks(3; k_max = 3), yticklabelspace = YSPACE_D
    )
    zigg!(axvd, cstats["V_hist"]; dropfirst = true)   # first bin is the Vr reset pile-up
    vlines!(axvd, [craw["mean_V"]]; color = :gray, linestyle = :dash)

    axid = Axis(
        gd[2, 1]; title = "Step sizes", ylabel = "Density", yticklabelspace = YSPACE_D,
        xlabel = unitlabel(mit("|ΔI|"), "nA"), xscale = log10, yscale = log10,
        xticks = LogTicks(WilkinsonTicks(3; k_max = 3)),
        yticks = LogTicks(WilkinsonTicks(3; k_max = 3))
    )
    zigg!(axid, cstats["dI_hist"]; logy = true)

    rowgap!(gd, 1, 6.0)                    # (f) and (g) sit closer together
    rowgap!(gtr, 1, 0.0)
    rowgap!(gtr, 2, 0.0)
    rowsize!(gtr, 2, Relative(0.28))
    colsize!(gl, 1, Relative(0.62))
    return (; gtr, gd, axv, axr, axi, axvd, axid)
end

function combined_curves_panels!(gl)
    mad = plot_data["mad_curves"][STIM]
    psd = plot_data["spectral_curves"][STIM]["VISp"]
    fano = plot_data["fano_curves"][STIM]

    # * MAD. Both curves are translated in log space (`logshift`, not min-max normalised), so the
    #   drawn slopes ARE the exponents and the two curves are directly comparable by eye. Each
    #   dashed line is a MAPPLE fit computed HERE on the drawn median curve, over the 0-8 ms band;
    #   the curve steepens then rolls into its plateau, so that band is steeper than the rise as a
    #   whole and the guide is correspondingly steeper than the curve away from it.
    # Exponent labels are placed in RELATIVE space: (h) and (i) have wildly different y ranges
    # (0.9 decades against 7.2), so a fixed gap in data units reads as a different gap on the page,
    # and every change to a limit used to move them off their marks. One gap, both panels.
    LABEL_X, LABEL_GAP = 0.02, 0.155

    mfit_exp = mad_mapple(mad.t_all, mad.mu)
    mfit_circ = mad_mapple(circuit.mad.t, circuit.mad.mu)
    axm = Axis(
        gl[1, 1]; ylabel = "MAD (arb. units)", xlabel = unitlabel("Time lag", "s"),
        title = "Diffusion",
        xscale = log10, yscale = log10,
        limits = ((10^(-3.4), 10^0.1), nothing)
    )
    band!(
        axm, mad.t_all, logshift(mad.σl, mad.mu), logshift(mad.σh, mad.mu);
        color = experiment_color, alpha = 0.5
    )
    lines!(
        axm, mad.t_all, logshift(mad.mu, mad.mu);
        color = experiment_color, label = "Experiment\n(LFP)"
    )
    idxs = mad.fit_t .< 0.005
    lines!(
        axm, mad.fit_t[idxs] ./ 2, logshift(predict(mfit_exp, mad.fit_t), mad.mu)[idxs];
        linestyle = :dash, color = experiment_color
    )
    lines!(
        axm, circuit.mad.t, logshift(circuit.mad.mu, circuit.mad.mu);
        color = circuit_color, label = "Circuit\n(input)"
    )
    idxs = circuit.mad.fit_t .< 0.005
    lines!(
        axm, circuit.mad.fit_t[idxs] .* 2,
        logshift(predict(mfit_circ, circuit.mad.fit_t), circuit.mad.mu)[idxs];
        color = circuit_color, linestyle = :dash
    )
    text!(
        axm, LABEL_X, 0.96; text = rich(mit("a"), " = $(round(only(betas(mfit_exp)), sigdigits = 2))"),
        space = :relative, color = experiment_color, align = (:left, :top)
    )
    text!(
        axm, LABEL_X, 0.96 - LABEL_GAP; text = rich(mit("a"), " = $(round(only(betas(mfit_circ)), sigdigits = 2))"),
        space = :relative, color = circuit_color, align = (:left, :top)
    )
    axislegend(axm; position = :rb, framevisible = false)

    # * PSD. The dashed lines are the APERIODIC component of each fit, a straight line in log-log
    #   whose slope is exactly the quoted exponent; only the slope carries information. The
    #   exponents come from the calculation pipelines (`WRExperiment.mapple_fit`, which holds both
    #   MAPPLE knots, and demo_run.jl's `fit_spectrum`) rather than being refit here: an
    #   unconstrained refit is not identifiable on these curves and returns a rising low-frequency
    #   arm paired with an over-steep high-frequency one (+0.91/-4.08 experiment, +2.6/-5.92
    #   circuit, against true band slopes of about -0.9 and -2.5).
    axp = Axis(
        gl[1, 2]; xlabel = unitlabel("Frequency", "Hz"), ylabel = "PSD (arb. units)",
        title = "Power spectrum",
        xscale = log10, yscale = log10,
        xticks = [3, 10, 30, 100], limits = ((2, 500), nothing)
    )
    # Both spectra are shifted through 1 at `F0`, so they overlay where the comparison is made and
    # the eye reads the difference in slope rather than a difference in offset. The guides span
    # `GUIDE` only: a straight line of slope -1.75 across the full 2-500 Hz axis would run four
    # decades and say nothing about the band the exponent describes.
    F0 = 30
    # Decades the circuit spectrum is dropped. It has to clear not just the experiment's curve but
    # the experiment's guide, which dives below that curve: at 100 Hz the guide sits 1.4 decades
    # under the experiment, so anything less than ~2 puts the circuit curve on top of it.
    CIRCUIT_DROP = 2.2
    # The circuit's band is `fit_f` itself (11-1000 Hz), clipped to the axis. The experiment's `β`
    # is the SECOND MAPPLE component, whose knots `WRExperiment.mapple_fit` fixes at PSD_KNEE = 3 Hz
    # and the top of PSD_RANGE = 500 Hz --- but its guide is drawn only from the circuit band's
    # lower edge, so both segments span the same frequencies and the only difference a reader sees
    # between them is a difference in slope.
    CIRC_BAND = (max(minimum(circuit.psd.fit_f), 2), min(maximum(circuit.psd.fit_f), 500))
    EXP_BAND = (first(CIRC_BAND), 500.0)
    GUIDE_LIFT = 0.45
    ey = logshift(psd.μ, logat(psd.f, psd.μ, F0))
    lines!(axp, psd.f, ey; color = (experiment_color, 0.8))
    band!(
        axp, psd.f, logshift(psd.σl, logat(psd.f, psd.μ, F0)),
        logshift(psd.σh, logat(psd.f, psd.μ, F0)); color = (experiment_color, 0.32)
    )
    slopeguide!(
        axp, psd.f, ey, EXP_BAND, psd.spectral_exponent_median, GUIDE_LIFT;
        color = experiment_color, linestyle = :dash
    )
    cy = logshift(circuit.psd.mu, logat(circuit.psd.f, circuit.psd.mu, F0) + CIRCUIT_DROP)
    lines!(axp, circuit.psd.f, cy; color = circuit_color)
    slopeguide!(
        axp, circuit.psd.f, cy, CIRC_BAND, circuit.psd.exponent, GUIDE_LIFT;
        color = circuit_color, linestyle = :dash
    )
    text!(
        axp, LABEL_X, 0.43; text = rich(mit("b"), " = $(round(psd.spectral_exponent_median; sigdigits = 3))"),
        space = :relative, color = experiment_color, align = (:left, :top)
    )
    text!(
        axp, LABEL_X, 0.43 - LABEL_GAP; text = rich(mit("b"), " = $(round(circuit.psd.exponent; sigdigits = 3))"),
        space = :relative, color = circuit_color, align = (:left, :top)
    )

    # * Fano factor, unnormalised (both are dimensionless counts).
    axf = Axis(
        gl[1, 3]; xlabel = unitlabel("Time lag", "s"), ylabel = "Fano factor",
        title = "Fano factor",
        xscale = log10, yscale = log10
    )
    band!(axf, 0.001 .* fano.t_all, fano.sl, fano.su; color = experiment_color, alpha = 0.3)
    lines!(axf, 0.001 .* fano.t_all, fano.mu; color = experiment_color)
    # The unified fit of the DRAWN experiment median, cached by collect_calculations.jl (the stored
    # `fano.mslope` is the legacy fixed-band OLS); the dashed guide spans the fitted scaling band,
    # slope-β through its centre. Read rather than refit, so this figure and the manuscript cannot
    # disagree about a multistart-seeded quantity.
    ffit_exp = fano.cfit
    let tt = exp10.(range(log10(ffit_exp.lo), log10(min(ffit_exp.hi, 1.0e3)); length = 20)),
            t0 = exp10((log10(ffit_exp.lo) + log10(min(ffit_exp.hi, 1.0e3))) / 2)

        y0 = fano.mu[argmin(abs.(fano.t_all .- t0))]
        lines!(
            axf, 0.001 .* tt ./ 2, y0 .* (tt ./ t0) .^ ffit_exp.β;
            color = experiment_color, linestyle = :dash, linewidth = 3
        )
    end
    lines!(axf, 0.001 .* circuit.fano.t, circuit.fano.mu; color = circuit_color)
    idxs = circuit.fano.band[1] .< circuit.fano.t .< circuit.fano.band[2] # the fit's measured band
    lines!(
        axf, 0.001 .* circuit.fano.t[idxs] ./ 2, circuit.fano.mu[idxs];
        color = circuit_color, linestyle = :dash
    )
    text!(
        axf, 0.001 .* 40, 10^0.32; text = rich(mit("c"), " = $(round(ffit_exp.β, digits = 2))"),
        color = experiment_color, align = (:left, :center)
    )
    text!(
        axf, 0.001 .* 1.2, 1.4; text = rich(mit("c"), " = $(round(circuit.fano.exponent, digits = 2))"),
        color = circuit_color, align = (:left, :center)
    )
    return (;
        axm, axp, axf, mfit_exp, mfit_circ,
        b_exp = psd.spectral_exponent_median, b_circ = circuit.psd.exponent,
        c_exp = ffit_exp, c_circ = circuit.fano.exponent,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

begin # * Render
    # Right padding raised from the default 16: (f) and (g) centre their last x tick label on the
    # cell's right edge, so it overhangs by ~12 pt and the group box has to reach past it (-16
    # below). Without the extra padding that box would sit 2 pt from the page edge.
    f = SixPanel(; figure_padding = (16, 26, 16, 16))
    gtop = f[1, 1] = GridLayout()
    gmid = f[2, 1] = GridLayout()
    gbot = f[3, 0:1] = GridLayout(; alignmode = Makie.Mixed(left = 0))

    # Group boxes, as in Fig 2: experiment in blue, circuit in red; the combined curves below
    # belong to both and stay unboxed. Negative `Outside` reaches over the protrusions so the
    # panel letters, titles and axis labels sit inside the box --- no row can hold them, since
    # they ARE the protrusion. Values are measured from the solved layout, not guessed: the rows
    # protrude 60 pt left (to the page margin), 29 pt above and 53 pt below.
    groupbox(gp, c; fillalpha = 0.05, top = -34) = Box(
        gp; color = (c, fillalpha), strokecolor = (c, 0.35), strokewidth = 1.5,
        cornerradius = 8, alignmode = Outside(-64, -16, -56, top)
    )
    # Vertical block labels outside the boxes, in a new column 0 of the FIGURE layout (not of each
    # row's own layout, or the two boxed rows would size that column differently and their left
    # edges would stop agreeing). The summary row spans 0:1 so that it reaches the page margin too,
    # rather than starting right of the label column; `Mixed(left = 0)` is what makes that safe,
    # since it keeps the row's 57 pt ylabel protrusion inside its own cell instead of reserving it
    # to the LEFT of the label column and pushing the labels off the margin. Its axes are therefore
    # narrower than the rows above and no longer align with them --- deliberate. Only the left side
    # is switched: a full `Outside` would pull the titles and xlabels inside the cell too and
    # squash the row's height.
    blocklabel(gp, text) = Label(
        gp, text; rotation = pi / 2, font = :bold, fontsize = 18, tellheight = false
    )
    lab_exp = blocklabel(f[1, 0], "Experiment")
    lab_cir = blocklabel(f[2, 0], "Circuit model")
    box_exp = groupbox(gtop[1, 1:3], baikal)
    # A deeper top reach than the blue box: (d)'s title stacks above the time colorbar's own label
    # and ticks, so this row protrudes ~50 pt further than the other. The same value on the blue box
    # would carry it off the top of the page.
    box_cir = groupbox(gmid[1, 1:2], bermejo; fillalpha = 0.03, top = -86)
    # A blocklabel is centred on its row's CELL; the box reaches past that cell by different amounts
    # above and below (56 pt down, 34 or 86 up), so the two centres differ. Translate the label onto
    # the box --- after `addlabels!`, which re-solves the layout.
    centre_on_box!(lab, box) = let b = lab.layoutobservables.computedbbox[],
            d = box.layoutobservables.computedbbox[]

        Makie.translate!(
            lab.blockscene, 0,
            (d.origin[2] + d.widths[2] / 2) - (b.origin[2] + b.widths[2] / 2), 0
        )
    end

    # Top: brain illustration | LFP + raster | increment distribution
    # (a) carries no ylabel or xlabel, so the protrusion bands its neighbours fill with "Time (s)"
    # and tick labels sit empty around it. `Mixed` lets the aspect-locked brain reach into them ---
    # left to its own panel letter, right into the column gap, down into the xlabel band --- which
    # costs (b) and (c) nothing. The top is left alone so the title stays outside.
    ax_brain = pdfpanel!(
        gtop[1, 1], projectdir("brain.pdf"); title = "Mouse brain", flipx = true,
        # Shifted left by widening the left reach and insetting the right by the same amount, so the
        # drawn width is unchanged. The limit is the group box's left border (x = 10), not the panel
        # letter, which sits above the brain's top edge and so never meets it.
        alignmode = Makie.Mixed(left = -51, right = 28, bottom = -20)  # `Mixed` is ambiguous here
    )
    g_traces = gtop[1, 2] = GridLayout()
    ax_lfp, ax_raster = traces_panel!(g_traces, traces)
    ax_incr = Axis(
        gtop[1, 3]; yscale = log10, xlabel = unitlabel("Increment", "SD"), ylabel = "Density",
        title = "L2/3 increments",
        # pinned: the headroom the legend needs pushes Makie onto half-decade exponents otherwise
        yticks = LogTicks(-6:2:0)
    )
    increment_panel!(ax_incr, incr)

    # Middle: input field + trajectory | traces and densities
    # Centred, not pinned to the top: (d)'s title now stacks above the time colorbar rather than
    # sitting on the axis, so pinning would drive it into the blue box above. Its baseline no longer
    # agrees with (e) and (f), which is inherent to the title being two bands higher than theirs.
    g_field = gmid[1, 1] = GridLayout()
    ax_field, lab_field = circuit_field_panel!(g_field)
    # (e) and (f)/(g) share a cell with (d), but (d) protrudes 72 pt above it (time colorbar band
    # plus title) where this block protrudes only 28, so its titles sat 45 pt low. That band is
    # empty on this side, just unclaimed, and `Mixed` lets the block reach into it while leaving the
    # other three sides as `Inside`. The value is (d)'s own top protrusion, so the two blocks' outer
    # edges coincide and the titles line up. `Outside` cannot do this: it folds every protrusion
    # inside the box, so the block then reports none, the layout stops reserving the 57 pt gutter
    # (e)'s ylabels need, and they land on top of (d)'s colorbar.
    g_ctr = gmid[1, 2] = GridLayout(; alignmode = Makie.Mixed(top = -72))
    ctr = circuit_trace_panels!(g_ctr)
    colsize!(gmid, 1, Relative(0.36)) # the field is aspect-locked; an even split letterboxes it

    # Bottom: the combined curves; `cc` carries the MAPPLE fits quoted on the panels
    cc = combined_curves_panels!(gbot)

    # Row heights: the circuit row carries the tallest content (field + its colorbar, and the
    # three-panel trace stack). The top row's columns stay even, which keeps the brain large.
    # (d)'s title stack makes the circuit row protrude ~86 pt above its cell while the experiment
    # row's box reaches 56 pt below its own; the default gap between the two rows is not wide enough
    # to hold both, and the boxes overlap.
    colgap!(f.layout, 1, 10.0) # block labels to the panels' own ylabel band
    rowgap!(f.layout, 1, 26.0)
    rowsize!(f.layout, 1, Relative(0.3))
    rowsize!(f.layout, 2, Relative(0.38))
    addlabels!(
        [
            gtop[1, 1], g_traces[1, 1], gtop[1, 3],       # a-c  experiment
            g_field[0, 1], ctr.gtr[1, 1], ctr.gd[1, 1], ctr.gd[2, 1],  # d-g  circuit
            gbot[1, 1], gbot[1, 2], gbot[1, 3],           # h-j  combined curves
        ], f;
        # (a) has no ylabel to fill the gutter its box reaches into, unlike the rows below, so
        # its letter alone is pulled out to the page margin the ylabels set (x = 16 pt).
        # Narrowing the rows for the block labels pushed several titles into their letters; these
        # offsets restore ~12 pt of clearance (measured as title left edge minus letter right edge).
        dx = [-40, -20, 0, -12, -10, -5, -6, 0, 0, 0],
        # (d)'s letter anchors to the time colorbar's cell, whose top protrusion holds the colorbar
        # label and, above it, the panel title; lift it level with that title.
        dy = [0, 0, 0, 50, 0, 0, 0, 0, 0, 0]
    )
    # (d)'s title is a `Label` centred on its CELL, but the field axis is inset within that column
    # by its ylabel and ticks, so the title sat 17 pt left of the panel it titles. Translating it is
    # what `addlabels!` does for the letters; padding would not move it 1:1. After `addlabels!`,
    # since adding the letters re-solves the layout.
    let b = lab_field.layoutobservables.computedbbox[], d = Fathom.drawnbox(ax_field)
        Makie.translate!(
            lab_field.blockscene,
            (d.origin[1] + d.widths[1] / 2) - (b.origin[1] + b.widths[1] / 2), 0, 0
        )
    end
    centre_on_box!(lab_exp, box_exp)
    centre_on_box!(lab_cir, box_cir)
    display(f)

    # Two lines from (a)'s blue probe out to the top and bottom of (b), bracketing it: this is what
    # that probe records. They span the gap between two panels, so they cannot live in either axis;
    # the layout is solved by now, so the anchors are read off the drawn boxes rather than guessed.
    #
    # They go in an OVERLAY scene, not `f.scene`. Every Block paints its own child scene over the
    # figure's root scene, so a line drawn into the root is hidden wherever it crosses a panel --- and
    # the lower line runs most of the way across (a) before it clears the brain, leaving it to appear
    # from nowhere halfway to (b). A scene built after the blocks draws over all of them.
    let vp = Fathom.drawnbox(ax_brain), lim = ax_brain.finallimits[],
            top = Fathom.drawnbox(ax_lfp), bot = Fathom.drawnbox(ax_raster),
            overlay = Scene(f.scene; camera = campixel!, clear = false)

        px = vp.origin[1] + (BRAIN_PROBE[1] - lim.origin[1]) / lim.widths[1] * vp.widths[1]
        py = vp.origin[2] + (BRAIN_PROBE[2] - lim.origin[2]) / lim.widths[2] * vp.widths[2]
        for y in (top.origin[2] + top.widths[2], bot.origin[2])
            lines!(
                overlay, [Point2f(px, py), Point2f(top.origin[1], y)];
                color = (experiment_color, 0.6), linewidth = 2.2, linestyle = :dash
            )
        end
        @info "probe bracket" probe = round.((px, py)) panel_b_left = round(top.origin[1]) panel_b_y = round.((top.origin[2] + top.widths[2], bot.origin[2]))
    end
end

begin # * Save figure
    wsave(joinpath(outdir, "$NAME.pdf"), f)
    wsave(joinpath(outdir, "$NAME.png"), f)
    @info "Saved $(joinpath(outdir, "$NAME.pdf"))"
end

begin # * Statistics --- the same files plot_demo_run.jl and combined_curves.jl wrote, in this figure's folder
    open(joinpath(outdir, "fano_statistics.txt"), "w") do io
        println(io, percentilebootmedian(collect(cstats["mfano"])))
    end

    open(joinpath(outdir, "statistics.txt"), "w") do io
        # Outside WRCircuit these NamedTuples come back as JLD2 reconstructions, so reach their
        # entries by property rather than by `keys`/`getindex`.
        for v in propertynames(cstats["spectra"])
            println(io, "\n=== Variable: $(v) ===")
            println(io, "-- Spectrum fit --")
            println(
                io, percentilebootmedian(
                    map(x -> last(x.m.params.components.β), getproperty(cstats["spectrum_fits"], v))
                )
            )
            println(io, "-- MAD fit --")
            println(
                io,
                percentilebootmedian(
                    map(getproperty(cstats["mad_fits"], v)) do x
                        x isa Number ? x : first(x.m.params.components.β)
                    end
                )
            )
        end
    end

    open(joinpath(outdir, "combined_curves_$(STIM).txt"), "w") do io
        mad = plot_data["mad_curves"][STIM]
        m, (lo, hi) = hasproperty(mad, :slope) ? percentilebootmedian(collect(mad.slope)) :
            (mad.meanslope, (NaN, NaN))
        println(io, "$STIM mad median: $m, CI: ($lo, $hi)")
        m, (lo, hi) = percentilebootmedian(
            collect(_select(plot_data["spectral_exponents"][STIM], :Structure => "VISp", :layer => 2))
        )
        println(io, "$STIM spectral median: $m, CI: ($lo, $hi)")
        m, (lo, hi) = percentilebootmedian(
            collect(_select(plot_data["fano_slopes"][STIM], :Structure => "VISp", :layer => 2))
        )
        println(io, "$STIM fano median: $m, CI: ($lo, $hi)")
        for (label, sub) in
            (("mad", circuit.mad), ("spectral", circuit.psd), ("fano", circuit.fano))
            if hasproperty(sub, :exponents)
                m, (lo, hi) = percentilebootmedian(collect(sub.exponents))
                println(io, "circuit $label median: $m, CI: ($lo, $hi)")
            else
                println(io, "circuit $label exponent: $(sub.exponent)")
            end
        end
        # The numbers actually printed on the figure. The MAD exponents are MAPPLE re-fits of the
        # drawn median curves; the PSD exponents are the pipelines' own fixed-knot fits, quoted as
        # they come (the panel does not refit them).
        println(io, "figure mad a (fit to drawn median): experiment $(only(betas(cc.mfit_exp))), circuit $(only(betas(cc.mfit_circ)))")
        println(io, "figure psd b: experiment $(cc.b_exp), circuit $(cc.b_circ)")
        println(io, "figure fano c (unified fit to drawn median): experiment $(cc.c_exp), circuit $(cc.c_circ)")
    end

    open(joinpath(outdir, "increment_statistics.txt"), "w") do io
        println(io, "L2/3 increment excess kurtosis (median across sessions of per-session channel medians; QUOTED IN TEXT): $(increment_kurtosis(incr))")
        println(io, "L2/3 increment excess kurtosis (flat median over channels; shape diagnostic only): $(median(incr["kurtosis"]))")
        println(io, "FT surrogate excess kurtosis (median across sessions): $(haskey(incr, "kurtosis_surrogate_session") ? median(incr["kurtosis_surrogate_session"]) : median(incr["kurtosis_surrogate"]))")
        println(io, "FT surrogate excess kurtosis (flat median over channels): $(median(incr["kurtosis_surrogate"]))")
        println(io, "channels: $(incr["nchannels"]), sessions: $(length(incr["sessions"]))")
    end
    @info "Saved statistics to $outdir"
end

begin # * Source data, one file per panel
    writedlm(
        joinpath(outdir, "panelB_lfp.tsv"),
        vcat(["t" "lfp"], hcat(traces["t"], traces["lfp"])), '\t'
    )
    open(joinpath(outdir, "panelB_spikes.tsv"), "w") do io
        println(io, "neuron\tspike_time")
        for (i, s) in enumerate(traces["spikes"]), t in s
            println(io, "$i\t$t")
        end
    end
    writedlm(
        joinpath(outdir, "panelD_increments.tsv"),
        vcat(
            ["increment_sd" "density" "density_surrogate" "gaussian"],
            hcat(incr["centres"], incr["density"], incr["density_surrogate"], gaussian.(incr["centres"]))
        ), '\t'
    )
    let mad = plot_data["mad_curves"][STIM], psd = plot_data["spectral_curves"][STIM]["VISp"],
            fano = plot_data["fano_curves"][STIM]
        writedlm(
            joinpath(outdir, "panelJ_mad.tsv"),
            vcat(["t" "mu" "sigma_lo" "sigma_hi"], hcat(mad.t_all, mad.mu, mad.σl, mad.σh)), '\t'
        )
        writedlm(
            joinpath(outdir, "panelJ_mad_circuit.tsv"),
            vcat(["t" "mu"], hcat(circuit.mad.t, circuit.mad.mu)), '\t'
        )
        writedlm(
            joinpath(outdir, "panelK_psd.tsv"),
            vcat(["f" "mu" "sigma_lo" "sigma_hi"], hcat(psd.f, psd.μ, psd.σl, psd.σh)), '\t'
        )
        writedlm(
            joinpath(outdir, "panelK_psd_circuit.tsv"),
            vcat(["f" "mu"], hcat(circuit.psd.f, circuit.psd.mu)), '\t'
        )
        writedlm(
            joinpath(outdir, "panelL_fano.tsv"),
            vcat(["t" "mu" "sigma_lo" "sigma_hi"], hcat(fano.t_all, fano.mu, fano.sl, fano.su)), '\t'
        )
        writedlm(
            joinpath(outdir, "panelL_fano_circuit.tsv"),
            vcat(["t" "mu"], hcat(circuit.fano.t, circuit.fano.mu)), '\t'
        )
    end
    @info "Saved source data to $outdir"
end
