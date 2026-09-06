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

const NAME = "Fig1_combined_curves"
const outdir = plotsdir(NAME)
const STIM = "spontaneous"
const experiment_color = :cornflowerblue
const circuit_color = :crimson

mkpath(outdir)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"Bootstrap median + 95% CI. Local copy so this script needs no Bootstrap.jl (as combined_curves.jl does)."
function bootstrapmedian(x; N = 10_000, α = 0.05)
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
function pdfpanel!(gp, pdf; dpi = 600)
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
    ax = Axis(gp; aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    image!(ax, rotr90(wload(png)))
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
normalise(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))

"Map `y` into the log-space min-max frame of the drawn curve `ref`, so a fit overlays
`exp10.(normalise(log10.(ref)))` with its position and log-log slope relative to that curve intact."
lognorm(y, ref) = exp10.(
    (log10.(y) .- minimum(log10.(ref))) ./
        (maximum(log10.(ref)) - minimum(log10.(ref)))
)

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
include(joinpath(@__DIR__, "variability_exponent.jl"))


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
const FIELD_TINDEX = 42_571
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
        gl[1, 1]; ylabel = "LFP", yticks = ([], []), xticks = ([], []),
        xgridvisible = false, limits = ((0, maximum(d["t"])), nothing),
        title = "Neuropixels recordings"
    )
    lines!(axl, d["t"], d["lfp"]; color = experiment_color, linewidth = 1.2)

    axr = Axis(
        gl[2, 1]; ylabel = "Neuron", xlabel = "Time (s)", yticks = ([], []),
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
        lines!(ax, x[ks], ys[ks]; color = (abyad, 0.9), linewidth = 1.2, label = "Surrogate")
    end
    xg = range(-xlim, xlim, length = 400)
    lines!(ax, xg, gaussian.(xg); color = chernoe, linestyle = :dash, linewidth = 1.5, label = "Gaussian")
    keep = (y .> 0) .& (abs.(x) .<= xlim)
    lines!(ax, x[keep], y[keep]; color = experiment_color, linewidth = 2, label = "Data")
    text!(
        ax, 0.04, 0.92; text = @sprintf("κ = %.2f", increment_kurtosis(d)), space = :relative,
        align = (:left, :top), fontsize = 11, color = experiment_color
    )
    ylims!(ax, 1.0e-6, 1.5)
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
        gl[1, 1]; xlabel = "X (mm)", ylabel = "Y (mm)", limits = ((0, DX), (0, DX)),
        xticks = 0:0.25:0.5, yticks = 0:0.25:0.5, xtickformat = terseticks,
        ytickformat = terseticks, aspect = 1          # the patch is a disc, so keep the field square
    )
    h = heatmap!(ax, xx, xx, frame'; colormap = seethrough(reverse(sunrise)), rasterize = 10)
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
    Colorbar(gl[1, 2], h; label = "Input (nA)", width = 8)
    Colorbar(
        gl[0, 1], p; vertical = false, label = "Time (s)", tickformat = terseticks, height = 8
    )
    rowgap!(gl, 1, Relative(0.02))
    colgap!(gl, 1, Relative(0.02))
    return ax
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
        gtr[1, 1]; title = "Circuit model", ylabel = "V (mV)", yticklabelspace = YSPACE,
        yticks = WilkinsonTicks(3; k_max = 3), xticks = ([], []), xgridvisible = false,
        limits = (xlim, nothing)
    )
    hlines!(axv, [-50]; color = bermejo)
    hlines!(axv, [-70]; color = bermejo, linestyle = :dash)
    hlines!(axv, [craw["mean_V"]]; color = :gray, linestyle = :dash)
    lines!(axv, tsec, collect(ustripall(V[ts, neuron])); linewidth = 1.2, color = experiment_color)
    text!(
        axv, 0.97, 0.05; text = @sprintf("ν ≈ %.1f Hz", craw["nu"]), space = :relative,
        align = (:right, :bottom), fontsize = 9
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
        gtr[3, 1]; ylabel = "I (nA)", xlabel = "Time (s)", yticklabelspace = YSPACE,
        yticks = WilkinsonTicks(3; k_max = 3), limits = (xlim, (-1, 3))
    )
    lines!(axi, tsec, collect(ustripall(INPUT[ts, neuron])); linewidth = 1.2, color = experiment_color)

    axvd = Axis(
        gd[1, 1]; title = "Potential", ylabel = "Density", xlabel = "V (mV)",
        xticks = WilkinsonTicks(3; k_max = 3),
        yticks = WilkinsonTicks(3; k_max = 3), yticklabelspace = YSPACE_D
    )
    zigg!(axvd, cstats["V_hist"]; dropfirst = true)   # first bin is the Vr reset pile-up
    vlines!(axvd, [craw["mean_V"]]; color = :gray, linestyle = :dash)

    axid = Axis(
        gd[2, 1]; title = "Step sizes", ylabel = "Density", yticklabelspace = YSPACE_D,
        xlabel = "|ΔI| (nA)", xscale = log10, yscale = log10,
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

    # * MAD. Both curves are min-max normalised in log space, so only the SLOPES are comparable;
    #   each dashed line is a MAPPLE fit computed HERE on the drawn median curve, drawn over the
    #   band it was fit on.
    mfit_exp = mad_mapple(mad.t_all, mad.mu)
    mfit_circ = mad_mapple(circuit.mad.t, circuit.mad.mu)
    axm = Axis(
        gl[1, 1]; ylabel = "MAD (arb. units)", xlabel = "Time lag (s)", title = "Diffusion",
        xscale = log10, yscale = log10,
        limits = ((10^(-3.4), 10^0.1), nothing)
    )
    band!(
        axm, mad.t_all, exp10.(normalise(log10.(mad.σl))), exp10.(normalise(log10.(mad.σh)));
        color = experiment_color, alpha = 0.5
    )
    lines!(
        axm, mad.t_all, exp10.(normalise(log10.(mad.mu)));
        color = experiment_color, label = "Experiment\n(LFP)"
    )
    idxs = mad.fit_t .< 0.005
    lines!(
        axm, mad.fit_t[idxs] .* 2, lognorm(predict(mfit_exp, mad.fit_t), mad.mu)[idxs];
        linestyle = :dash, color = experiment_color
    )
    lines!(
        axm, circuit.mad.t, exp10.(normalise(log10.(circuit.mad.mu)));
        color = circuit_color, label = "Circuit\n(input)"
    )
    idxs = circuit.mad.fit_t .< 0.005
    lines!(
        axm, circuit.mad.fit_t[idxs] ./ 2,
        lognorm(predict(mfit_circ, circuit.mad.fit_t), circuit.mad.mu)[idxs];
        color = circuit_color, linestyle = :dash
    )
    text!(
        axm, 1.0e-2, 10^0.7; text = "a = $(round(only(betas(mfit_exp)), sigdigits = 2))",
        color = experiment_color, align = (:left, :top)
    )
    text!(
        axm, 10^(-3.35), 10^1; text = "a = $(round(only(betas(mfit_circ)), sigdigits = 2))",
        color = circuit_color, align = (:left, :top)
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
        gl[1, 2]; xlabel = "Frequency (Hz)", ylabel = "PSD (arb. units)", title = "Power spectrum",
        xscale = log10, yscale = log10,
        xticks = [3, 10, 30, 100], limits = ((2, 500), nothing)
    )
    lines!(axp, psd.f, exp10.(normalise(log10.(psd.μ))); color = (experiment_color, 0.8))
    band!(
        axp, psd.f, exp10.(normalise(log10.(psd.σl))), exp10.(normalise(log10.(psd.σh)));
        color = (experiment_color, 0.32)
    )
    lines!(
        axp, psd.f, 1.25 .* exp10.(normalise(log10.(psd.f .^ psd.spectral_exponent_median)));
        color = experiment_color, linestyle = :dash
    )
    lines!(axp, circuit.psd.f, exp10.(normalise(log10.(circuit.psd.mu))) .* 1.35; color = circuit_color)
    lines!(
        axp, circuit.psd.fit_f,
        1.25 .* exp10.(normalise(log10.(circuit.psd.fit_f .^ circuit.psd.exponent)));
        color = circuit_color, linestyle = :dash
    )
    text!(
        axp, 7, 10^0.4; text = "b = $(round(psd.spectral_exponent_median; sigdigits = 3))",
        color = experiment_color, align = (:left, :top)
    )
    text!(
        axp, 20, 10; text = "b = $(round(circuit.psd.exponent; sigdigits = 3))",
        color = circuit_color, align = (:left, :bottom)
    )

    # * Fano factor, unnormalised (both are dimensionless counts).
    axf = Axis(
        gl[1, 3]; xlabel = "Time lag (s)", ylabel = "Fano factor", title = "Fano factor",
        xscale = log10, yscale = log10
    )
    band!(axf, 0.001 .* fano.t_all, fano.sl, fano.su; color = experiment_color, alpha = 0.3)
    lines!(axf, 0.001 .* fano.t_all, fano.mu; color = experiment_color)
    # Unified refit of the DRAWN experiment median (the stored `fano.mslope` is the legacy
    # fixed-band OLS); the dashed guide spans the fitted scaling band, slope-β through its centre.
    ffit_exp = variability_exponent(fano.t_all, fano.mu)
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
        axf, 0.001 .* 40, 10^0.32; text = "c = $(round(ffit_exp.β, digits = 2))",
        color = experiment_color, align = (:left, :center)
    )
    text!(
        axf, 0.001 .* 1.2, 1.4; text = "c = $(round(circuit.fano.exponent, digits = 2))",
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
    f = SixPanel()
    gtop = f[1, 1] = GridLayout()
    gmid = f[2, 1] = GridLayout()
    gbot = f[3, 1] = GridLayout()

    # Top: brain illustration | LFP + raster | increment distribution
    ax_brain = pdfpanel!(gtop[1, 1], projectdir("brain.pdf"))
    g_traces = gtop[1, 2] = GridLayout()
    traces_panel!(g_traces, traces)
    ax_incr = Axis(
        gtop[1, 3]; yscale = log10, xlabel = "Increment (SD)", ylabel = "Density",
        title = "L2/3 increments"
    )
    increment_panel!(ax_incr, incr)

    # Middle: input field + trajectory | traces and densities
    g_field = gmid[1, 1] = GridLayout()
    circuit_field_panel!(g_field)
    g_ctr = gmid[1, 2] = GridLayout()
    ctr = circuit_trace_panels!(g_ctr)
    colsize!(gmid, 1, Relative(0.36)) # the field is aspect-locked; an even split letterboxes it

    # Bottom: the combined curves; `cc` carries the MAPPLE fits quoted on the panels
    cc = combined_curves_panels!(gbot)

    # Row heights: the circuit row carries the tallest content (field + its colorbar, and the
    # three-panel trace stack). The top row's columns stay even, which keeps the brain large.
    rowsize!(f.layout, 1, Relative(0.3))
    rowsize!(f.layout, 2, Relative(0.38))
    addlabels!(
        [
            gtop[1, 1], g_traces[1, 1], gtop[1, 3],       # a-c  experiment
            g_field[0, 1], ctr.gtr[1, 1], ctr.gd[1, 1], ctr.gd[2, 1],  # d-g  circuit
            gbot[1, 1], gbot[1, 2], gbot[1, 3],           # h-j  combined curves
        ], f; fontsize = 16,
        # (d) anchors to the time colorbar's row, whose own label occupies the protrusion the panel
        # letter would otherwise use; lift it clear so it sits above the bar.
        dy = [0, 0, 0, 26, 0, 0, 0, 0, 0, 0]
    )
    display(f)
end

begin # * Save figure
    wsave(joinpath(outdir, "$NAME.pdf"), f)
    wsave(joinpath(outdir, "$NAME.png"), f)
    @info "Saved $(joinpath(outdir, "$NAME.pdf"))"
end

begin # * Statistics --- the same files plot_demo_run.jl and combined_curves.jl wrote, in this figure's folder
    open(joinpath(outdir, "fano_statistics.txt"), "w") do io
        println(io, bootstrapmedian(collect(cstats["mfano"])))
    end

    open(joinpath(outdir, "statistics.txt"), "w") do io
        # Outside WRCircuit these NamedTuples come back as JLD2 reconstructions, so reach their
        # entries by property rather than by `keys`/`getindex`.
        for v in propertynames(cstats["spectra"])
            println(io, "\n=== Variable: $(v) ===")
            println(io, "-- Spectrum fit --")
            println(
                io, bootstrapmedian(
                    map(x -> last(x.m.params.components.β), getproperty(cstats["spectrum_fits"], v))
                )
            )
            println(io, "-- MAD fit --")
            println(
                io,
                bootstrapmedian(
                    map(getproperty(cstats["mad_fits"], v)) do x
                        x isa Number ? x : first(x.m.params.components.β)
                    end
                )
            )
        end
    end

    open(joinpath(outdir, "combined_curves_$(STIM).txt"), "w") do io
        mad = plot_data["mad_curves"][STIM]
        m, (lo, hi) = hasproperty(mad, :slope) ? bootstrapmedian(collect(mad.slope)) :
            (mad.meanslope, (NaN, NaN))
        println(io, "$STIM mad median: $m, CI: ($lo, $hi)")
        m, (lo, hi) = bootstrapmedian(
            collect(_select(plot_data["spectral_exponents"][STIM], :Structure => "VISp", :layer => 2))
        )
        println(io, "$STIM spectral median: $m, CI: ($lo, $hi)")
        m, (lo, hi) = bootstrapmedian(
            collect(_select(plot_data["fano_slopes"][STIM], :Structure => "VISp", :layer => 2))
        )
        println(io, "$STIM fano median: $m, CI: ($lo, $hi)")
        for (label, sub) in
            (("mad", circuit.mad), ("spectral", circuit.psd), ("fano", circuit.fano))
            if hasproperty(sub, :exponents)
                m, (lo, hi) = bootstrapmedian(collect(sub.exponents))
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
