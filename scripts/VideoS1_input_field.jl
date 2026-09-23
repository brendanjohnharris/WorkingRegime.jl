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
using Statistics
using DelimitedFiles

set_theme!(Fathom.fathom())

const NAME = "VideoS1_input_field"
const outdir = plotsdir(NAME)
mkpath(outdir)

# The input field of Fig 1d, animated: the same field, the same colormap and the same centre-of-mass
# track, run forward in time.
#
# The window came from sweeping the trace for the clearest JUMPS: score the centre of mass in 5 ms
# blocks, counting blocks that move more than 20 grid units (a hop across ~28% of the field) against
# the median block step. The sweep's best 300 ms started at sample 29_501 (10 hops, median step 2.7,
# a max/median ratio of 15 --- the bump sits, flicks to a new place, sits again); this runs the
# second onward from it. Fig 1d's own window scores 3 hops at a ratio of 7, having been picked for a
# long unbroken track instead, so the video does not contain the frame that figure draws.
const START = 29_501 + 1_500       # 150 ms past the sweep's pick
const DURATION_MS = 1_000          # of simulation time
const FRAME_MS = 1                 # simulation time per frame; the field flickers on ~2.5 ms, so
                                   # a coarser step aliases the very fluctuation the video is for
const FRAMERATE = 30               # => 1000 frames, a 33 s clip at 1/33 speed
const STAMP_STEP_MS = 10           # the counter ticks in these steps; at the 1 ms frame step it
                                   # flickers through digits too fast to read
const TRAIL_MS = 40                # about one hop of history: long enough to see where the bump
                                   # came from, short enough that successive hops do not overlap
# Fixed across every frame, unlike Fig 1d which autoscales its single frame: a per-frame range
# would rescale the colours on each flare and the field would appear to pulse in brightness when
# it is only the scale moving. Clipped at 0 below, as Fig 1d is, and at a high quantile above.
# NOTE this quantile is relative to WINDOW, so it must be re-checked whenever the window moves: the
# same 0.99 gave 3.3 nA over the 300 ms window and only 2.3 nA over this calmer second, which
# flattened every flare into a solid blob. 0.995 puts it back at 3.2 nA, where the ordinary field
# still carries colour and only the bump cores saturate.
const CLIP_QUANTILE = 0.995

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading circuit data"
craw = jldopen(
    datadir("WRCircuit", "demo_run.jld2"); typemap = toolsarray_typemap
) do f                                       # 2.2 GB on disk: take only the input field
    Dict("fixed_params" => f["fixed_params"], "N" => f["N"], "E_input" => f["E_input"])
end

const N_GRID = craw["N"]
const DX = craw["fixed_params"].dx
const INPUT = craw["E_input"]
const TIME_MS = ustripall(collect(times(INPUT)))
const DT_MS = TIME_MS[2] - TIME_MS[1]        # ~0.1 ms; sample counts are NOT milliseconds

const STRIDE = round(Int, FRAME_MS / DT_MS)
const NFRAMES = round(Int, DURATION_MS / FRAME_MS)
const TRAIL = round(Int, TRAIL_MS / DT_MS)
const FRAMES = START:STRIDE:(START + NFRAMES * STRIDE)
const BLOCK = round(Int, 5 / DT_MS)          # the 5 ms block the jump sweep scored on

"""
    comtrack(grid, ts)

Centre of mass of each frame in `ts`, by circular mean, so the track does not jump when the bump
crosses the periodic boundary. Mirrors `track_com` in `Fig1_combined_curves.jl`, vectorised: the
four sums are matrix-vector products, which matters here because the video needs a COM per frame
over a long stretch rather than one short window.

Returns `(x, y)` in mm, matching how the panel plots them: the SECOND grid index is x.
"""
function comtrack(grid, ts)
    n = size(grid, 2)
    θ = Float32.(2π .* (0:(n - 1)) ./ n)
    W = abs.(reshape(view(grid, ts, :, :), length(ts), n * n))   # (t, a + (b-1)n)
    v(f, which) = vec(Float32[f(θ[which == :a ? a : b]) for a in 1:n, b in 1:n])
    circmean(s, c) = mod.(atan.(W * s, W * c) .* n ./ (2f0π), n) .+ 1f0
    a = circmean(v(sin, :a), v(cos, :a))     # first index  -> y
    b = circmean(v(sin, :b), v(cos, :b))     # second index -> x
    return DX .* b ./ n, DX .* a ./ n
end

const GRID = reshape(parent(ustripall(INPUT)), (size(INPUT, 1), N_GRID, N_GRID))
const TOP = quantile(vec(view(GRID, first(FRAMES):last(FRAMES), :, :)), CLIP_QUANTILE)

# One COM per SAMPLE over the window plus its lead-in trail, so the trail is drawn at full
# resolution rather than at the frame rate.
const TRACK_TS = (first(FRAMES) - TRAIL):last(FRAMES)
const TRACK_X, TRACK_Y = comtrack(GRID, TRACK_TS)

# Blank the sample after a seam crossing, so the trail BREAKS there instead of being drawn straight
# across the field, as Fig 1d does for its own track. It matters far more here: that figure picked
# one of the few windows with no crossing at all, while 300 ms of free running crosses repeatedly.
# The head keeps the unblanked track, or it would vanish on the crossing frames.
const WRAPPED = [false; (abs.(diff(TRACK_X)) .> DX / 2) .| (abs.(diff(TRACK_Y)) .> DX / 2)]
const TRAIL_X = [WRAPPED[i] ? NaN32 : TRACK_X[i] for i in eachindex(TRACK_X)]
const TRAIL_Y = [WRAPPED[i] ? NaN32 : TRACK_Y[i] for i in eachindex(TRACK_Y)]

@info "Field" size(GRID) colorrange = (0, round(TOP, digits = 2)) frames = length(FRAMES) TOP

# ──────────────────────────────────────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────────────────────────────────────

begin # * Build the figure once; only the observables change per frame
    fig = OnePanel()
    xx = range(0, DX, length = N_GRID)

    frame = Observable(Matrix(GRID[first(FRAMES), :, :]'))
    trail = Observable(Point2f[])   # NaN-blanked at seam crossings; see TRAIL_X
    head = Observable(Point2f[Point2f(TRACK_X[TRAIL + 1], TRACK_Y[TRAIL + 1])])
    stamp = Observable("0 ms")

    ax = Axis(
        fig[1, 1]; xlabel = "X (mm)", ylabel = "Y (mm)", limits = ((0, DX), (0, DX)),
        xticks = 0:0.25:0.5, yticks = 0:0.25:0.5, xtickformat = terseticks,
        ytickformat = terseticks, aspect = 1        # square, as in Fig 1d
    )
    cmap = seethrough(reverse(sunrise))
    h = heatmap!(ax, xx, xx, frame; colormap = cmap, colorrange = (0, TOP), highclip = cmap[1.0])
    # White underlay then the coloured track, as Fig 1d draws it; here the colour is recency
    # within the trail rather than absolute time, since the window keeps moving.
    lines!(ax, trail; color = :white, linewidth = 4)
    lines!(
        ax, trail; color = 1:TRAIL, colorrange = (1, TRAIL),
        colormap = reverse(cgrad(:turbo)), linewidth = 2
    )
    scatter!(ax, head; color = chernoe, markersize = 9, strokecolor = :white, strokewidth = 1.5)
    text!(
        ax, 0.03, 0.97; text = stamp, space = :relative, align = (:left, :top),
        font = :bold, glowcolor = :white, glowwidth = 8
    )
    Colorbar(fig[1, 2], h; label = "Input (nA)", width = 8)
    colgap!(fig.layout, 1, 6.0)
    fig
end

begin # * Record
    path = joinpath(outdir, "$NAME.mp4")
    @info "Recording $(length(FRAMES)) frames to $path"
    record(fig, path, enumerate(FRAMES); framerate = FRAMERATE) do (i, t)
        frame[] = Matrix(GRID[t, :, :]')
        j = t - first(TRACK_TS) + 1                    # index into the full-resolution track
        rng = (j - TRAIL + 1):j
        trail[] = Point2f.(view(TRAIL_X, rng), view(TRAIL_Y, rng))
        head[] = Point2f[Point2f(TRACK_X[j], TRACK_Y[j])]
        stamp[] = string(STAMP_STEP_MS * fld((i - 1) * FRAME_MS, STAMP_STEP_MS), " ms")
    end
    @info "Saved $path"
end

begin # * Source data: the track, not the field (the field is the raw simulation output)
    writedlm(
        joinpath(outdir, "track.tsv"),
        vcat(
            ["t_ms" "x_mm" "y_mm"],
            hcat((TRACK_TS .- first(FRAMES)) .* DT_MS, TRACK_X, TRACK_Y)
        ), '\t'
    )
    open(joinpath(outdir, "settings.txt"), "w") do io
        # Measured on THIS window rather than quoted from the sweep, which scored a 300 ms window
        # starting 150 ms earlier; the two do not have the same jump statistics.
        blocks = 1:BLOCK:(length(TRACK_X) - TRAIL - BLOCK)
        pd(u, v) = (d = abs(u - v); min(d, DX - d))
        hop(i) = sqrt(
            pd(TRACK_X[i + TRAIL], TRACK_X[i + TRAIL + BLOCK])^2 +
                pd(TRACK_Y[i + TRAIL], TRACK_Y[i + TRAIL + BLOCK])^2
        )
        d = hop.(blocks)
        big = count(>(20 * DX / N_GRID), d)      # a hop across ~28% of the field, in BLOCK samples
        println(io, "start sample: $START (the jump sweep's pick, 29501, plus 150 ms)")
        println(io, "jumps in this window: $big hops > 20 grid units per 5 ms, median step $(round(Statistics.median(d) * N_GRID / DX, digits = 2)) grid units")
        println(io, "frames: $(length(FRAMES)) at $FRAME_MS ms of simulation each, $FRAMERATE fps")
        println(io, "duration: $DURATION_MS ms of simulation in $(round(length(FRAMES) / FRAMERATE, digits = 1)) s of video")
        println(io, "colorrange: (0, $TOP) nA, the $(CLIP_QUANTILE) quantile over the window, high-clipped")
        println(io, "trail: $TRAIL_MS ms")
        println(io, "time counter: ticks every $STAMP_STEP_MS ms")
    end
    @info "Saved source data to $outdir" seam_crossings = sum(WRAPPED)
end
