#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.13 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate :WorkingRegime
using CairoMakie
using Fathom
using TimeseriesTools
using TimeseriesMakie
using DelimitedFiles
import StatsBase
import Fathom: crimson, cornflowerblue, cucumber, california # Foresight-era names, unexported by Fathom

set_theme!(Fathom.fathom()) # LaTeXStrings render in STIX Two serif by Fathom default
# Legends one point below the theme's 14, for this figure only: it carries several, some of them
# inside small inset panels. Set on the theme rather than on each `axislegend` call. Fathom's
# Legend block sets framevisible, padding, patchcolor and titlefont but no sizes, so both label and
# title otherwise inherit the global font size.
update_theme!(
    Legend = (;
        labelsize = Fathom.fathomfontsize() - 1,
        titlesize = Fathom.fathomfontsize() - 1,
    )
)

begin # * Options
    inset_log = false  # true: log-log as inset over linear plot; false: log-log as full axis
    order_rows = true  # true: rows = {space, time}; false: columns = {space, time}
    NAME = "Fig2_bFNS"
    outdir = plotsdir(NAME)
    datafile = datadir("WRTheory", "bFNS_data.jld2")
    nyticks = 4
end

begin # * Load data (produced by WRTheory/scripts/bFNS_data.jl)
    data = wload(datafile)
    xs, Vs, Ṽs = data["xs"], data["Vs"], data["Ṽs"]
    xs_pdf, p1, p2 = data["xs_pdf"], data["p1"], data["p2"]
    xs_a1, xs_a2 = data["xs_a1"], data["xs_a2"]
    ts, R1, R2 = data["ts"], data["R1"], data["R2"]
    s_hi, s_lo = data["s_hi"], data["s_lo"]
    ts_windows, configs = data["ts_windows"], data["configs"]
    τs, mads, gmads = data["τs"], data["mads"], data["gmads"]
    τfit, mad_fit, a_exponent = data["τfit"], data["mad_fit"], data["a_exponent"]
    psd, gpsd = data["psd"], data["gpsd"]
    psd_fit_x, psd_fit_y = data["psd_fit_x"], data["psd_fit_y"]
    b_exponent = data["b_exponent"]
    ma, ms = data["ma"], data["ms"]
    prms = data["params"]
    α1, α2, β1, β2, β_hi, β_lo = prms.α1, prms.α2, prms.β1, prms.β2, prms.β_hi, prms.β_lo
end

begin # * Set up figure
    f = SixPanel()
    g_main = f[1:2, 1:2] = GridLayout()
    gs = [g_main[1, 1], g_main[1, 2], g_main[2, 1], g_main[2, 2]] # row-major
    # Panel indices: potential, step-size, relaxation, spectrum
    if order_rows
        # Row 1 = space (potential, step-size), Row 2 = time (relaxation, spectrum)
        gi_potential, gi_stepsize, gi_relax, gi_spectrum = 1, 2, 3, 4
    else
        # Col 1 = space (potential, step-size), Col 2 = time (relaxation, spectrum)
        gi_potential, gi_stepsize, gi_relax, gi_spectrum = 1, 3, 2, 4
    end

    # Group boxes: space terms (α) vs time terms (β). Negative Outside reaches over the
    # protrusions so titles, panel letters, and axis labels sit inside the box.
    # `bottom` is per-box: the c/d box has to reach past the lowest trace in the right-hand column,
    # which the a/b box must not do or it would run into the c/d box below it.
    groupbox(gp, c; fillalpha = 0.05, bottom = -54) = Box(
        gp; color = (c, fillalpha), strokecolor = (c, 0.35), strokewidth = 1.5,
        cornerradius = 8, alignmode = Outside(-62, -18, bottom, -38) # measured: clears ylabels/xlabels and overhanging right ticks
    )
    # Vertical block labels outside the boxes, in a new column 0 of g_main: they push only the
    # boxed rows' content right, leaving the summary row (a separate layout) at full width, so the
    # left edges of the two no longer align --- deliberate.
    lab_space = blocklabel(g_main[1, 0], "Space-fractional")
    lab_time = blocklabel(g_main[2, 0], "Time-fractional")
    box_space, box_time = if order_rows
        groupbox(g_main[1, 1:2], baikal),                              # space
            groupbox(g_main[2, 1:2], bermejo; fillalpha = 0.03, bottom = -66) # time
    else
        groupbox(g_main[1:2, 1], baikal), groupbox(g_main[1:2, 2], bermejo; fillalpha = 0.03)
    end
    # Widening this gap does three things at once: it separates the two group rectangles, and since
    # g_main then asks the figure for more height, it pushes the c/d block down and takes that height
    # off the summary row below, squaring up (f)/(g)'s aspect.
    order_rows ? rowgap!(g_main, 1, 31) : colgap!(g_main, 1, 31) # air between the group boxes
end

begin # * Panel 1 — Effective potential for unimodal density
    ax = Axis(
        gs[gi_potential], xlabel = mit("x"), ylabel = mit("V(x)"),
        limits = (extrema(xs), (-0.5, maximum(Vs))), title = "Potential function",
        yticks = WilkinsonTicks(nyticks)
    )
    lines!(ax, xs, Vs; color = :cornflowerblue, label = "Potential")
    lines!(
        ax, xs, Ṽs; color = :crimson,
        label = "Effective potential"
    )
    axislegend(ax; position = :ct, title = rich(mit("α"), " = $(prms.α)"))
end

begin # * Panel 2 — Step-size distribution: α = 2 (Gaussian) vs α = 1.5 (heavy-tailed)
    if inset_log
        ax = Axis(
            gs[gi_stepsize], xlabel = mit("|x|"), ylabel = mit("p(|x|)"),
            title = "Step-size distribution",
            limits = ((0, 6), (0, nothing)),
            yticks = WilkinsonTicks(nyticks)
        )
        lines!(ax, xs_pdf, p1; color = :cornflowerblue, label = rich(mit("α"), " = $(α1)"))
        lines!(ax, xs_pdf, p2; color = :crimson, label = rich(mit("α"), " = $(α2)"))
        axislegend(ax; position = :lb)

        ax_inset = Axis( # log-log view of the pdf
            gs[gi_stepsize]; width = Relative(0.5), height = Relative(0.5),
            halign = 0.95, valign = 0.9,
            xscale = log10, yscale = log10,
            xlabelsize = 10, ylabelsize = 10,
            xticklabelsize = 8, yticklabelsize = 8,
            backgroundcolor = :white,
            limits = ((nothing, 25), (1.0e-4, 1.0e0)),
            yticks = LogTicks(WilkinsonTicks(nyticks - 1))
        )
        lines!(ax_inset, xs_pdf, p1; color = :cornflowerblue)
        lines!(ax_inset, xs_pdf, p2; color = :crimson)
        translate!(ax_inset.blockscene, 0, 0, 100)
    else
        bins = logrange(1, 40, length = 20)
        ax = Axis(
            gs[gi_stepsize]; xlabel = mit("|x|"), ylabel = mit("p(|x|)"),
            title = "Step-size distribution",
            xscale = log10, yscale = log10,
            limits = ((1, 25), (1.0e-4, nothing)),
            yticks = LogTicks(WilkinsonTicks(nyticks))
        )
        ziggurat!(
            ax, xs_a2; strokecolor = :crimson, label = rich(mit("α"), " = $(α2)"), bins,
            normalization = :pdf, fillalpha = 0.3, color = brighten(crimson, 0.5)
        )
        ziggurat!(
            ax, xs_a1; strokecolor = :cornflowerblue, label = rich(mit("α"), " = $(α1)"), bins,
            normalization = :pdf, fillalpha = 0.3,
            color = brighten(cornflowerblue, 0.5)
        )
        axislegend(ax; position = :rt)
    end
end

begin # * Panel 3 — Relaxation function: β = 1 (exponential) vs β < 1 (Mittag-Leffler)
    ts_log = ts[2:end] # skip t = 0 for the log axes
    R1_log = R1[2:end]
    R2_log = R2[2:end]

    if inset_log
        ax = Axis(
            gs[gi_relax], xlabel = "Time", ylabel = mit("R(t)"),
            title = "Relaxation function", limits = ((0, 5), (0, 1.05))
        )
        lines!(ax, ts, R1; color = :cornflowerblue, label = rich(mit("β"), " = $(β1)"))
        lines!(
            ax, ts, R2; color = :crimson,
            label = rich(mit("β"), " = $(β2)")
        )
        hlines!(ax, [0]; color = :gray, linewidth = 0.5)
        axislegend(ax; position = :lb)

        ax_inset = Axis( # log-log view
            gs[gi_relax]; width = Relative(0.5), height = Relative(0.5),
            halign = 0.95, valign = 0.9,
            xscale = log10, yscale = log10,
            xlabelsize = 10, ylabelsize = 10,
            xticklabelsize = 8, yticklabelsize = 8,
            backgroundcolor = :white,
            limits = ((minimum(ts_log), maximum(ts_log)), (1.0e-2, 1.0e0 + 0.5)),
            yticks = LogTicks(WilkinsonTicks(nyticks))
        )
        lines!(ax_inset, ts_log, R1_log; color = :cornflowerblue)
        lines!(ax_inset, ts_log, abs.(R2_log); color = :crimson)
        translate!(ax_inset.blockscene, 0, 0, 100)
    else
        ax = Axis(
            gs[gi_relax]; xlabel = mit("t"), ylabel = mit("R(t)"),
            title = "Relaxation function",
            xscale = log10, yscale = log10,
            limits = ((minimum(ts_log), maximum(ts_log)), (1.0e-2, 1.0e0 + 0.5)),
            yticks = LogTicks(WilkinsonTicks(nyticks))
        )
        lines!(ax, ts_log, R1_log; color = :cornflowerblue, label = rich(mit("β"), " = $(β1)"))
        lines!(ax, ts_log, abs.(R2_log); color = :crimson, label = rich(mit("β"), " = $(β2)"))
        axislegend(ax; position = :lb)
    end
end

begin # * Panel 4 — Unconfined power spectra: β = 1.0 vs β = 0.5
    ax = Axis(
        # `plotspectrum!` sets its own xlabel/ylabel over these; this panel's frequency axis is
        # dimensionless, so it wants no unit anyway (cf. (g), which sets its label AFTER the call).
        gs[gi_spectrum]; xlabel = "Frequency", ylabel = "Power",
        title = "Unconfined spectrum", yticks = LogTicks(WilkinsonTicks(nyticks))
    )
    plotspectrum!(ax, s_hi; color = :cornflowerblue)
    plotspectrum!(ax, s_lo; color = :crimson)
    ax.limits = ((1.0e-1, 1.0e0), (nothing, nothing))
    axislegend(
        ax, [LineElement(color = :cornflowerblue), LineElement(color = :crimson)],
        [rich(mit("β"), " = $(β_hi)"), rich(mit("β"), " = $(β_lo)")]; position = :lb
    )
end

begin # * Right column — sample time series: effect of α, β, γ
    ts_colors = [:black, cornflowerblue, crimson, california]
    # The titles arrive from the data as plain strings ("... (α=2, β=1, γ=0)"); `mathify` splits the
    # Greek out so it is set in the math face like every other symbol in the figure.
    # Negative Outside: reach down past the panel rows (into the group boxes' label band)
    # so the stack fills the column; the top term cancels the first title's folded protrusion
    g_ts = f[1:2, 3] = GridLayout(; alignmode = Outside(0, 0, -60, -26))
    for (i, (s, c)) in enumerate(zip(ts_windows, configs))
        ax = Axis(
            g_ts[i, 1];
            title = mathify(c.label),
            titlesize = 11,
            titlealign = :right
        )
        lines!(ax, times(s), collect(s); color = ts_colors[i], linewidth = 2)
        hidedecorations!(ax)
        hidespines!(ax)
    end
    rowgap!(g_ts, 28)
end

# ──────────────────────────────────────────────────────────────────────────────
# Row 3 — model summary: scaling of the flat vs unimodal samplers
# ──────────────────────────────────────────────────────────────────────────────
begin # * Panel e --- mean absolute deviation
    # `Mixed(left = 0)` keeps this row's ~50 pt ylabel protrusion inside its own cell. Left to
    # protrude, it sets column 1's left edge for the whole figure and holds the boxed rows (and so
    # the vertical block labels) 50 pt off the page margin. The cost is that its axes no longer line
    # up with the ones above --- deliberate. Only the left side is switched: a full `Outside` would
    # pull the titles and xlabels inside the fixed 135 pt row as well and flatten the axes.
    g_sum = f[3, 1:3] = GridLayout(; alignmode = Makie.Mixed(left = 0))
    ax = Axis(
        g_sum[1, 1]; xlabel = unitlabel("Time lag", "s"), ylabel = "MAD", ylabelpadding = -2,
        xscale = log10, yscale = log10, title = "Superdiffusion", titlealign = :right, # clear the letter
        # (h)/(i)'s titles clear a colorbar band these two do not have, so they sit ~19 pt higher.
        # A wider titlegap lifts the title alone; the axis stays put.
        titlegap = 23,
        limits = ((1.0e-4, 1.0), (0.02, 3)),
        yticks = LogTicks(-2:0) # integer decades: the default lands on half-decades, whose labels
        # are wider and eat the width this narrow panel has least of
    )
    lines!(ax, τs, mads; label = "Unconfined")
    lines!(ax, τs, gmads; color = cucumber, label = "Unimodal")
    lines!(ax, τfit, mad_fit; color = :crimson, linestyle = :dash)
    text!( # fitted exponent beside its line; glow keeps it legible over the curves
        ax, 0.95, 0.05; text = rich(mit("a"), " = $(round(a_exponent, digits = 2))"),
        space = :relative, align = (:right, :bottom), color = :crimson,
        glowcolor = :white, glowwidth = 8
    )
    axislegend(ax; position = :lt, patchsize = (10, 10)) # colors mean the same in (e) and (f)
end

begin # * Panel f --- power spectral density
    ax = Axis(
        g_sum[1, 2]; title = "LRTCs", ylabelpadding = -2, titlegap = 23,
        xtickformat = x -> string.(round.(Int, x))
    )
    plotspectrum!(ax, psd)
    plotspectrum!(ax, gpsd; color = cucumber)
    lines!(ax, psd_fit_x, psd_fit_y; color = :red, linewidth = 2, linestyle = :dash)
    text!(
        ax, 0.05, 0.05; text = rich(mit("b"), " = $(round(b_exponent, digits = 2))"),
        space = :relative, align = (:left, :bottom), color = :red,
        glowcolor = :white, glowwidth = 8
    )
    ax.xlabel = unitlabel("Frequency", "Hz")
    ax.limits = ((1, 1000), (2.0e-6, 1.0e-1))
end

begin # * Panels g, h --- exponent maps over (α, β) for the flat sampler
    # These blocks share the summary row with (f)/(g), whose titles are `Axis` titles and so sit in
    # the band ABOVE the row, costing them no height. This title used to be a ROW of the cell, so it
    # came out of the map's height while that band went unused beside it --- which is why the maps
    # were squashed. Putting it in the colorbar cell's top protrusion moves it into that same band
    # and hands its height back to the map.
    #
    # (`Mixed(top = ...)` would be the direct way to reach into the band and throws
    # `Unknown AlignMode` for these blocks, which are determinable in that direction.)
    function exponent_map!(gp::GridLayout, X; levels, colormap, title, hidey = false)
        ax = Axis(
            gp[2, 1]; xlabel = mit("α"), ylabel = mit("β"),
            xgridvisible = false, ygridvisible = false, backgroundcolor = :gray88,
            limits = ((1.2, 2.0), (0.2, 1.0)), xticks = [1.2, 1.6, 2.0]
        )
        hidey && hideydecorations!(ax; grid = false) # shares (h)'s β axis
        p = contourf!(ax, X; levels, colormap, extendhigh = :auto, extendlow = :auto)
        contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash) # a = 1/2 boundary
        scatter!( # the working-regime point shared by the whole figure
            ax, [prms.α], [prms.β]; color = cucumber,
            markersize = 10, strokecolor = :white, strokewidth = 1
        )
        cb = Colorbar(
            gp[1, 1], p; vertical = false, height = 8, flipaxis = true,
            ticks = WilkinsonTicks(3), ticklabelsize = 9
        )
        lab = Label( # styled as an Axis title, above the colorbar's own tick labels
            gp[1, 1, Top()], title; font = :bold, tellwidth = false,
            fontsize = to_value(Makie.current_default_theme()[:Axis][:titlesize]),
            # read off the colorbar's protrusion, not guessed: both anchor at the cell edge, so
            # without this the title would land ON the tick labels rather than above them
            padding = lift(
                d -> (0.0f0, 0.0f0, d.outer.top + 3.0f0, 0.0f0),
                cb.layoutobservables.reporteddimensions
            )
        )
        rowgap!(gp, 5) # colorbar and map sit tight
        return ax, lab
    end
    gg = g_sum[1, 3] = GridLayout()
    gh = g_sum[1, 4] = GridLayout()
    axg, labg = exponent_map!(
        gg, ma; levels = range(0.25, 0.75, length = 10),
        colormap = darksunset, title = rich("Diffusion exp. ", mit("a"))
    )
    axh, labh = exponent_map!(
        gh, ms; levels = range(-2.0, -1.0, length = 10),
        colormap = lightsunset, title = rich("Spectral exp. ", mit("b")), hidey = true
    )
    linkyaxes!(axg, axh)
    # Centre each title on its AXIS rather than on its cell. The cell is the column; the axis' own
    # ylabel and ticks inset its drawn box within that column, so a cell-centred Label sits left of
    # the panel it titles (measured: 5.5 pt for (g), 8.2 pt for (h)). Corrected by translating the
    # Label, as `addlabels!` does for the panel letters --- a translation costs no layout, whereas
    # padding changes the box the Label is centred in and so does not move it 1:1.
    centre_on_axis!(lab, ax) = let b = lab.layoutobservables.computedbbox[], d = Fathom.drawnbox(ax)
        Makie.translate!(
            lab.blockscene,
            (d.origin[1] + d.widths[1] / 2) - (b.origin[1] + b.widths[1] / 2), 0, 0
        )
    end
    colsize!(g_sum, 3, Auto(1.18)) # map columns wide enough for their full-size titles
    colsize!(g_sum, 4, Auto(1.18))
    colgap!(g_sum, 10)
end

begin # * Equation banner --- eq:bifractional_neural_sampling, one-line form
    # Placed after all panels: `f[0, ...]` prepends a row and shifts existing rows down,
    # so any later `f[row, ...]` index would otherwise be off by one.
    eq = L"\textbf{bFNS:}\; {}^{C}D_t^{\beta}\, x = -\eta \nabla \tilde{V}_{\alpha} + \gamma p + \eta^{1/\alpha} \xi_{\alpha,\beta}\,, \quad \frac{dp}{dt} = -\gamma \nabla \tilde{V}_{\alpha}"
    # Two offsets the layout cannot see, both measured from the rendered ink. The cell (columns 1:3)
    # spans 16..552, so its centre is 284 against a page centre of 360; and MathTeX's bounding box
    # is right-heavy relative to its ink, putting the glyphs a further 14 left. Left padding shifts
    # the ink 1:1, hence 90. Re-measure whenever the string changes --- the "bFNS:" prefix alone
    # moved it 35. `tellwidth = false` so that a padding this large cannot start dictating column
    # widths.
    Label(f[0, 1:3], eq; fontsize = 22, padding = (90, 0, 12, 2), tellwidth = false)
end

begin # * Save figure
    # The summary row is pinned shorter than its Auto size (167 pt): (f)/(g) are the only portrait
    # panels in the figure and were at 0.69, and the height given up here goes to the rows above,
    # which also pushes the c/d block's bottom further clear of the trace column.
    rowsize!(f.layout, 3, Fixed(135))
    rowgap!(f.layout, 3, 28) # air between the c/d rectangle and (f)/(g)'s titles (was 18)
    colsize!(f.layout, 3, Relative(0.28)) # sample-trace column
    colgap!(f.layout, 2, 30) # air between the group boxes (which reach right) and the traces
    addlabels!(
        [
            gs[1], gs[2], gs[3], gs[4],
            g_ts[1, 1],                                    # (e): the trace stack as a block
            g_sum[1, 1], g_sum[1, 2], g_sum[1, 3], g_sum[1, 4],
        ], f;
        # (f) clears its full-width title. (i) sits between (h)'s title on its left and its own on
        # its right, so its offset splits the difference rather than clearing one side only.
        # (a)-(d) narrowed when the block labels took a column, putting their titles into the
        # letters; these restore ~12 pt of clearance. (f) clears its full-width title, and (i) sits
        # between (h)'s title and its own so its offset splits the difference.
        dx = [-10, -23, -16, -22, 0, -14, 0, 0, -4],
        # (h) and (i)'s titles now sit in the colorbar cell's top protrusion, ABOVE the letter's
        # natural anchor rather than below it, so the letter is lifted onto that line instead of
        # dropped onto it.
        dy = [0, 0, 0, 0, 0, 23, 23, 23, 23]
    )
    # After `addlabels!`: adding the letters re-solves the layout and moves the panels ~4 pt, so
    # centring measured before it is stale by exactly that much.
    centre_on_axis!(labg, axg)
    centre_on_axis!(labh, axh)
    # Same correction, vertically: a blocklabel is centred on its row's CELL, but the group box
    # reaches past that cell by different amounts above and below (38 pt up, 54 or 66 down), so the
    # two centres differ by 8--14 pt.
    centre_on_box!(lab_space, box_space)
    centre_on_box!(lab_time, box_time)
    wsave(joinpath(outdir, "$NAME.pdf"), f)
    wsave(joinpath(outdir, "$NAME.svg"), f)
    wsave(joinpath(outdir, "$NAME.png"), f)
    f |> display
end

begin # * Save source data
    """
    Write one tsv per panel of `$NAME` into the figure's own directory, reusing the
    arrays handed to each plot call so the saved data is exactly what was drawn.
    """
    function save_source_data()
        mkpath(outdir)
        savedir(x) = joinpath(outdir, x)

        writedlm( # a: potential and effective potential over the plotted grid
            savedir("panelA_potential.tsv"),
            vcat(["x" "V" "V_effective"], hcat(collect(xs), Vs, Ṽs)), '\t'
        )

        if inset_log # b: analytic folded stable densities
            writedlm(
                savedir("panelB_stepsize.tsv"),
                vcat(
                    ["abs_x" "alpha_$α1" "alpha_$α2"],
                    hcat(collect(xs_pdf), p1, p2)
                ), '\t'
            )
        else # b: binned densities, as `ziggurat!` computes them
            edges = collect(bins)
            zig(x) = StatsBase.normalize(
                StatsBase.fit(StatsBase.Histogram, x, edges); mode = :pdf
            ).weights
            writedlm(
                savedir("panelB_stepsize.tsv"),
                vcat(
                    ["bin_lower" "bin_upper" "alpha_$α1" "alpha_$α2"],
                    hcat(edges[1:(end - 1)], edges[2:end], zig(xs_a1), zig(xs_a2))
                ), '\t'
            )
        end

        writedlm( # c: relaxation functions (t = 0 dropped for the log axis)
            savedir("panelC_relaxation.tsv"),
            vcat(
                ["t" "beta_$β1" "beta_$β2"],
                hcat(collect(ts_log), R1_log, abs.(R2_log))
            ), '\t'
        )

        @assert freqs(s_hi) == freqs(s_lo)
        writedlm( # d: peak-normalised spectra over the plotted band
            savedir("panelD_spectrum.tsv"),
            vcat(
                ["frequency" "beta_$β_hi" "beta_$β_lo"],
                hcat(collect(freqs(s_hi)), collect(s_hi), collect(s_lo))
            ), '\t'
        )

        writedlm( # e: MAD curves over the plotted lags
            savedir("panelE_mad.tsv"),
            vcat(
                ["tau" "unconfined" "unimodal"],
                hcat(collect(τs), collect(mads), collect(gmads))
            ), '\t'
        )

        @assert freqs(psd) == freqs(gpsd)
        writedlm( # f: power spectral densities over the plotted band
            savedir("panelF_psd.tsv"),
            vcat(
                ["frequency" "unconfined" "unimodal"],
                hcat(collect(freqs(psd)), collect(psd), collect(gpsd))
            ), '\t'
        )
        writedlm( # e, f: fitted scaling exponents quoted in the legends
            savedir("panelEF_exponents.tsv"),
            [["a" "b"]; [a_exponent b_exponent]], '\t'
        )

        writegrid(savedir("panelG_diffusion_exponent.tsv"), ma) # g, h: (α × β) grids, α down the rows
        writegrid(savedir("panelH_spectral_exponent.tsv"), ms)

        vs = map(configs) do c # right column: the four windowed traces, shared time base
            Symbol("alpha$(c.α)_beta$(c.β)_gamma$(c.γ)")
        end
        savetimeseries(
            savedir("timeseries.tsv"),
            Timeseries(hcat(collect.(ts_windows)...), times(first(ts_windows)), vs)
        )
        return @info "wrote source data" outdir
    end
    save_source_data()
end
