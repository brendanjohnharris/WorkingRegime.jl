#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate :WorkingRegime
using CairoMakie
using Fathom
using TimeseriesTools
using TimeseriesMakie
using Statistics
using DelimitedFiles
import StatsBase
import Fathom: cornflowerblue, cucumber # Foresight-era names, unexported by Fathom

set_theme!(Fathom.fathom())

# Two supplementary summaries of the confined bFNS samplers, at the working point Fig2 uses:
# FigS2 (unimodal) and FigS3 (bimodal). Both read Fig2's data file --- the unimodal sampler is
# already simulated there --- and neither repeats the unconfined comparison Fig2 and Fig3 carry.

begin # * Options
    NAME = "FigS23_summaries"
    outdir = plotsdir(NAME)
    datafile = datadir("WRTheory", "bFNS_data.jld2") # shared with Fig2
    nyticks = 4
    MAP_DY = -29 # see the `addlabels!` calls: drops a map letter onto its Label title's line
    # Drawn spacing is `gap + protrusion`, and the band above the map row is nearly all
    # protrusion: the row above reserves for its x-label, the map row for its letters and title
    # Label. A negative gap reaches through that, which is the only height the maps can gain
    # without narrowing them. Stop short of eating the whole band: at -40 the row above brings
    # its x-label down onto the letter line.
    MAP_ROWGAP = -20
    # A map letter sits in its cell's TopLeft protrusion, so the first map --- which keeps its β
    # decorations --- gets 51 pt of margin for free while the other two, whose decorations are
    # hidden, get ~28 and would sit on top of their centred title. MAP_DX pushes those two out to
    # the same 51 pt, and MAP_COLGAP opens the room for them to move into.
    MAP_DX = -23
    MAP_COLGAP = 34
end

begin # * Load data (produced by WRTheory/scripts/bFNS_data.jl)
    data = wload(datafile)
    xs, bins = data["xs"], data["bins"]
    # the unimodal potential is Fig2's panel (a); the supplement redraws it beside its density
    uni_V, uni_Ṽ = data["Vs"], data["Ṽs"]
    uni_samples, uni_target = data["uni_samples"], data["uni_target"]
    uni_accuracy = data["uni_accuracy"]
    uni_ma, uni_ms, uni_macc = data["uni_ma"], data["uni_ms"], data["uni_macc"]

    bi_V, bi_Ṽ = data["bi_V"], data["bi_Ṽ"]
    bi_samples, bi_target = data["bi_samples"], data["bi_target"]
    bi_accuracy = data["bi_accuracy"]
    τs, τfit = data["τs"], data["τfit"] # lag grid shared with Fig2's MAD panel
    bi_mads, bi_mad_fit = data["bi_mads"], data["bi_mad_fit"]
    bi_a_exponent = data["bi_a_exponent"]
    bi_psd = data["bi_psd"]
    bi_psd_fit_x, bi_psd_fit_y = data["bi_psd_fit_x"], data["bi_psd_fit_y"]
    bi_b_exponent = data["bi_b_exponent"]
    bi_ma, bi_ms, bi_macc = data["bi_ma"], data["bi_ms"], data["bi_macc"]

    prms = data["params"]
    α, β, η = prms.α, prms.β, prms.η
end

# ──────────────────────────────────────────────────────────────────────────────
# Shared panel builders — the two figures differ only in which sampler they draw
# ──────────────────────────────────────────────────────────────────────────────

"Potential and effective potential, styled as Fig2's panel (a)."
function potential_panel!(gp, Vs, Ṽs; α)
    ax = Axis(
        gp; xlabel = "x", ylabel = "V(x)", title = "Potential function",
        limits = (extrema(xs), (-0.5, maximum(Vs))), yticks = WilkinsonTicks(nyticks)
    )
    lines!(ax, xs, Vs; color = :cornflowerblue, label = "Potential")
    lines!(ax, xs, Ṽs; color = :crimson, label = "Effective potential")
    axislegend(ax; position = :ct, title = "α = $α")
    return ax
end

"""
Sampled distribution against its target. `Δ` is the sampling accuracy, quoted in the legend:
it is the panel's quantitative claim, and it is not recoverable from the drawn curves.
"""
function density_panel!(gp, samples, target; Δ)
    ax = Axis(
        gp; xlabel = "x", ylabel = "𝜋(x)", title = "Distribution",
        limits = (extrema(xs), (0, 2.0)), yticks = WilkinsonTicks(nyticks; k_max = 5)
    )
    ziggurat!(
        ax, samples; normalization = :pdf, bins, fillalpha = 0.3,
        color = brighten(cornflowerblue, 0.5), strokecolor = :cornflowerblue,
        label = "Empirical (Δ = $(round(Δ, sigdigits = 2)))"
    )
    lines!(ax, xs, target; color = :crimson, label = "Target")
    l = axislegend(ax; position = :lt, orientation = :horizontal, patchsize = (10, 10))
    reverselegend!(l)
    return ax
end

"""
    exponent_map!(gp, X; levels, colormap, title, ...)

One (α, β) map with its title and horizontal colorbar stacked above it, as in Fig2's panels
(g, h). `ma` supplies the dashed a = 1/2 contour on every map, and the working point is marked
on all of them.
"""
function exponent_map!(
        gp, X; ma, title, levels = nothing, colormap = nothing,
        tickformat = Makie.automatic, hidey = false
    )
    Label(
        gp[1, 1], title; font = :bold, tellwidth = false, # don't let the title set the column width
        fontsize = Fathom.fathomfontsize() * 1.25, padding = (0, 0, 2, 0)
    )
    ax = Axis(
        gp[3, 1]; xlabel = "α", ylabel = "β",
        xgridvisible = false, ygridvisible = false, backgroundcolor = :gray88,
        limits = ((1.2, 2.0), (0.2, 1.0)), xticks = [1.2, 1.6, 2.0]
    )
    hidey && hideydecorations!(ax; grid = false)
    kw = (; extendhigh = :auto, extendlow = :auto)
    isnothing(levels) || (kw = (; kw..., levels))
    isnothing(colormap) || (kw = (; kw..., colormap))
    p = contourf!(ax, X; kw...)
    contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash) # a = 1/2 boundary
    scatter!( # the working-regime point this figure's sampler sits at
        ax, [α], [β]; color = cucumber, markersize = 10,
        strokecolor = :white, strokewidth = 1
    )
    Colorbar(
        gp[2, 1], p; vertical = false, height = 8, flipaxis = true,
        ticks = WilkinsonTicks(3), ticklabelsize = 9, tickformat
    )
    rowgap!(Makie.content(gp), 5) # title / colorbar / map sit tight
    return ax
end

"Superscript tick labels for the log₁₀ accuracy colorbar."
log_ticks(x) = [rich("10", superscript(string(round(v, digits = 2)))) for v in x]

"The three (α, β) maps shared by both figures: diffusion, spectral, and sampling accuracy."
function map_row!(g, ma, ms, macc)
    axa = exponent_map!(
        g[1, 1], ma; ma, title = "Diffusion exponent",
        levels = range(0.25, 0.75, length = 10), colormap = darksunset
    )
    axb = exponent_map!(
        g[1, 2], ms; ma, title = "Spectral exponent",
        levels = range(-2.0, -1.0, length = 10), colormap = lightsunset, hidey = true
    )
    axc = exponent_map!(
        g[1, 3], log10.(macc); ma, title = "Sampling accuracy",
        tickformat = log_ticks, hidey = true
    )
    linkyaxes!(axa, axb, axc)
    colgap!(g, MAP_COLGAP)
    return axa, axb, axc
end

"Parameter annotation for the scaling panels; a supplement should say where it was measured."
function param_note!(ax, γ; align = (:right, :bottom), pos = (0.95, 0.05))
    return text!(
        ax, pos...; space = :relative, align, fontsize = 9,
        text = "α = $α\nβ = $β\nγ = $γ\nη = $η",
        glowcolor = :white, glowwidth = 6
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# FigS2 — unimodal sampler
# ──────────────────────────────────────────────────────────────────────────────

begin # * Render FigS2
    fS2 = FourPanel()
    # Separate grids for the two rows: the top holds two panels and the bottom three, and they
    # are not meant to share column edges.
    g2_top = fS2[1, 1] = GridLayout()
    g2_map = fS2[2, 1] = GridLayout()

    potential_panel!(g2_top[1, 1], uni_V, uni_Ṽ; α)
    density_panel!(g2_top[1, 2], uni_samples, uni_target; Δ = uni_accuracy)
    map_row!(g2_map, uni_ma, uni_ms, uni_macc)

    # The map cells hold their title in a `Label` row inside the cell, not in an Axis-title
    # protrusion, so their letters sit a full title row high; MAP_DY drops them onto that line
    # (measured: the letter box bottoms out exactly on the title box top, 29 px of text centres).
    rowgap!(fS2.layout, 1, MAP_ROWGAP)
    addlabels!(
        [g2_top[1, 1], g2_top[1, 2], g2_map[1, 1], g2_map[1, 2], g2_map[1, 3]], fS2;
        dy = [0, 0, MAP_DY, MAP_DY, MAP_DY], dx = [0, 0, 0, MAP_DX, MAP_DX]
    )
    display(fS2)
end

# ──────────────────────────────────────────────────────────────────────────────
# FigS3 — bimodal sampler
# ──────────────────────────────────────────────────────────────────────────────

begin # * Render FigS3
    fS3 = SixPanel()
    g3_top = fS3[1, 1] = GridLayout()
    g3_mid = fS3[2, 1] = GridLayout()
    g3_map = fS3[3, 1] = GridLayout()

    potential_panel!(g3_top[1, 1], bi_V, bi_Ṽ; α)
    density_panel!(g3_top[1, 2], bi_samples, bi_target; Δ = bi_accuracy)

    begin # * Mean absolute deviation
        ax = Axis(
            g3_mid[1, 1]; xlabel = "Time lag (s)", ylabel = "MAD",
            xscale = log10, yscale = log10, title = "Superdiffusion"
        )
        lines!(ax, τs, bi_mads; label = "Bimodal")
        lines!(ax, τfit, bi_mad_fit; color = :crimson, linestyle = :dash)
        text!( # fitted exponent beside its line; glow keeps it legible over the curve
            ax, 0.95, 0.35; text = "a = $(round(bi_a_exponent, digits = 2))",
            space = :relative, align = (:right, :bottom), color = :crimson,
            glowcolor = :white, glowwidth = 8
        )
        param_note!(ax, prms.γ_bimodal)
    end

    begin # * Power spectral density
        ax = Axis(
            g3_mid[1, 2]; title = "LRTCs", xtickformat = x -> string.(round.(Int, x))
        )
        plotspectrum!(ax, bi_psd)
        lines!(ax, bi_psd_fit_x, bi_psd_fit_y; color = :red, linewidth = 2, linestyle = :dash)
        text!(
            ax, 0.05, 0.05; text = "b = $(round(bi_b_exponent, digits = 2))",
            space = :relative, align = (:left, :bottom), color = :red,
            glowcolor = :white, glowwidth = 8
        )
        param_note!(ax, prms.γ_bimodal; align = (:right, :top), pos = (0.95, 0.95))
        ax.xlabel = "Frequency (Hz)"
        ax.limits = ((1, 1000), (nothing, nothing))
    end

    map_row!(g3_map, bi_ma, bi_ms, bi_macc)

    rowgap!(fS3.layout, 2, MAP_ROWGAP) # only the map row; the two panel rows already sit tight
    addlabels!(
        [
            g3_top[1, 1], g3_top[1, 2], g3_mid[1, 1], g3_mid[1, 2],
            g3_map[1, 1], g3_map[1, 2], g3_map[1, 3],
        ], fS3;
        dy = [0, 0, 0, 0, MAP_DY, MAP_DY, MAP_DY], dx = [0, 0, 0, 0, 0, MAP_DX, MAP_DX]
    )
    display(fS3)
end

begin # * Save figures
    mkpath(outdir)
    for (name, fig) in (("FigS2_unimodal_summary", fS2), ("FigS3_bimodal_summary", fS3))
        wsave(joinpath(outdir, "$name.pdf"), fig)
        wsave(joinpath(outdir, "$name.svg"), fig)
        wsave(joinpath(outdir, "$name.png"), fig)
    end
    @info "wrote figures" outdir
end

begin # * Save source data
    """
    One tab-separated file per panel of both figures, into this script's own directory. Reuses
    the arrays handed to each plot call, so the files cannot drift from what was drawn. The
    ziggurat panels are saved as the binned densities the recipe computes, not as the raw
    samples, since the bins are part of what the panel claims.
    """
    function save_source_data()
        savedir(x) = joinpath(outdir, x)

        function writezig(path, samples) # binned density, as `ziggurat!` computes it
            w = StatsBase.normalize(
                StatsBase.fit(StatsBase.Histogram, samples, bins); mode = :pdf
            ).weights
            return writedlm(
                path,
                vcat(
                    ["bin_lower" "bin_upper" "pdf"],
                    hcat(bins[1:(end - 1)], bins[2:end], w)
                ), '\t'
            )
        end

        for (prefix, V, Ṽ, samples, target, Δ, ma, ms, macc) in (
                (
                    "FigS2", uni_V, uni_Ṽ, uni_samples, uni_target, uni_accuracy,
                    uni_ma, uni_ms, uni_macc,
                ),
                (
                    "FigS3", bi_V, bi_Ṽ, bi_samples, bi_target, bi_accuracy,
                    bi_ma, bi_ms, bi_macc,
                ),
            )
            writedlm( # a: potential and effective potential over the plotted grid
                savedir("$(prefix)_panelA_potential.tsv"),
                vcat(["x" "V" "V_effective"], hcat(xs, V, Ṽ)), '\t'
            )
            writezig(savedir("$(prefix)_panelB_empirical.tsv"), samples)
            writedlm( # b: target density on the same grid as the potential
                savedir("$(prefix)_panelB_target.tsv"),
                vcat(["x" "pdf"], hcat(xs, target)), '\t'
            )
            writedlm( # b: sampling accuracy quoted in the legend
                savedir("$(prefix)_accuracy.tsv"),
                [["sampling_accuracy"]; [Δ]], '\t'
            )
            writegrid(savedir("$(prefix)_diffusion_exponent.tsv"), ma) # (α × β) grids, α down the rows
            writegrid(savedir("$(prefix)_spectral_exponent.tsv"), ms)
            writegrid(savedir("$(prefix)_sampling_accuracy.tsv"), macc)
        end

        writedlm( # FigS3 c: MAD curve over the plotted lags, plus its fitted segment
            savedir("FigS3_panelC_mad.tsv"),
            vcat(["tau" "mad"], hcat(τs, bi_mads)), '\t'
        )
        writedlm(
            savedir("FigS3_panelC_mad_fit.tsv"),
            vcat(["tau" "mad_fit"], hcat(τfit, bi_mad_fit)), '\t'
        )
        writedlm( # FigS3 d: power spectrum over the plotted band, plus its fitted segment
            savedir("FigS3_panelD_psd.tsv"),
            vcat(["frequency" "psd"], hcat(collect(freqs(bi_psd)), collect(bi_psd))), '\t'
        )
        writedlm(
            savedir("FigS3_panelD_psd_fit.tsv"),
            vcat(["frequency" "psd_fit"], hcat(bi_psd_fit_x, bi_psd_fit_y)), '\t'
        )
        writedlm( # c, d: fitted scaling exponents quoted on the panels
            savedir("FigS3_exponents.tsv"),
            [["a" "b"]; [bi_a_exponent bi_b_exponent]], '\t'
        )
        return @info "wrote source data" outdir
    end
    save_source_data()
end
