#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Circuit exponents across the δ/Δg_K plane.
#
# The δ/τ_d bottom row of hierarchical_variation.jl, re-drawn for the second circuit plane stored in
# circuit_exponents.jld2: the adaptation conductance Δg_K against the E/I ratio δ. Same reduction
# (median over seeds and neurons per cell), same line-per-second-parameter presentation, so the two
# planes can be read side by side. Unlike the τ_d plane this grid has no empty cells and no a > 1
# artefact, so nothing is masked.

using DrWatson
@quickactivate "WorkingRegime"
using JLD2
using TimeseriesTools
using CairoMakie
using Fathom
using Statistics
using DelimitedFiles

set_theme!(fathom())

const NAME = "dgk_variation"
const PATH = plotsdir(NAME)

const N_GK_LINES = 5      # lines per panel, evenly spaced across the swept Δg_K values
const HEATMAP = sunrise
const δlab = "δ  (I:E ratio)"
const gklab = "Δg_K  (mS/cm²)"

const circuit_path = projectdir("WRCircuit", "data", "circuit_exponents.jld2")
circuit = jldopen(
    f -> Dict(k => f[k] for k in keys(f)), circuit_path;
    typemap = toolsarray_typemap
)

"NaN-aware median of one cell's per-neuron exponents POOLED across all seeds (the trailing grid
axis). Each `(δ, Δg_K, seed)` cell is either a plain `Vector{Float64}` (the empty-cell sentinel) or a
typemap-upgraded `ToolsArray`; `collect` normalises both. Empty cells stay NaN."
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

# `invokelatest` for Julia 1.12's stricter world-age rules on top-level globals.
const A_dg = Base.invokelatest(seed_pooled_median, parent(circuit["a_dg"]))
const B_dg = Base.invokelatest(seed_pooled_median, parent(circuit["b_dg"]))
const δ_lookup = Float64.(collect(circuit["delta"]))
const gk_lookup = Float64.(collect(circuit["Delta_g_K"]))
const δ_0 = Float64(circuit["delta_0"])
const gk_0 = Float64(circuit["Delta_g_K_0"])

"Indices of the Δg_K values the figure draws: `n` evenly spaced across the sweep. Shared by the
drawing and the saved source data so the two cannot disagree."
drawn_gks(n = N_GK_LINES) = unique(round.(Int, range(1, length(gk_lookup); length = n)))

"""
    circuit_lines!(pos, grid; ylabel, title)

Exponent against δ, one line per drawn Δg_K value. The colorbar is categorical and ticked with the
Δg_K values actually drawn, so it is a legend for the lines rather than a continuous scale over
values that are not shown.
"""
function circuit_lines!(pos, grid; ylabel, title, n = N_GK_LINES)
    ax = Axis(pos[1, 1]; xlabel = δlab, ylabel = ylabel, title = title)
    sel = drawn_gks(n)
    cols = cgrad(HEATMAP, max(2, length(sel)); categorical = true)
    for (i, j) in enumerate(sel)
        v = grid[:, j]
        k = findall(!isnan, v)
        isempty(k) && continue
        lines!(ax, δ_lookup[k], v[k]; color = cols[i], linewidth = 2.5)
    end
    Colorbar(
        pos[1, 2]; colormap = cols, limits = (0, length(sel)),
        ticks = ((1:length(sel)) .- 0.5, string.(gk_lookup[sel])),
        label = gklab, width = 12
    )
    return ax
end

begin # * Render
    f = TwoPanel()
    gs = subdivide(f, 1, 2)
    ax_a = circuit_lines!(
        gs[1], A_dg; ylabel = "Diffusion exponent  a", title = "Circuit:  a vs δ"
    )
    ax_b = circuit_lines!(
        gs[2], B_dg; ylabel = "Spectral exponent  b", title = "Circuit:  b vs δ"
    )
    addlabels!(f)
    display(f)
end

begin # * Report the spread each knob accounts for, so the figure's claim is checkable
    for (sym, G) in (("a", A_dg), ("b", B_dg))
        across_gk = median(maximum(G[i, :]) - minimum(G[i, :]) for i in eachindex(δ_lookup))
        across_δ = median(maximum(G[:, j]) - minimum(G[:, j]) for j in eachindex(gk_lookup))
        @info "$sym: median span $(round(across_gk; digits = 3)) across Δg_K, " *
            "$(round(across_δ; digits = 3)) across δ"
    end
    @info "Working point: δ_0 = $δ_0, Δg_K_0 = $gk_0"
end

begin # * Save figure
    wsave(PATH * ".pdf", f)
    wsave(PATH * ".png", f)
    @info "Saved $PATH"
end

begin # * Save source data --- δ against the exponent, one column per Δg_K drawn
    mkpath(PATH)
    sel = drawn_gks()
    for (name, grid) in (("panelA", A_dg), ("panelB", B_dg))
        writedlm(
            joinpath(PATH, "$name.tsv"),
            vcat(
                hcat("delta", permutedims(["Delta_g_K=$(gk_lookup[j])" for j in sel])),
                hcat(δ_lookup, grid[:, sel])
            ), '\t'
        )
    end
    @info "Saved source data to $PATH"
end
