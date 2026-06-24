#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Sanity check for the diffusion-exponent fit.
#
# Pulls a random MAD curve from the circuit sweep, fits the same 2-component
# MAPPLE model used in critical_sweep.jl, and reports fit quality three ways:
#   1. visual overlay of data vs. fitted curve (log-log),
#   2. R² of the fit in log-log space (where a power law is linear),
#   3. residual structure across the fit window.
#
# Run repeatedly (it reseeds each time) to spot-check that the recovered
# diffusion exponent is backed by a faithful fit, not an artefact.

using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
using MoreMaps
using Statistics
using Random
using Optim
using ForwardDiff
WRCircuit.@preamble
set_theme!(foresight(:physics))

# The diffusion exponent is the slope of MAPPLE's *first* component, so the fit
# window is read from the fitted model (the first component's domain) rather than
# hardcoded — units-agnostic, and matches the region the reported exponent
# governs. NB the circuit MAD lag axis is in ms (values ~1–1000), not seconds, so
# a hardcoded (1e-3, 1e-2) s window falls entirely off the data → empty → NaN R².

# ──────────────────────────────────────────────────────────────────────────────
# Pick a random sweep file and a random neuron within it
# ──────────────────────────────────────────────────────────────────────────────

Random.seed!(4912121213333)

begin
    files = filter(readdir(datadir("critical_sweep"); join = true)) do f
        fname = parse_savename(f; connector = string(connector))[2]
        !haskey(fname, "key") && haskey(fname, "delta") && haskey(fname, "Delta_g_K")
    end

    file = rand(files)
    params = parse_savename(file; connector = string(connector))[2]
    @info "Selected file" delta = params["delta"] Delta_g_K = params["Delta_g_K"]

    mad = load(file, "inputs/mad")
    n_neurons = size(mad, 2)
    neuron = rand(1:n_neurons)
    @info "Selected neuron $neuron / $n_neurons"

    # Single neuron's MAD curve, units stripped (as in the fitting recipe)
    y = ustripall(mad[:, neuron])
end

# ──────────────────────────────────────────────────────────────────────────────
# Fit the MAPPLE model (matching critical_sweep.jl: 2 components, 0 peaks)
# ──────────────────────────────────────────────────────────────────────────────

begin
    m = fit(MAPPLE, y; components = 2, peaks = 0)
    fit!(m, y)
    diffusion_exponent = first(m.params.components.β)
    @info "Recovered diffusion exponent a = $(round(diffusion_exponent; sigdigits = 3))"

    t = lookup(y, 𝑡)
    ŷ = predict(m, t)
end

# ──────────────────────────────────────────────────────────────────────────────
# Quantitative fit quality — log-log R² and residuals over the fit window
# ──────────────────────────────────────────────────────────────────────────────

# Log-log R²: a power law is a straight line in log-log, so R² of log data vs.
# log fit is the natural goodness measure. Computed over a given index mask.
function loglog_r2(y, ŷ, mask)
    logy = log10.(collect(y)[mask])
    logŷ = log10.(collect(ŷ)[mask])
    res = logy .- logŷ
    r2 = 1 - sum(abs2, res) / sum(abs2, logy .- mean(logy))
    rmse = sqrt(mean(abs2, res))
    return r2, rmse, res
end

begin
    # Window = first MAPPLE component's domain, i.e. lags ≤ 10^(log_f_stop) of the
    # component whose β is the reported diffusion exponent. Clamp into the data
    # range; fall back to the whole curve if the breakpoint is degenerate.
    win_hi = 10^first(m.params.components.log_f_stop)
    if !(minimum(t) < win_hi <= maximum(t))
        win_hi = maximum(t)
    end
    FIT_WINDOW = (minimum(t), win_hi)
    in_window = FIT_WINDOW[1] .<= t .<= FIT_WINDOW[2]
    tw = t[in_window]

    # Two complementary numbers: the windowed R² tests the exponent's regime; the
    # whole-curve R² tests whether the 2-component model reproduces the full MAD.
    r2, rmse, residuals_w = loglog_r2(y, ŷ, in_window)
    r2_full, rmse_full, _ = loglog_r2(y, ŷ, trues(length(t)))

    @info "Fit quality (log-log)" window_ms = round.(FIT_WINDOW; sigdigits = 3) R²_window = round(r2; sigdigits = 4) R²_whole = round(r2_full; sigdigits = 4) RMSE_window = round(rmse; sigdigits = 3)
end

# ──────────────────────────────────────────────────────────────────────────────
# Visual check — overlay + residuals
# ──────────────────────────────────────────────────────────────────────────────

begin
    f = TwoPanel()

    # Panel 1: data vs. fit on log-log axes, with the diffusion window shaded
    ax1 = Axis(
        f[1, 1]; xscale = log10, yscale = log10,
        xlabel = "Time lag (ms)", ylabel = "MAD",
        title = "δ=$(params["delta"]), Δg_K=$(params["Delta_g_K"]), neuron $neuron"
    )
    vspan!(ax1, FIT_WINDOW...; color = (:gray, 0.15))
    lines!(ax1, t, collect(y); color = cornflowerblue, label = "Data")
    lines!(ax1, t, collect(ŷ); color = crimson, linestyle = :dash, label = "MAPPLE fit")
    text!(
        ax1, FIT_WINDOW[1], maximum(y);
        text = "a = $(round(diffusion_exponent; sigdigits = 3))\nR² (window) = $(round(r2; sigdigits = 3))\nR² (whole) = $(round(r2_full; sigdigits = 3))",
        align = (:left, :top), fontsize = 14
    )
    axislegend(ax1; position = :rb)

    # Panel 2: log-log residuals within the diffusion window (flat = good)
    ax2 = Axis(
        f[1, 2]; xscale = log10,
        xlabel = "Time lag (ms)", ylabel = "log₁₀ residual",
        title = "Fit residuals (window only)"
    )
    hlines!(ax2, [0]; color = :gray, linestyle = :dash)
    scatterlines!(ax2, tw, residuals_w; color = crimson, markersize = 6)

    display(f)

    OUTDIR = plotdir("diffusion_exponent_test")
    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, savename(params, "pdf"; connector = string(connector)))
    wsave(outfile, f)
    @info "Saved $outfile"
end
