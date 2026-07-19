#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
using SpecialFunctions
using MittagLeffler
WRTheory.@preamble()
using Random
import FractionalNeuralSampling.Samplers: gen_lfsm_fns
import FractionalNeuralSampling: lfsn
set_theme!(foresight(:physics))

begin # * Options
    inset_log = false  # true: log-log as inset over linear plot; false: log-log as full axis
    order_rows = true  # true: rows = {space, time}; false: columns = {space, time}
end

begin # * Shared parameters
    tspan = 5000.0
    dt = 0.1
    seed = 44
    τ = 1000.0
    nyticks = 4
end

begin # * Set up figure
    f = FourPanel(; size = (720, 480))
    gs = subdivide(f[1:2, 1:2], 2, 2)
    # Panel indices: potential, step-size, relaxation, spectrum
    if order_rows
        # Row 1 = space (potential, step-size), Row 2 = time (relaxation, spectrum)
        gi_potential, gi_stepsize, gi_relax, gi_spectrum = 1, 2, 3, 4
    else
        # Col 1 = space (potential, step-size), Col 2 = time (relaxation, spectrum)
        gi_potential, gi_stepsize, gi_relax, gi_spectrum = 1, 3, 2, 4
    end
end

begin # * Panel 1 — Effective potential for unimodal density
    α_space = 1.5
    β_time = 0.85
    shared_params = (;
                     α = α_space,
                     β = β_time,
                     γ = 0.03,
                     η = 0.01,
                     domain = -10 .. 10,
                     boundaries = PeriodicBox(-5 .. 5),
                     u0 = [0.0, 0.0],
                     dt,
                     tspan,
                     approx_n_modes = 1000,
                     τ,
                     seed,
                     λ = 1e-4)

    noise = gen_lfsm_fns(shared_params.α, shared_params.β;
                         tspan = shared_params.tspan,
                         dt = shared_params.dt, seed = shared_params.seed,
                         nhist = round(Int, shared_params.τ / shared_params.dt))

    𝜋 = test_density(:unimodal)
    ps = (; shared_params..., 𝜋)
    S = bFNS(; ps..., noise)

    function effective_potential(S)
        ps, 𝜋 = S.p
        @unpack ∇𝒟𝜋, 𝜋s, λ = ps
        ∇V = x -> ∇𝒟𝜋(x) / (𝜋s(x) + λ)
        function V(xs::AbstractRange)
            dx = step(xs)
            xs = range(first(xs) - dx / 4, stop = last(xs) + dx / 4, step = dx)
            ∇Vs = .-∇V.(xs)
            V = cumsum(∇Vs) .* dx
            V .-= minimum(V)
            return V
        end
    end

    prange = (-1.1, 1.1)
    _xs = range(prange..., length = 1000)
    xs = _xs[1:10:end]

    V = potential(Density(S))
    Vs = V.(xs)
    Vs = Vs .- minimum(Vs)

    Ṽ = effective_potential(S)
    Ṽs = Ṽ(_xs)
    idxs = indexin(xs, _xs) |> Vector{Int}
    Ṽs = Ṽs[idxs]
    Ṽs .-= minimum(Ṽs)

    ax = Axis(gs[gi_potential], xlabel = "x", ylabel = "V(x)",
              limits = (extrema(xs), (-0.5, maximum(Vs))), title = "Potential function",
              yticks = WilkinsonTicks(nyticks))
    lines!(ax, xs, Vs; color = :cornflowerblue, label = "Potential")
    lines!(ax, xs, Ṽs; color = :crimson,
           label = "Effective potential")
    axislegend(ax; position = :ct, title = "α = $(α_space)")
end

begin # * Panel 2 — Step-size distribution: α = 2 (Gaussian) vs α = 1.5 (heavy-tailed)
    α1 = 2.0
    α2 = 1.5

    xs_pdf = range(0, 50, length = 500)

    # Stable(α, skewness=0, scale=1, location=0); fold onto positive axis: p(|x|) = 2*pdf(x)
    d1 = Stable(α1, 0, 1, 0)
    d2 = Stable(α2, 0, 1, 0)

    p1 = 2 .* pdf.(d1, xs_pdf)
    p2 = 2 .* pdf.(d2, xs_pdf)

    if inset_log
        ax = Axis(gs[gi_stepsize], xlabel = "|x|", ylabel = "p(|x|)",
                  title = "Step-size distribution",
                  limits = ((0, 6), (0, nothing)),
                  yticks = WilkinsonTicks(nyticks))
        lines!(ax, xs_pdf, p1; color = :cornflowerblue, label = "α = $(α1)")
        lines!(ax, xs_pdf, p2; color = :crimson, label = "α = $(α2)")
        axislegend(ax; position = :lb)

        # Inset: log-log view of the pdf
        p1_log = 2 .* pdf.(d1, xs_pdf)
        p2_log = 2 .* pdf.(d2, xs_pdf)

        ax_inset = Axis(gs[gi_stepsize]; width = Relative(0.5), height = Relative(0.5),
                        halign = 0.95, valign = 0.9,
                        xscale = log10, yscale = log10,
                        xlabelsize = 10, ylabelsize = 10,
                        xticklabelsize = 8, yticklabelsize = 8,
                        backgroundcolor = :white,
                        limits = ((nothing, 25), (1e-4, 1e0)),
                        yticks = LogTicks(WilkinsonTicks(nyticks - 1)))
        lines!(ax_inset, xs_pdf, p1_log; color = :cornflowerblue)
        lines!(ax_inset, xs_pdf, p2_log; color = :crimson)
        translate!(ax_inset.blockscene, 0, 0, 100)
    else
        bins = logrange(1, 40, length = 20)
        ax = Axis(gs[gi_stepsize]; xlabel = "|x|", ylabel = "p(|x|)",
                  title = "Step-size distribution",
                  xscale = log10, yscale = log10,
                  limits = ((1, 25), (1e-4, nothing)), #(1e-4, 1e0)),
                  yticks = LogTicks(WilkinsonTicks(nyticks)))

        xs = abs.(rand(d2, 100000))
        ziggurat!(ax, xs; linecolor = :crimson, label = "α = $(α2)", bins,
                  normalization = :pdf, fillalpha = 0.3, color = brighten(crimson, 0.5))

        xs = abs.(rand(d1, 100000))
        ziggurat!(ax, xs; linecolor = :cornflowerblue, label = "α = $(α1)", bins,
                  normalization = :pdf, fillalpha = 0.3,
                  color = brighten(cornflowerblue, 0.5))
        axislegend(ax; position = :rt)
    end
end

begin # * Panel 3 — Relaxation function: β = 1 (exponential) vs β < 1 (Mittag-Leffler)
    ts = range(0, 50, length = 500)

    β1 = 1.0
    β2 = 0.5

    R1 = exp.(-ts)
    R2 = mittleff.(β2, .-ts)

    ts_log = ts[2:end]
    R1_log = R1[2:end]
    R2_log = R2[2:end]

    if inset_log
        ax = Axis(gs[gi_relax], xlabel = "Time", ylabel = "R(t)",
                  title = "Relaxation function", limits = ((0, 5), (0, 1.05)))
        lines!(ax, ts, R1; color = :cornflowerblue, label = "β = $(β1)")
        lines!(ax, ts, R2; color = :crimson,
               label = "β = $(β2)")
        hlines!(ax, [0]; color = :gray, linewidth = 0.5)
        axislegend(ax; position = :lb)

        # Inset: log-log view (skip t = 0 to avoid log(0))
        ax_inset = Axis(gs[gi_relax]; width = Relative(0.5), height = Relative(0.5),
                        halign = 0.95, valign = 0.9,
                        xscale = log10, yscale = log10,
                        xlabelsize = 10, ylabelsize = 10,
                        xticklabelsize = 8, yticklabelsize = 8,
                        backgroundcolor = :white,
                        limits = ((minimum(ts_log), maximum(ts_log)), (1e-2, 1e0 + 0.5)),
                        yticks = LogTicks(WilkinsonTicks(nyticks)))
        lines!(ax_inset, ts_log, R1_log; color = :cornflowerblue)
        lines!(ax_inset, ts_log, abs.(R2_log); color = :crimson)
        translate!(ax_inset.blockscene, 0, 0, 100)
    else
        ax = Axis(gs[gi_relax]; xlabel = "t", ylabel = "R(t)",
                  title = "Relaxation function",
                  xscale = log10, yscale = log10,
                  limits = ((minimum(ts_log), maximum(ts_log)), (1e-2, 1e0 + 0.5)),
                  yticks = LogTicks(WilkinsonTicks(nyticks)))
        lines!(ax, ts_log, R1_log; color = :cornflowerblue, label = "β = $(β1)")
        lines!(ax, ts_log, abs.(R2_log); color = :crimson, label = "β = $(β2)")
        axislegend(ax; position = :lb)
    end
end

begin # * Panel 4 — Power spectrum of tFOLE (Caputo-integrated): β = 1.0 vs β = 0.75
    β_hi = 1.0
    β_lo = 0.5
    η_psd = 0.1
    γ_psd = 0.1
    dt_psd = 0.25

    tspan_psd = 50000.0

    𝜋_psd = Normal(0, 10.0) |> Density
    noise_hi = gen_lfsm_fns(2.0, β_hi; tspan = tspan_psd, dt = dt_psd, seed,
                            nhist = round(Int, τ / dt_psd))
    noise_lo = gen_lfsm_fns(2.0, β_lo; tspan = tspan_psd, dt = dt_psd, seed,
                            nhist = round(Int, τ / dt_psd))
    S_hi = bFNS(; η = η_psd, β = β_hi, γ = γ_psd, α = 2.0, u0 = [0.0, 0.0],
                tspan = tspan_psd, dt = dt_psd, seed, 𝜋 = 𝜋_psd, noise = noise_hi,
                domain = -50 .. 50,
                approx_n_modes = 1000, τ)
    S_lo = bFNS(; η = η_psd, β = β_lo, γ = γ_psd, α = 2.0, u0 = [0.0, 0.0],
                tspan = tspan_psd, dt = dt_psd, seed, 𝜋 = 𝜋_psd, noise = noise_lo,
                domain = -50 .. 50,
                approx_n_modes = 1000, τ)

    sol_hi = solve(S_hi) |> Timeseries |> eachcol |> first
    sol_lo = solve(S_lo) |> Timeseries |> eachcol |> first

    sol_hi = rectify(sol_hi, dims = 𝑡; tol = 1)
    sol_lo = rectify(sol_lo, dims = 𝑡; tol = 1)

    n = (length(sol_hi) ÷ 2 + 1)
    sol_hi = sol_hi[n:end]
    sol_lo = sol_lo[n:end]

    s_hi = spectrum(sol_hi .- mean(sol_hi), 0.05)
    s_lo = spectrum(sol_lo .- mean(sol_lo), 0.05)

    s_hi = s_hi[𝑓 = 1e-1 .. 1e0]
    s_lo = s_lo[𝑓 = 1e-1 .. 1e0]

    s_hi = s_hi ./ maximum(s_hi)
    s_lo = s_lo ./ maximum(s_lo)

    ax = Axis(gs[gi_spectrum]; xlabel = "Frequency (Hz)", ylabel = "Power",
              title = "Unconfined spectrum", yticks = LogTicks(WilkinsonTicks(nyticks)))
    plotspectrum!(ax, s_hi; label = "β = $(β_hi)", color = :cornflowerblue)
    plotspectrum!(ax, s_lo; label = "β = $(β_lo)", color = :crimson)
    ax.limits = ((1e-1, 1e0), (nothing, nothing))
    axislegend(ax; position = :lb)
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Effect of α, β, γ on sample time series
# ──────────────────────────────────────────────────────────────────────────────
begin
    begin # * Time-series parameters
        ts_tspan = 5.0
        ts_dt = 0.01
        ts_seed = 42
        ts_τ = 1000.0
        ts_η = 0.01
        ts_𝜋 = test_density(:unimodal)
        ts_domain = -5 .. 5
        ts_boundaries = PeriodicBox(-3 .. 3)
        ts_u0 = [0.0, 0.0]
        ts_approx_n_modes = 1000
        ts_λ = 1e-4

        α_mod = 1.6
        β_mod = 0.5
        γ_mod = 4

        ts_common = (;
                     domain = ts_domain,
                     boundaries = ts_boundaries,
                     u0 = ts_u0,
                     dt = ts_dt,
                     tspan = ts_tspan,
                     approx_n_modes = ts_approx_n_modes,
                     τ = ts_τ,
                     seed = ts_seed,
                     λ = ts_λ,
                     η = ts_η,
                     𝜋 = ts_𝜋)

        # Four configurations, each progressively changing one parameter
        configs = [
            (label = "Standard diffusion\n(α=2, β=1, γ=0)",
             α = 2.0, β = 1.0, γ = 0.0),
            (label = "+ Lévy superdiffusion\n(α=$(α_mod), β=1, γ=0)",
             α = α_mod, β = 1.0, γ = 0.0),
            (label = "+ LRTC subdiffusion\n(α=$(α_mod), β=$(β_mod), γ=0)",
             α = α_mod, β = β_mod, γ = 0.0),
            (label = "+ Oscillations\n(α=$(α_mod), β=$(β_mod), γ=$(γ_mod))",
             α = α_mod, β = β_mod, γ = γ_mod)
        ]
    end

    begin # * Generate time series
        ts_sols = map(configs) do c
            noise = gen_lfsm_fns(c.α, c.β;
                                 tspan = ts_tspan, dt = ts_dt, seed = ts_seed,
                                 nhist = round(Int, ts_τ / ts_dt))
            S = bFNS(; ts_common..., α = c.α, β = c.β, γ = c.γ, noise)
            sol = solve(S) |> Timeseries |> eachcol |> first
            rectify(sol, dims = 𝑡; tol = 1)
        end
    end

    begin # * Set up and plot figure 2
        nr = length(configs)
        # Show only a representative window
        t_show = 2 .. ts_tspan

        ts_colors = [:black, cornflowerblue, crimson, california]

        axs = []

        for (i, (sol, c)) in enumerate(zip(ts_sols, configs))
            s = sol[𝑡 = t_show]
            ax = Axis(f[:, 3][i, 1];
                      title = c.label,
                      titlesize = 11,
                      titlealign = :right)
            lines!(ax, times(s), collect(s); color = ts_colors[i], linewidth = 2)
            hidedecorations!(ax)
            hidespines!(ax)
            push!(axs, ax)
        end
    end
end

begin # * Save figure
    colsize!(f.layout, 3, Relative(0.2))
    addlabels!(f, ["(a)", "(b)", "", "(c)", "(d)", "", "", ""])
    wsave(plotdir("fractional_operator_summary", "fractional_operator_summary.pdf"), f)
    f |> display
end
