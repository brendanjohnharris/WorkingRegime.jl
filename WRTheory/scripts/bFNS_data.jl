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
using ForwardDiff # with Optim, triggers TimeseriesTools' OptimExt; without both, MAPPLE `fit!` silently degrades
import FractionalNeuralSampling.Samplers: gen_lfsm_fns
import FFTW
FFTW.set_num_threads(1) # FFTW's own threads segfault (ip: nil) under `julia -t auto` on cartman; these 1-D FFTs lose nothing

# Produces rootdatadir("bFNS_data.jld2"), plotted by the top-level scripts/Fig2_bFNS.jl (the bFNS
# figure) and scripts/FigS23_summaries.jl (the unimodal and bimodal supplementary summaries).
# The three samplers --- unconfined, unimodal, bimodal --- share one noise realisation, so any
# panel that compares them differs only by the potential.

begin # * Shared parameters
    tspan = 5000.0
    dt = 0.1
    seed = 44
    τ = 1000.0
end

begin # * Potential and effective potential for the unimodal density
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
        λ = 1.0e-4,
    )

    noise = gen_lfsm_fns(
        shared_params.α, shared_params.β;
        tspan = shared_params.tspan,
        dt = shared_params.dt, seed = shared_params.seed,
        nhist = round(Int, shared_params.τ / shared_params.dt)
    )

    𝜋 = test_density(:unimodal)
    ps = (; shared_params..., 𝜋)
    S = bFNS(; ps..., noise)

    function effective_potential(S)
        ps, 𝜋 = S.p
        @unpack ∇𝒟𝜋, 𝜋s, λ = ps
        ∇V = x -> ∇𝒟𝜋(x) / (𝜋s(x) + λ)
        return function V(xs::AbstractRange)
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
end

begin # * Step-size distributions: α = 2 (Gaussian) vs α = 1.5 (heavy-tailed)
    α1 = 2.0
    α2 = 1.5

    # Stable(α, skewness=0, scale=1, location=0); fold onto positive axis: p(|x|) = 2*pdf(x)
    d1 = Stable(α1, 0, 1, 0)
    d2 = Stable(α2, 0, 1, 0)

    xs_pdf = range(0, 50, length = 500)
    p1 = 2 .* pdf.(d1, xs_pdf)
    p2 = 2 .* pdf.(d2, xs_pdf)

    Random.seed!(seed) # samples for the binned view; seeded so the data file is reproducible
    xs_a1 = abs.(rand(d1, 100000))
    xs_a2 = abs.(rand(d2, 100000))
end

begin # * Relaxation functions: β = 1 (exponential) vs β < 1 (Mittag-Leffler)
    ts = range(0, 50, length = 500)
    β1 = 1.0
    β2 = 0.5
    R1 = exp.(-ts)
    R2 = mittleff.(β2, .-ts)
end

begin # * Unconfined spectra: β = 1.0 vs β = 0.5
    β_hi = 1.0
    β_lo = 0.5
    η_psd = 0.1
    γ_psd = 0.1
    dt_psd = 0.25
    tspan_psd = 50000.0

    𝜋_psd = Normal(0, 10.0) |> Density
    function unconfined_spectrum(β)
        noise = gen_lfsm_fns(
            2.0, β; tspan = tspan_psd, dt = dt_psd, seed,
            nhist = round(Int, τ / dt_psd)
        )
        S = bFNS(;
            η = η_psd, β, γ = γ_psd, α = 2.0, u0 = [0.0, 0.0],
            tspan = tspan_psd, dt = dt_psd, seed, 𝜋 = 𝜋_psd, noise,
            domain = -50 .. 50,
            approx_n_modes = 1000, τ
        )
        sol = solve(S) |> Timeseries |> eachcol |> first
        sol = rectify(sol, dims = 𝑡; tol = 1)
        sol = sol[(length(sol) ÷ 2 + 1):end] # discard transient half
        s = spectrum(sol .- mean(sol), 0.05)[𝑓 = 1.0e-1 .. 1.0e0]
        return s ./ maximum(s)
    end
    s_hi = unconfined_spectrum(β_hi)
    s_lo = unconfined_spectrum(β_lo)
end

begin # * Sample time series: effect of α, β, γ
    ts_tspan = 5.0
    ts_dt = 0.01
    ts_seed = 99 # chosen from plots/Fig2_bFNS/seed_examples.png: no global drift over the shown window
    ts_common = (;
        domain = -5 .. 5,
        boundaries = PeriodicBox(-3 .. 3),
        u0 = [0.0, 0.0],
        dt = ts_dt,
        tspan = ts_tspan,
        approx_n_modes = 1000,
        τ = 1000.0,
        seed = ts_seed,
        λ = 1.0e-4,
        η = 0.01,
        𝜋 = test_density(:unimodal),
    )

    α_mod = 1.6
    β_mod = 0.5
    γ_mod = 4

    # Four configurations, each progressively changing one parameter
    configs = [
        (
            label = "Standard diffusion\n(α=2, β=1, γ=0)",
            α = 2.0, β = 1.0, γ = 0.0,
        ),
        (
            label = "+ Lévy superdiffusion\n(α=$(α_mod), β=1, γ=0)",
            α = α_mod, β = 1.0, γ = 0.0,
        ),
        (
            label = "+ LRTC subdiffusion\n(α=$(α_mod), β=$(β_mod), γ=0)",
            α = α_mod, β = β_mod, γ = 0.0,
        ),
        (
            label = "+ Oscillations\n(α=$(α_mod), β=$(β_mod), γ=$(γ_mod))",
            α = α_mod, β = β_mod, γ = γ_mod,
        ),
    ]

    ts_sols = map(configs) do c
        noise = gen_lfsm_fns(
            c.α, c.β;
            tspan = ts_tspan, dt = ts_dt, seed = ts_seed,
            nhist = round(Int, ts_common.τ / ts_dt)
        )
        S = bFNS(; ts_common..., α = c.α, β = c.β, γ = c.γ, noise)
        sol = solve(S) |> Timeseries |> eachcol |> first
        rectify(sol, dims = 𝑡; tol = 1)
    end
    ts_windows = [sol[𝑡 = 2 .. ts_tspan] for sol in ts_sols] # representative window
end

begin # * Long simulations for scaling estimates (flat, unimodal and bimodal samplers)
    sum_tmax = 25000.0
    sum_tmin = 5000.0 # ms transient
    sum_params = (; shared_params..., tspan = sum_tmax)
    γ_bimodal = 0.02 # the momentum coupling the bimodal sweep was run at
    sum_noise = gen_lfsm_fns(
        sum_params.α, sum_params.β;
        tspan = sum_tmax, dt = sum_params.dt, seed = sum_params.seed,
        nhist = round(Int, sum_params.τ / sum_params.dt)
    ) # same noise for all three samplers

    summary_sampler(𝜋; kwargs...) = bFNS(;
        (; sum_params..., 𝜋, kwargs...)..., noise = sum_noise
    )
    function summary_sol(S)
        sol = solve(S) |> Timeseries |> eachcol |> first
        sol = rectify(sol, dims = 𝑡; tol = 1)[𝑡 = sum_tmin .. sum_tmax]
        return set(sol, 𝑡 => times(sol) ./ 1000) # to s
    end

    sS = summary_sampler(test_density(:flat)) # unconfined
    gS = summary_sampler(test_density(:unimodal))
    bS = summary_sampler(test_density(:bimodal); γ = γ_bimodal)
    ssol, gsol, bsol = summary_sol.((sS, gS, bS))
end

begin # * MAD scaling + fitted diffusion exponent
    τs = logrange(step(ssol), 1.0, length = 100) # s
    mads = madev(ssol, τs) * 1.5
    gmads = madev(gsol, τs)

    frange = 1.0e-4 .. 1.0e-2
    m_mad = fit(MAPPLE, mads[𝑡 = frange]; peaks = 0, components = 1)
    fit!(m_mad, mads[𝑡 = frange])
    a_exponent = m_mad.params.components[1].β
    τfit = τs[τs .∈ [frange]]
    mad_fit = predict(m_mad, τfit)
end

begin # * Power spectra + fitted spectral exponent
    psd = spectrum(ssol .- mean(ssol), 1.0; padding = 1000)[𝑓 = eps() .. 1000]
    gpsd = spectrum(gsol .- mean(gsol), 1.0; padding = 1000)[𝑓 = eps() .. 1000]

    logs = logsample(ustripall(psd)[𝑓 = 10 .. 1000])
    m_psd = fit(MAPPLE, logs; peaks = 0, components = 1)
    fit!(m_psd, logs)
    b_exponent = m_psd.params.components[end].β
    psd_fit_x = collect(lookup(logs, 1))
    psd_fit_y = predict(m_psd, psd_fit_x)
end

# ──────────────────────────────────────────────────────────────────────────────
# Supplementary summaries of the two confined samplers (FigS2, FigS3)
# ──────────────────────────────────────────────────────────────────────────────

begin # * Bimodal potential, on the grid the unimodal one already uses
    bi_V = potential(Density(bS)).(xs)
    bi_V = bi_V .- minimum(bi_V)
    bi_Ṽ = effective_potential(bS)(_xs)[idxs]
    bi_Ṽ .-= minimum(bi_Ṽ)
end

begin # * Sampled distributions against their targets
    box = only(FractionalNeuralSampling.domain(shared_params.boundaries))
    uni_accuracy = samplingaccuracy(gsol, Density(gS); domain = box)
    bi_accuracy = samplingaccuracy(bsol, Density(bS); domain = box)
    uni_target = Density(gS).(xs)
    bi_target = Density(bS).(xs)
    bins = range(prange..., length = 25) # the binning the density panels draw
end

begin # * Bimodal MAD scaling + fitted diffusion exponent
    bi_mads = madev(bsol, τs)
    m_bi_mad = fit(MAPPLE, bi_mads[𝑡 = frange]; peaks = 0, components = 1)
    fit!(m_bi_mad, bi_mads[𝑡 = frange])
    bi_a_exponent = m_bi_mad.params.components[1].β
    bi_mad_fit = predict(m_bi_mad, τfit)
end

begin # * Bimodal power spectrum + fitted spectral exponent
    # f_min = 3 Hz rather than the 1 Hz used above: the shorter windows average down the
    # mode-switching variance, which otherwise swamps the scaling band.
    bi_psd = spectrum(bsol .- mean(bsol), 3.0; padding = 1000)[𝑓 = eps() .. 1000]
    bi_logs = logsample(ustripall(bi_psd)[𝑓 = 10 .. 1000])
    m_bi_psd = fit(MAPPLE, bi_logs; peaks = 0, components = 1)
    fit!(m_bi_psd, bi_logs)
    bi_b_exponent = m_bi_psd.params.components[end].β
    bi_psd_fit_x = collect(lookup(bi_logs, 1))
    bi_psd_fit_y = predict(m_bi_psd, bi_psd_fit_x)
end

begin # * Exponent and accuracy maps over (α, β), from the bFNS sweep
    function sweep_maps(density, γ, η)
        sweep = wload(rootdatadir("bFNS_sweep", "$(density)_γ=$(γ)_η=$(η).jld2"))
        slice(k) = Dropdims(mean)(sweep[k][η = At(η), γ = At(γ)], dims = Obs)
        return slice("diffusion_exponent"), slice("spectral_exponent"), slice("accuracy")
    end
    ma, ms, _ = sweep_maps("flat", shared_params.γ, shared_params.η)
    uni_ma, uni_ms, uni_macc = sweep_maps("unimodal", shared_params.γ, shared_params.η)
    bi_ma, bi_ms, bi_macc = sweep_maps("bimodal", γ_bimodal, shared_params.η)
end

begin # * Save
    tagsave(
        rootdatadir("bFNS_data.jld2"),
        Dict(
            # a: potential + effective potential on the plotted grid
            "xs" => collect(xs), "Vs" => Vs, "Ṽs" => Ṽs,
            # b: analytic folded stable densities + samples for the binned view
            "xs_pdf" => collect(xs_pdf), "p1" => p1, "p2" => p2,
            "xs_a1" => xs_a1, "xs_a2" => xs_a2,
            # c: relaxation functions
            "ts" => collect(ts), "R1" => R1, "R2" => R2,
            # d: peak-normalised unconfined spectra
            "s_hi" => s_hi, "s_lo" => s_lo,
            # right column: windowed sample traces + their configurations
            "ts_windows" => ts_windows, "configs" => configs,
            # e: MAD curves + fit
            "τs" => collect(τs), "mads" => collect(mads), "gmads" => collect(gmads),
            "τfit" => collect(τfit), "mad_fit" => mad_fit, "a_exponent" => a_exponent,
            # f: power spectra + fit
            "psd" => psd, "gpsd" => gpsd,
            "psd_fit_x" => psd_fit_x, "psd_fit_y" => psd_fit_y,
            "b_exponent" => b_exponent,
            # g, h: Obs-mean exponent maps
            "ma" => ma, "ms" => ms,
            # FigS2 --- unimodal sampler (its potential is Fig2's "Vs"/"Ṽs")
            "uni_samples" => collect(gsol), "uni_target" => uni_target,
            "uni_accuracy" => uni_accuracy,
            "uni_ma" => uni_ma, "uni_ms" => uni_ms, "uni_macc" => uni_macc,
            # FigS3 --- bimodal sampler
            "bi_V" => bi_V, "bi_Ṽ" => bi_Ṽ,
            "bi_samples" => collect(bsol), "bi_target" => bi_target,
            "bi_accuracy" => bi_accuracy,
            "bi_mads" => collect(bi_mads), "bi_mad_fit" => bi_mad_fit,
            "bi_a_exponent" => bi_a_exponent,
            "bi_psd" => bi_psd,
            "bi_psd_fit_x" => bi_psd_fit_x, "bi_psd_fit_y" => bi_psd_fit_y,
            "bi_b_exponent" => bi_b_exponent,
            "bi_ma" => bi_ma, "bi_ms" => bi_ms, "bi_macc" => bi_macc,
            # binning shared by the two density panels
            "bins" => collect(bins),
            # scalars for labels and the working-regime marker
            "params" => (;
                α = shared_params.α, β = shared_params.β,
                γ = shared_params.γ, γ_bimodal, η = shared_params.η,
                tspan, dt, seed, τ, α1, α2, β1, β2, β_hi, β_lo,
            ),
        )
    )
    @info "wrote" rootdatadir("bFNS_data.jld2")
end
