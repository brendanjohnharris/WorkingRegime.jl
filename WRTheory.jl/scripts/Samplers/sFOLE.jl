#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
WRTheory.@preamble()
set_theme!(foresight(:physics))

import FractionalNeuralSampling: Density

begin # * Generate a sample path and test the distribution is ok
    α = 2.0
    tspan = 5000.00
    dt = 0.01
    η = 0.5
    u0 = [0.0]
    domain = -20.0 .. 20.0 # Should well cover the pdf and then some to avoid edge effects

    𝜋 = MixtureModel([Normal(-0, 0.5)]) |> Density #
    boundary = PeriodicBox((-10,), (10,))
    S = sFOLE(; η, α, u0, 𝜋, tspan, dt, domain, boundaries = boundary())
    # S = FNS(; γ = η, α, β = 0, u0 = [0.0, 0.0], 𝜋, tspan, dt,
    #         boundaries = boundary())

    _sol = solve(S)
    sol = _sol |> Timeseries |> eachcol |> first
    sol = rectify(sol, dims = 𝑡, tol = 1)

    f = Figure()
    ax = Axis(f[1, 1], limits = ((-4, 4), nothing))
    xs = -3:0.01:3
    ys = 𝜋.(xs)
    hist!(ax, sol; bins = -4:0.1:4, normalization = :pdf, color = (:crimson, 0.5))
    lines!(ax, xs, ys)

    display(f)
end

begin
    s = spectrum(sol, 1 / 10)
    s = logsample(s[10:end])[1:(end - 10)] # Remove edge effects
    m = fit(MAPPLE, s; peaks = 0, components = 1)
    fit!(m, s)
    spectral_exponent = m.params.components.β |> first

    f = Figure()
    ax = Axis(f[1, 1]; xscale = log10, yscale = log10,
              xlabel = "Frequency (Hz)", ylabel = "Power",
              title = "Spectral Exponent = $(round(spectral_exponent, digits=2))")
    lines!(s)
    lines!(lookup(s, 1), predict(m, lookup(s, 1)))
    current_figure()
end
