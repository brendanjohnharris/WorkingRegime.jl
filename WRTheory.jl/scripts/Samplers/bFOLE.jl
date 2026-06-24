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
    α = 1.7
    β = 1.0
    tspan = 2000.00
    dt = 0.01
    η = 0.1
    N = Int(tspan / dt) + 1
    u0 = [0.0]
    domain = -20.0 .. 20.0 # Should well cover the pdf and then some to avoid edge effects

    𝜋 = MixtureModel([Normal(-1, 0.2), Normal(1, 0.2)]) |> Density
    boundary = PeriodicBox((-10,), (10,))
    S = bFOLE(; η, α, β, u0, 𝜋, tspan, dt, domain, boundaries = boundary)

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
begin # * Plot trajectory
    lines(sol[1:1000:end])
end
begin # * plot spectral exponent
    s = spectrum(sol, 0.1)
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Frequency", ylabel = "Power", xscale = log10,
              yscale = log10)
    s = logsample(s[10:end])[1:(end - 10)] # Remove edge effects
    m = fit(MAPPLE, s; peaks = 0, components = 1)
    fit!(m, s)
    beta = m.params.components.β |> first

    lines!(ax, s; label = "β = $(round(beta, digits = 2))")
    lines!(ax, freqs(s), predict(m, freqs(s)); linestyle = :dash, color = :gray)
    axislegend(ax)
    display(f)
end

begin # * Accuracy
    τs = logrange(1000, 200000, length = 25)
    τs = round.(Int, τs)
    y = samplingaccuracy(sol, 𝜋, τs; p = 1000)  #/ sqrt(samplingpower(x, dt))
    acc = ToolsArray(y, 𝑡(τs .* samplingperiod(sol)))
    lines(acc .|> mean; axis = (; xscale = log10, limits = (nothing, (0, 1))))
end

# begin
#     etas = 0.2:0.4:2.0
#     dt = 0.01
#     β = 0.6
#     τs = round.(Int, logrange(1, 1000, 10)) .÷ dt .|> Int
#     𝜋 = MixtureModel([Normal(-2, 0.5), Normal(2, 0.5)]) |> Density
#     u0 = [0.0]
#     tspan = 5000.00

#     xs = map(Chart(ProgressLogger(), Threaded()), Dim{:η}(etas)) do η
#         S = OLE(; η, u0, 𝜋, tspan)
#         sol = solve(S, CaputoEM(β, 1000); dt) |> Timeseries |> eachcol |> first
#         return rectify(sol, dims = 𝑡; tol = 1)
#     end

#     accuracy = map(Chart(Threaded(), ProgressLogger()), xs) do x
#         y = samplingaccuracy(x, 𝜋, τs; p = 1000)  #/ sqrt(samplingpower(x, dt))
#         ToolsArray(y, 𝑡(τs))
#     end |> stack
#     accuracy = map(mean, accuracy)

#     # _τs = τs * dt # For efficiency
#     # efficiency = map(Chart(Threaded(), ProgressLogger()), xs) do x
#     #     y = samplingefficiency(x, 𝜋, _τs; downsample = 5, p = 1000)
#     #     ToolsArray(y, 𝑡(_τs))
#     # end |> stack
#     # efficiency = map(mean, efficiency)
# end

# begin
#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Time lag", ylabel = "Accuracy", xscale = log10)
#     p = traces!(ax, accuracy, linewidth = 2)
#     hlines!(ax, [1.0]; color = :gray, linestyle = :dash)
#     Colorbar(f[1, 2], p; label = "η")
#     display(f)
# end
# begin
#     ts = 1:100
#     vd = map(Chart(Threaded(), ProgressLogger()), xs) do x
#         y = map(ts) do t
#             samplingpower(x[1:t:end])
#         end
#         ToolsArray(y, 𝑡(ts))
#     end |> stack

#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Time step", ylabel = "Sampling power", xscale = log10)
#     p = traces!(ax, vd, linewidth = 2)
#     Colorbar(f[1, 2], p; label = "η")
#     display(f)
# end

# begin
#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Time lag", ylabel = "Efficiency", xscale = log10)
#     p = traces!(ax, efficiency, linewidth = 2)
#     hlines!(ax, [1.0]; color = :gray, linestyle = :dash)
#     Colorbar(f[1, 2], p; label = "η")
#     display(f)
# end
