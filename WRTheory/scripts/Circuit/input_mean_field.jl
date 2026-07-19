#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
using Optim
WRTheory.@preamble()
import FractionalNeuralSampling.Samplers: gen_lfsm_fns
set_theme!(foresight(:physics))
update_theme!(; labelsize = 12)

begin # * Create sampler
    tspan = 55000.0
    tmin = 5000.0 # ms transient
    dt = 0.1
    _dt = 0.05
    u0 = [0.0, 0.0]
    τ = 1000.0
    seed = 42

    α = 1.5
    β = 0.8
    η = 0.03
    γ = 0.01

    # 𝜋 = Stable(1.5, -0.1, 0.13, 0.22) |> Density # (α, β, σ, μ)
    𝜋 = Stable(1.5, 0, 0.14, 0.20) |> Density # (α, β, σ, μ)

    # * Precompute noise
    noise = gen_lfsm_fns(α, β; tspan, dt, seed, nhist = round(Int, τ / dt))

    params = (;
              α,
              β,
              γ,
              η,
              𝜋,
              domain = -10 .. 10,
              boundaries = PeriodicBox(-5 .. 5),
              u0,
              τ,
              approx_n_modes = 1000,
              λ = 1e-4,
              dt = _dt,
              saveat = dt,
              tspan,
              noise,
              seed = 42)

    S = bFNS(; params...) |> solve

    input_sol = S |> Timeseries
    input_sol = input_sol[𝑡 = tmin .. tspan]

    # begin # * Filter the inputs through a model neuron
    u0 = [-60.0, 0.0]
    prob = NeuronSampler(S; tspan, u0)
    neuron_sol = solve(prob)
    # end
    # begin # Plot
    sol = neuron_sol |> Timeseries
    sol = sol[𝑡 = tmin .. tspan]
    spikes = times(DimensionalData.metadata(sol)[:callback_values])
    spikes = spikes[spikes .> tmin]

    times(sol) ./= 1000 # To s
    times(input_sol) ./= 1000
    spikes ./= 1000

    begin
        isis = diff(spikes)
        fr = length(spikes) / (duration(sol))

        # # lisis = log10.(isis)
        # if !isempty(spikes)
        #     bins = 1:1:1000.0
        #     ax = Axis(f[2, 2], xlabel = "Inter-spike interval (ms)", ylabel = "Density",
        #               yscale = log10, xscale = log10,
        #               title = "Firing rate = $(round(fr, digits=2)) Hz")
        #     hist!(ax, isis; normalization = :pdf, bins)
        #     vlines!(ax, median(isis), color = :red, linestyle = :dash,
        #             label = "Mean = $(round(mean(isis), digits=2)) ms")
        #     # axislegend(ax, position = :rt)
        # end
    end

    f = OnePanel(; size = (468, 324))

    begin # * Membrane potential
        ax = Axis(f[1, 1], title = "Membrane potential", xlabel = "Time (s)",
                  ylabel = "𝑉 (mV)",
                  limits = (nothing, (-71, -49)),
                  yticks = WilkinsonTicks(4; k_max = 5))

        hlines!(ax, [-50], color = :red)
        hlines!(ax, [-70], color = :red, linestyle = :dash)
        hlines!(ax, [mean(sol[:, 1])], color = :gray, linestyle = :dash)

        x = sol[1:10000, 1]
        times(x) .-= first(times(x))
        lines!(ax, x, linewidth = 2)

        # # * Add red spikes whenever we go above -50
        # # sloc = findall(x .> -50)
        # s = spikes[minimum(times(x)) .< spikes .< maximum(times(x))]
        # scatter!(ax, s, fill(maximum(x), length(s)),
        #          color = :red,
        #          markersize = 30,
        #          marker = :vline, marker_offset = (0, 10))
        axislegend(ax, [LineElement(color = :transparent, linestyle = nothing)],
                   [L"\nu \approx %$(round(fr, digits=1)) \textrm{ Hz }"];
                   position = :rb, framevisible = true, patchsize = (0.1, 0.1))
        # text!(ax, [0.0], [-55]; text = "(ν=$(round(fr, digits=1)) Hz)")
    end

    begin # * Distribution
        bins = -70:1:-50
        ax = Axis(f[1, 2], title = "Density", xlabel = "𝑉 (mV)", ylabel = "PDF",
                  yticks = WilkinsonTicks(2),
                  xticks = WilkinsonTicks(3), limits = ((-70, -50), nothing))
        density!(sol[:, 1])
    end

    begin # Inputs
        ax = Axis(f[2, 1], xlabel = "Time (s)", ylabel = "𝐼 (nA)",
                  title = "Input current",
                  limits = (nothing, (-2, 5)), yticks = WilkinsonTicks(3; k_max = 4))
        x = input_sol[1:5000, 1]
        times(x) .-= first(times(x))
        lines!(ax, x, linewidth = 2)
    end

    begin # * Input diff distirbution
        y = diff(input_sol[1:10:end, 1]) .|> abs
        bins = 0:0.1:3.1
        bins = bins[2:end]
        ax = Axis(f[2, 2], xlabel = "|Δ𝐼| (nA)", ylabel = "Frequency",
                  yscale = log10, xscale = log10,
                  title = "Step sizes",
                  xticks = WilkinsonTicks(4; k_max = 4) |> LogTicks,
                  yticks = WilkinsonTicks(4; k_max = 4) |> LogTicks)
        ziggurat!(ax, y[:]; bins, normalization = :pdf)
    end

    colsize!(f.layout, 1, Relative(0.75))
    wsave(plotdir("input_mean_field", "input_mean_field.pdf"), f)
    display(f)
end

begin # * Fano factor
    # sspikes = spikes ./ 1000 # To s
    dt = 0.1
    rate = WRTheory.rates(spikes, dt)
    f = TwoPanel()
    ax = Axis(f[1, 1], xlabel = "Time window (s)", ylabel = "Spike counts")
    lines!(ax, range(0, step = dt, length = length(rate)), rate)

    τs = logrange(dt * 10 / 1000, 1, 100)
    fanos = fano_factor(spikes, τs)

    m = fit(MAPPLE, fanos; peaks = 0, components = 3)
    fit!(m, fanos)
    ff = predict(m, times(fanos))

    ax = Axis(f[1, 2]; xlabel = "Time window (s)", ylabel = "Fano factor",
              xscale = log10, yscale = log10, title = "Fano factor")
    lines!(ax, fanos)
    lines!(ax, times(fanos), ff; color = :red, linestyle = :dash)
    f |> display
end

if false # * try spectrum
    y = first(eachcol(input_sol))
    y = y .- mean(y)
    # y = set(y, 𝑡 => times(y) ./ 1000) # To s
    s = spectrum(rectify(y, dims = 𝑡, tol = 1), 10)
    fax = plotspectrum(s)
    ax = fax.axis

    ls = logsample(ustripall(s)[𝑓 = 100 .. 2000])
    m = fit(MAPPLE, ls; components = 1, peaks = 0)
    fit!(m, ls)

    # f = Figure()
    # ax = Axis(f[1, 1], xlabel = "Frequency", ylabel = "Spectral density", yscale = log10,
    #           xscale = log10)
    scatter!(ax, ls)

    lines!(ax, lookup(ls, 1), predict(m, lookup(ls, 1)), color = :red)
    # ax.title = "Fitted 1/f^$(round(params.β, digits=2))"
    ax.title = "f^$(round(m.params.components[end].β, digits=2))"
    display(fax)
    display(m)
end

# begin # * Plot
#     f = Figure()
#     ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")

#     lines!(ax, -70:0.01:-50, potential(𝜋))

#     traj = Observable(Point2f[[NaN, NaN]])
#     now = Observable(Point2f(NaN, NaN))
#     trail!(ax, traj, linecolor = :black, n_points = 100)
#     scatter!(ax, now, color = :red, markersize = 10)

#     record(f, "voltage_mean_field.mp4", 1:100:10000) do i
#         x = Point2f(sol[i, 1], potential(𝜋)(sol[i, 1]))
#         push!(traj[], x)
#         now[] = x
#     end
# end
