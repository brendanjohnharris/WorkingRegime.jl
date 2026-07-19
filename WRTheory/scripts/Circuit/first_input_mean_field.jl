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

begin # * Set up a potential. Should eventually fit from data
    𝜋 = FractionalNeuralSampling.Density(Normal(0.0, 1.0))
    xs = -4:0.01:4
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")
    lines!(ax, xs, potential(𝜋))
    axx = Axis(f[1, 1], xlabel = "x", ylabel = "𝜋(x)", yaxisposition = :right)
    lines!(axx, xs, x -> 𝜋(x), color = crimson)
    hidedecorations!(axx)
    display(f)
end

begin # * Create sampler
    S = FNS(;
            tspan = 10000.0,
            α = 2.0,
            β = 0.0,
            γ = 0.1,
            u0 = [0.0, 0.0],
            𝜋,
            alg = EM())

    sol = solve(S; dt = 0.1) |> Timeseries
end

begin
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")
    lines!(ax, xs, potential(𝜋))
    axx = Axis(f[1, 1], xlabel = "t", ylabel = "x", title = "Trajectory",
               yaxisposition = :right)
    linkxaxes!(ax, axx)
    hidedecorations!(axx)
    hill!(axx, first(eachcol(sol))[1:10:10000], color = crimson)
    lines!(axx, xs, x -> 𝜋(x), color = crimson, linestyle = :dash)
    axx.limits = (extrema(xs), nothing)
    display(f)
end
begin # Plot
    x = first(eachcol(sol))[1:2000]
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")

    lines!(ax, x, linewidth = 1)

    # * Add red spikes whenever we go above -50
    # sloc = findall(x .> -50)
    # scatter!(ax, times(x)[sloc], fill(-50, length(sloc)), color = :red, markersize = 30,
    #          marker = :vline, marker_offset = (0, 10))
    display(f)
end

begin # * Filter the inputs through a model neuron
    dt = 0.1 # * Needs to match Dewdrop dt unless we correct for it
    tspan = 500000.0
    u0 = [-60.0, 0.0]
    𝜋 = FractionalNeuralSampling.Density(Normal(0.2, 2.0))
    S = FNS(;
            tspan = Inf, #tspan,
            α = 1.5,
            β = 0.0,
            γ = 10.0,
            u0 = [0.0, 0.0],
            𝜋,
            alg = EM(),
            dt)
    prob = NeuronSampler(S; tspan, u0)
    neuron_sol = solve(prob)# |> Timeseries
    lines(neuron_sol[1, 1:10000]) |> display
    lines(neuron_sol[2, 1:10000])
end
begin # Plot
    x = neuron_sol[1, 1:2000]
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")

    lines!(ax, x, linewidth = 2)

    # * Add red spikes whenever we go above -50
    # sloc = findall(x .> -50)
    # scatter!(ax, times(x)[sloc], fill(-50, length(sloc)), color = :red, markersize = 30,
    #          marker = :vline, marker_offset = (0, 10))
    display(f)
end

begin # * Extract spikes
    sol = neuron_sol |> Timeseries
    x = first(eachcol(sol))
    sloc = findall(x .> -50) # Assume walker only ever stays above threshold for 1 step
    spikes = times(x)[sloc]
    isis = diff(spikes)
    lisis = log10.(isis)
    if !isempty(spikes)
        f = Figure()
        ax = Axis(f[1, 1], xlabel = "Log10 inter-spike interval (ms)", ylabel = "Density",
                  yscale = log10)
        hist!(ax, lisis, normalization = :pdf, bins = 100)
        vlines!(ax, log10(median(isis)), color = :red, linestyle = :dash,
                label = "Median = $(round(median(isis), digits=2)) ms")
        axislegend(ax, position = :rt)
        display(f)
    end
end

begin # * Fano factor
    fanos = fano_factor(spikes)

    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Time window (ms)", ylabel = "Fano factor",
              xscale = log10, yscale = log10, title = "Fano factor")
    lines!(ax, fanos)
    f |> display
end

# begin # * try spectrum
#     y = first(eachcol(sol))
#     y = y .- mean(y)
#     # y = set(y, 𝑡 => u"s" * times(y) ./ 1000) # To s
#     s = spectrum(rectify(y, dims = 𝑡, tol = 1))
#     plotspectrum(s)

#     ls = logsample(ustripall(s)[5:end])
#     # params = fit_oneoneff(ls; n_peaks = 0)
#     # params = fit_oneoneff(ls, params)

#     f = Figure()
#     ax = Axis(f[1, 1], xlabel = "Frequency", ylabel = "Spectral density", yscale = log10,
#               xscale = log10)
#     lines!(ax, ls)
#     # lines!(ax, lookup(ls, 1), oneoneff(lookup(ls, 1), params), color = :red)
#     # ax.title = "Fitted 1/f^$(round(params.β, digits=2))"
#     display(f)
# end

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
