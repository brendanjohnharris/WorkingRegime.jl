using FractionalNeuralSampling
using CairoMakie
using Distributions
using Foresight
using TimeseriesTools
using TimeseriesMakie
using StableDistributions
using Unitful
using Random
using Optim
Random.seed!(0)

set_theme!(foresight(:physics))

begin # * Set up a potential, say quadratic shifted by some mean
    𝜋 = FractionalNeuralSampling.Density(Normal(-55, 3.0))

    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")
    lines!(ax, -70:0.01:-50, potential(𝜋))
    axx = Axis(f[1, 1], xlabel = "x", ylabel = "𝜋(x)", yaxisposition = :right)
    lines!(axx, -70:0.01:-50, x -> 𝜋(x), color = crimson)
    hidedecorations!(axx)
    display(f)
end

begin # * Create sampler
    bc = ReentrantBox(-50.0 => -70.0)
    bcf = FractionalNeuralSampling.Boundaries.getcondition(bc)
    S = FHMC(;
             tspan = 10000.0,
             α = 1.6,
             β = 0.1,
             γ = 0.1,
             u0 = [-55.0, 0.0],
             boundaries = bc(),
             𝜋,
             alg = EM())
end

begin # * Run a simulation
    sol = solve(S; dt = 0.1) |> Timeseries
end

begin
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")
    lines!(ax, -70:0.01:-50, potential(𝜋))
    axx = Axis(f[1, 1], xlabel = "t", ylabel = "x", title = "Trajectory",
               yaxisposition = :right)
    linkxaxes!(ax, axx)
    hidedecorations!(axx)
    hill!(axx, first(eachcol(sol))[1:10:10000], color = crimson)
    lines!(axx, -70:0.01:-50, x -> 𝜋(x), color = crimson, linestyle = :dash)
    axx.limits = ((-70, -50), nothing)
    display(f)
end
begin # Plot
    x = first(eachcol(sol))[1:2000]
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "V(x)")

    lines!(ax, x, linewidth = 1)

    # * Add red spikes whenever we go above -50
    sloc = findall(x .> -50)
    scatter!(ax, times(x)[sloc], fill(-50, length(sloc)), color = :red, markersize = 30,
             marker = :vline, marker_offset = (0, 10))
    display(f)
end
begin # * Extract spikes
    x = first(eachcol(sol))
    sloc = findall(x .> -50) # Assume walker only ever stays above threshold for 1 step
    spikes = times(x)[sloc]
    isis = diff(spikes)
    lisis = log10.(isis)
    if !isempty(spikes)
        f = Figure()
        ax = Axis(f[1, 1], xlabel = "Inter-spike interval (ms)", ylabel = "Density",
                  yscale = log10)
        hist!(ax, lisis, normalization = :pdf, bins = 100)
        vlines!(ax, log10(median(isis)), color = :red, linestyle = :dash,
                label = "Median = $(round(median(isis), digits=2)) ms")
        axislegend(ax, position = :rt)
        display(f)
    end
end

begin # * try spectrum
    y = first(eachcol(sol))
    y = y .- mean(y)
    y = set(y, 𝑡 => u"s" * times(y) ./ 1000) # To s
    s = spectrum(rectify(y, dims = 𝑡, tol = 1))
    plotspectrum(s)

    ls = logsample(ustripall(s)[10:end])
    params = fit_oneoneff(ls; n_peaks = 0)
    params = fit_oneoneff(ls, params)

    f = Figure()
    ax = Axis(f[1, 1], xlabel = "Log frequency", ylabel = "Log spectral density")
    lines!(ax, ls)
    lines!(ax, lookup(ls, 1), oneoneff(lookup(ls, 1), params), color = :red)
    ax.title = "Fitted 1/f^$(round(params.β, digits=2))"
    display(f)
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
