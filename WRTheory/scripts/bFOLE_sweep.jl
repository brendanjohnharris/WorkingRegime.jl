#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
using StochasticDiffEq
WRTheory.@preamble()
set_theme!(foresight(:physics))

import FractionalNeuralSampling: Density

function test_density()
    # Build a heterogeneous mixture of distribution types

    components = [
        (0.5, Normal(-5.0, 1)),
        (0.5, Normal(5.0, 1)),
        # (0.5, Laplace(-0.4, 0.05)),
        # (0.5, Laplace(0.4, 0.05))
    ]

    dists = [d for (w, d) in components]
    weights = [w for (w, d) in components]

    # Normalize weights
    weights = weights ./ sum(weights)

    D = MixtureModel(dists, weights)
    return Density(D)
end

# * The aim is to sweep over alpha, beta, and eta
# * We extract the sampling accuracy, sampling efficiency,
# * spectral exponent, and diffusion exponent (and seed)
begin # * Create parameter grid
    αs = range(1.2, 2.0; length = 16) |> Dim{:α}
    βs = range(0.2, 1.0; length = 16) |> Dim{:β}
    ηs = [0.01, 0.1, 1.0] |> Dim{:η}
    obs = Obs(1:10)

    params = (;
        tspan = 10000.0, # ms
        dt = 0.1,
        u0 = [-5.0],
        domain = -40.0 .. 40.0,
        boundaries = PeriodicBox((-20.0,), (20.0,)),
        𝜋 = test_density(),
    )
end

begin # * Heatmap of H values
    _αs = range(1.0, 2.0; length = 100) |> Dim{:α}
    _βs = range(0.0, 1.0; length = 100) |> Dim{:β}
    h = map((α, β) -> (1 - β) / 2 + 1 / α, Chart(Iterators.product), _αs, _βs)
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "α", ylabel = "β", title = "H = (1 - β)/2 + 1/α")
    p = heatmap!(ax, h; colormap = darksunset)
    contour!(ax, h; levels = [0, 1], color = :white, linestyle = :dash)
    Colorbar(f[1, 2], p)
    display(f)
end

begin # * Define the simulation function
    function simulate(; α, β, η, seed)
        H = (1 - β) / 2 + 1 / α
        if !(0 < H < 1)
            return (;
                #accuracy = NaN,
                diffusion_exponent = NaN,
                spectral_exponent = NaN,
                seed = NaN,
            )
        end

        S = bFOLE(; η, α, β, seed, params...)
        _sol = solve(S)

        sol = _sol |> Timeseries |> eachcol |> first
        sol = rectify(sol, dims = 𝑡, tol = 1)

        # * First, spectrum
        s = spectrum(sol, 1 / 200)
        s = logsample(s[10:end])[1:(end - 10)] # Remove edge effects
        m = fit(MAPPLE, s; peaks = 0, components = 1)
        fit!(m, s)
        spectral_exponent = m.params.components.β |> first

        # * Then MAD
        mad = madev(sol, logrange(0.1, 200.0, length = 50))
        m = fit(MAPPLE, mad; peaks = 0, components = 1)
        fit!(m, mad)
        diffusion_exponent = m.params.components.β |> first

        # # * Sampling accuracy
        # subsol = sol[1:10:end]
        # accuracy = samplingaccuracy(subsol, Density(S),
        #                             round.(Int,
        #                                    logrange(10, length(subsol) ÷ 10, length = 50)))

        return (; seed, spectral_exponent, diffusion_exponent) #, accuracy)
    end
end

begin # * Map over parameters
    C = Chart(
        Iterators.product,
        Threaded(),
        ProgressLogger()
    )
    res = map(C, αs, βs, ηs, obs) do α, β, η, _
        seed = rand(UInt32)
        simulate(; α, β, η, seed)
    end
    out = map(keys(first(res))) do k
        string(k) => map(Base.Fix2(getindex, k), res)
    end |> Dict{String, Any}
    out["params"] = params
    tagsave(datadir("bFOLE_sweep.jld2"), out)
end
