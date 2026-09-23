using Distributions
using StableDistributions
using TimeseriesTools

export test_density, simulate_bFNS_sweep

function test_density(::Val{:flat})
    Normal(0.0, 1000.0) |> Density
end
function test_density(::Val{:unimodal})
    Normal(0, 0.25) |> Density
    # 𝜋 = Stable(1.5, -0.03, 0.14, 0.20) |> Density
end

function test_density(::Val{:bimodal})
    [Normal(-0.5, 0.15), Normal(0.5, 0.15)] |> MixtureModel |> Density
end

test_density(s::Symbol) = test_density(Val(s))

function simulate_bFNS_sweep(params, α, β, γ, η, seed)
    H = (1 - β) / 2 + 1 / α
    if !(0 < H < 1)
        return (; accuracy = NaN,
                diffusion_exponent = NaN,
                spectral_exponent = NaN,
                seed = NaN)
    end

    S = bFNS(; γ, η, α, β, seed, params...)
    _sol = solve(S)

    sol = _sol |> Timeseries |> eachcol |> first
    ts = range(0, params[:tspan]; step = params[:dt])
    # sol = rectify(sol, dims = 𝑡, tol = 1)
    sol = set(sol, 𝑡 => ts ./ 1000)
    sol = sol[𝑡 = 5 .. Inf] # Remove 5s transient
    # * First, spectrum
    s = spectrum(sol, 3.0; padding = 500)
    s = logsample(s[𝑓 = 100 .. 1000]) # Remove edge effects
    m = fit(MAPPLE, s; peaks = 0, components = 1)
    fit!(m, s)
    spectral_exponent = m.params.components.β |> first

    # * Then MAD
    mad = madev(sol, logrange(step(sol), 1e-2, length = 100)) # ms
    m = fit(MAPPLE, mad; peaks = 0, components = 1)
    fit!(m, mad)
    diffusion_exponent = m.params.components.β |> first

    # * Sampling accuracy
    accuracy = samplingaccuracy(sol, Density(S);
                                domain = only(FractionalNeuralSampling.domain(params.boundaries)))

    (; seed, spectral_exponent, diffusion_exponent, accuracy)
end

function simulate_bFNS_sweep(params)
    function _simulate(α, β, γ, η, obs)
        seed = UInt32(obs + 100 * (round(Int, 1000α) + 10_000 * round(Int, 1000β))) # distinct per (α, β, obs); unlike `hash`, stable across Julia versions
        simulate_bFNS_sweep(params, α, β, γ, η, seed)
    end
end
# simulate_bFNS_sweep(α, β, γ, η, args...) = simulate_bFNS_sweep(; α, β, γ, η)
