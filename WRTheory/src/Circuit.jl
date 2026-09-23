import FractionalNeuralSampling.Samplers: gen_lfsm_fns
using StochasticDiffEqRODE: RandomEM # moved out of StochasticDiffEq in v7, which no longer re-exports it

export NeuronSampler, fano_factor

function neuron_f!(du, u, p, t, W)
    @unpack C, V_L, g_L, V_K, τ_K = p
    V, g_K = u
    I = first(W)
    du[1] = (-g_L * (V - V_L) - g_K * (V - V_K) + I) / C
    du[2] = -g_K / τ_K # * Adaptation current
end
function adapt!(integrator)
    g_K = view(integrator.u, 2)
    Δg_K = integrator.p.Δg_K
    g_K .+= Δg_K
end

function _NeuronSampler(S::SciMLBase.AbstractSciMLProblem; dt)
    if haskey(S.kwargs, :dt)
        if isnothing(dt)
            dt = S.kwargs[:dt]
        end
        dt_s = S.kwargs[:dt]
        @assert dt==dt_s "dt provided to NeuronSampler must match dt used in sampler S"
    else
        error("dt must be provided to NeuronSampler if not already specified in sampler S")
    end
    noise = init(S; dt) |> NoiseApproximation
    return noise, dt
end

function _NeuronSampler(S::SciMLBase.AbstractSciMLSolution; dt)
    if haskey(S.prob.kwargs, :dt)
        if isnothing(dt)
            dt = S.prob.kwargs[:dt]
        end
        dt_s = S.prob.kwargs[:dt]
        @assert dt==dt_s "dt provided to NeuronSampler must match dt used in sampler S"
    else
        error("dt must be provided to NeuronSampler if not already specified in sampler S")
    end
    return NoiseGrid(S.t, S.u), dt
end

function _NeuronSampler(S::NoiseGrid; dt)
    if isnothing(dt)
        @error("dt must be provided to NeuronSampler when passing a NoiseGrid")
    end
    return S, dt
end

function NeuronSampler(S; u0, tspan, dt = nothing) # Important: here dt must equal the dt for the circuit model
    noise, dt = _NeuronSampler(S; dt)
    params = (C = 0.25,
              V_L = -70,
              g_L = 0.0167,
              V_K = -85.0,
              τ_K = 40.0,
              Δg_K = 0.002)
    boundary = ReentrantBox(-50.0 => -70.0; reset = false) # ? Threshold and reset potential. Don't reset adaptation variable when reentering
    adaptation = DiscreteCallback(FractionalNeuralSampling.Boundaries.getcondition(boundary),
                                  adapt!; save_positions = (false, false))
    prob = RODEProblem(neuron_f!, u0, tspan, params;
                       noise, dt, alg = RandomEM(),
                       callback = CallbackSet(adaptation, boundary()))
    # Then solve with solve(prob)
end

function _count(spike_times, τ; bins = minimum(spike_times):τ:maximum(spike_times))
    return fit(Histogram, spike_times, bins).weights
end

function rates(spike_times, τ)
    bins = minimum(spike_times):τ:maximum(spike_times)
    if isempty(bins)
        return bins .* NaN
    else
        counts = _count(spike_times, τ; bins)
        return counts ./ τ
    end
end

# function fano_factor(spike_times, τ)
#     counts = _count(spike_times, τ)
#     m = mean(counts)
#     return var(counts, mean = m) / m
# end

# function fano_factor(spike_times, τ_values::AbstractVector = defaultfanobins(spike_times))
#     f = [fano_factor(spike_times, τ) for τ in τ_values]
#     return Timeseries(f, τ_values)
# end

# function defaultfanobins(ts)
#     maxwidth = (first ∘ diff ∘ collect ∘ extrema)(ts) / 10
#     minwidth = max((mean ∘ diff)(ts), maxwidth / 10000)
#     # spacing = minwidth
#     return logrange(minwidth, maxwidth, length = 100)
# end

function simsave(α, β, seed; γs, ηs, named, unnamed)
    @unpack tspan = named
    @unpack u0_input, u0_neuron, 𝜋, boundaries, domain, approx_n_modes, dt, τ, λ = unnamed

    H = one(α) / 2 - β / 2 + 1 / α
    if 0 < H < 1
        noise = gen_lfsm_fns(α, β; tspan, dt, seed, nhist = round(Int, τ / dt)) # One noise per obs 'seed'
    else
        noise = NaN
    end

    # * Fano bins
    transient = 5000.0
    τs = logrange(dt * 10, (tspan .- transient) / 10, length = 200)

    # * Loop over γ and η
    map(Chart(Iterators.product), γs, ηs) do γ, η
        # hash = Base.hash(unnamed)
        filepath = savename((; α, β, γ, η, Obs = seed, named...), "tsv")
        fanofile = "fano_$filepath"

        if any(isnan, noise)
            fano = Timeseries(τs .* NaN, τs)
            spikes = Float32[]
            writedlm(datadir("mean_field_sweep", filepath), spikes)
            savetimeseries(datadir("mean_field_sweep", fanofile), fano)
        elseif !isfile(filepath)
            S = bFNS(; α, β, γ, η,
                     noise, tspan, dt,
                     𝜋, boundaries, domain,
                     approx_n_modes, τ, λ, u0 = u0_input) |> solve
            prob = NeuronSampler(S; tspan, u0 = u0_neuron)
            neuron_sol = solve(prob)
            sol = neuron_sol |> Timeseries
            spikes = getindex(metadata(sol), :callback_values) |> times
            spikes = convert(Vector{Float32}, spikes)

            # * Save spikes
            # wsave(filepath, Dict("spikes" => spikes))
            writedlm(datadir("mean_field_sweep", filepath), spikes)

            # * Save fano factor
            fano = Timeseries(τs .* NaN, τs)
            try
                fano = fano_factor(spikes, τs)
            catch e
                @debug "Fano factor calculation failed" exception=(e, catch_backtrace())
            end
            savetimeseries(datadir("mean_field_sweep", fanofile), fano)
        end
        return true
    end
end
simsave(; kwargs...) = (x...,) -> simsave(x...; kwargs...)
