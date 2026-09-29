#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.13 -t auto --heap-size-hint=45G --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# How much of the input diffusion exponent `a` is due to synchronous spike arrival? For each connectome seed,
# one run at the working-regime defaults, as long as the demo run (Fig 1), records every spike,
# plus `itot` and `V` of a random 10% of E neurons. Each target's input is rebuilt from every spike incident on
# it (E→E, I→E and the external drive) with the engine's synapse update, and an isolated copy of the target is
# replayed on it, so V and the driving force follow the input. One surrogate per neuron and mode changes only
# the arrival times inside the analysis window:
#   circular: each incoming train rotated by its own random lag (no synchrony; each train's structure kept)
#   uniform:  each arrival moved to a uniform random step (Poisson arrivals)
# Replaying the real arrivals reproduces the recorded run (`match`, `Ierr`). `a` is saved per neuron and as
# Fig 1 quotes it (one fit to the neuron-median MAD curve). Needs a GPU; ~4.5 min per seed on an L40S.
using DrWatson
DrWatson.@quickactivate :WRCircuit
using Dewdrop, CUDA
using Optim
using ForwardDiff # activates TimeseriesTools OptimExt (MAPPLE fit!)
import WRCircuit: SWEEP # run length, transient, MAD lags: shared with the sweep
WRCircuit.@preamble

begin
    seeds = 1:10 # circuit_sweep.jl's connectome seeds
    frac = 0.1 # of E neurons per seed
    tmax = 55u"s" # demo_run.jl (Fig 1); the first SWEEP.transient_steps (5 s) are dropped
    modes = (:real, :circular, :uniform)
    outdir = datadir("synchrony_surrogates")
end

# Incident edges of every target, grouped by synapse kinetics (E→E + drive; I→E): arrival steps (presynaptic
# spike steps + delay) and weight per edge. Drive spikes are regenerated from `PoissonSource`'s counter RNG.
function incident(net, targets, spk, dt, T, nsteps)
    col = Dict(t => i for (i, t) in enumerate(targets))
    groups = [Dict{Any, Vector{Tuple{Vector{Int}, Float64}}}() for _ in targets]
    for p in net.projections
        syn = p.synapse
        ext = syn isa Dewdrop.PoissonSource
        conn = ext ? syn.extconn : p.conn
        post, src, w, d = Array(conn.post), Array(conn.src), Array(conn.weight), Array(conn.delay)
        e = findall(in(keys(col)), post)
        isempty(e) && continue
        D = eltype(d) <: Integer ? Int.(d[e]) : Dewdrop._ms_to_steps.(d[e], dt)
        c = map(Float64, Dewdrop._syn_coeffs(ext ? syn.synapse : syn, dt, T))
        trains = if ext
            pfire = T(syn.rate * dt / 1000) # PoissonSource p_spike
            srcs = unique(src[e])
            fired = map(Chart(Threaded()), srcs) do j
                [n for n in 0:(nsteps - 1) if Dewdrop.draw_uniform(Float64, syn.seed, n, j) < pfire]
            end
            Dict(zip(srcs, fired))
        else
            spk
        end
        for (k, x) in enumerate(e)
            push!(get!(groups[col[post[x]]], c, Tuple{Vector{Int}, Float64}[]), (trains[src[x]] .+ D[k], Float64(w[x])))
        end
    end
    return groups
end

# Weighted arrivals per step. Inside [n0, nsteps), `:circular` rotates each edge's train by one random lag and
# `:uniform` redraws each arrival's step; earlier arrivals stay put.
function arrivals(edges, nsteps, n0, mode, rng)
    s = zeros(nsteps)
    nW = nsteps - n0
    for (a, w) in edges
        k = rand(rng, 0:(nW - 1))
        for t in a
            t < nsteps || continue
            if t ≥ n0 && mode !== :real
                t = mode === :uniform ? rand(rng, n0:(nsteps - 1)) : n0 + mod(t - n0 + k, nW)
            end
            s[t + 1] += w
        end
    end
    return s
end

# The engine's dual-exponential update: kick both accumulators on arrival, read g = a(d − r), decay.
function conductance(s, c)
    g = similar(s)
    r = d = 0.0
    for n in eachindex(s)
        r += s[n]; d += s[n]
        g[n] = c.a * (d - r)
        r *= c.decay_r; d *= c.decay_d
    end
    return g
end

# An isolated FNS neuron on conductances `gs`, stepped as the engine does (frozen current from the
# start-of-step V, adaptation decay, membrane update clamped at Vr while refractory, threshold, reset). Kept in
# the engine's float type: in Float32 the 4 ms refractory countdown takes 41 steps of 0.1 ms, in Float64 40.
# Index n is step n − 1, as in the monitors; starts from the recorded V after step 0.
function replay(m, gs, V0, nsteps, dt::T) where {T}
    I = zeros(T, nsteps)
    spikes = Int[]
    v, w, refrac = T(V0), zero(T), zero(T)
    for n in 2:nsteps
        i = T(sum(g[n] * (c.Erev - v) for (c, g) in gs))
        w = Dewdrop._step_w(m, v, w, dt)
        v = refrac > 0 ? Dewdrop.reset_value(m) : Dewdrop._step_V(m, v, w, zero(T), i, dt)
        refrac = max(refrac - dt, zero(T))
        if refrac ≤ 0 && Dewdrop.threshold(m, v)
            v, refrac = Dewdrop.reset_value(m), Dewdrop.refractory(m)
            w += Dewdrop.spike_increment(m)
            push!(spikes, n - 1)
        end
        I[n] = i
    end
    return (; I, spikes)
end

function surrogate_seed(seed; frac, modes, tmax)
    dt = SWEEP.dt_ms
    n0 = SWEEP.transient_steps
    spec = WRCircuit.build_spatial(; seed = UInt64(seed), arch = WRCircuit.DEWDROP_BACKEND())
    net = Dewdrop.materialize(spec, FixedStep(dt); tspan = (0.0, ustrip(u"ms", tmax)))
    T = Dewdrop.float_type(net.model)
    E = net.subpops[:E]
    targets = sort(shuffle(Xoshiro(seed), collect(E))[1:round(Int, frac * length(E))])
    sol = solve(
        net, FixedStep(dt); v0 = (-70.0, -50.0), progress = false, scatter = :compacted,
        record = (; spikes = Spikes(), itot = Trace(:itot; of = targets), V = Trace(:V; of = targets))
    )
    nsteps = sol.nsteps
    W = (n0 + 1):nsteps
    Tw = length(W) * dt / 1000 # s
    itot = permutedims(sol.record.itot.data) # step × target
    V0 = sol.record.V.data[:, 1]
    S = sol.record.spikes.data # neuron × step
    spk = [Int[] for _ in axes(S, 1)] # 0-based spike steps
    for n in axes(S, 2), j in findall(view(S, :, n))
        push!(spk[j], n - 1)
    end
    groups = incident(net, targets, spk, dt, T, nsteps)
    neurons = CUDA.@allowscalar [Dewdrop._resolve(net.model, j) for j in targets]

    res = map(Chart(Threaded(), LogLogger()), eachindex(targets)) do i
        rec = itot[W, i]
        out = map(modes) do mode
            rng = Xoshiro(hash((seed, targets[i], mode)))
            gs = [(c, conductance(arrivals(e, nsteps, n0, mode, rng), c)) for (c, e) in groups[i]]
            replay(neurons[i], gs, V0[i], nsteps, sol.dt)
        end
        recspk = filter(≥(n0), spk[targets[i]])
        (;
            mad = reduce(hcat, [madev(rec, SWEEP.mad_lags), (madev(o.I[W], SWEEP.mad_lags) for o in out)...]),
            rate = [count(≥(n0), o.spikes) / Tw for o in out],
            match = length(intersect(filter(≥(n0), out[1].spikes), recspk)) / max(1, length(recspk)),
            Ierr = sqrt(mean(abs2, out[1].I[W] .- rec)) / std(rec),
        )
    end

    labels = Neuron(Symbol.(:E, targets))
    cond = Dim{:condition}([:recorded, modes...])
    mad = ToolsArray(stack((r.mad for r in res); dims = 2), (𝑡(SWEEP.τs), labels, cond))
    a = ToolsArray(stack([diffusion_exponents(mad[condition = At(k)]) for k in lookup(cond)]), (labels, cond))
    a_curve = ToolsArray([only(diffusion_exponents(median(mad[condition = At(k)]; dims = Neuron))) for k in lookup(cond)], (cond,))
    rate = ToolsArray(stack((r.rate for r in res); dims = 1) .* u"Hz", (labels, Dim{:mode}(collect(modes))))
    return Dict(
        "parameters" => (; SWEEP.model_defaults..., seed), "targets" => targets,
        "mad" => mad, "a" => a, "a_curve" => a_curve, "rate" => rate,
        "match" => [r.match for r in res], "Ierr" => [r.Ierr for r in res],
    )
end

begin # * Run every seed; one resumable file per seed
    files = map(seeds) do seed
        _, file = produce_or_load((; seed), outdir) do c
            @info "Seed $(c.seed)"
            surrogate_seed(c.seed; frac, modes, tmax)
        end
        GC.gc()
        return file
    end
end

begin # * Figure input: every seed's exponents as plain arrays (Fig 1 statistics)
    ds = load.(files)
    conditions = string.(lookup(first(ds)["a"], :condition))
    a_curve = permutedims(stack(parent(d["a_curve"]) for d in ds)) # seed x condition
    tagsave(
        rootdatadir("synchrony_surrogates.jld2"), Dict(
            "seeds" => collect(seeds), "conditions" => conditions, "tmax_s" => ustrip(u"s", tmax), "frac" => frac,
            "a" => stack(parent(d["a"]) for d in ds), # neuron × condition × seed
            "a_curve" => a_curve, "spikes_reproduced" => [mean(d["match"]) for d in ds],
        )
    )
    Δ = a_curve[:, findfirst(==("circular"), conditions)] .- a_curve[:, findfirst(==("recorded"), conditions)]
    @info "Δa (circular − recorded), fit to the neuron-median MAD curve" mean = mean(Δ) sd = std(Δ)
end
