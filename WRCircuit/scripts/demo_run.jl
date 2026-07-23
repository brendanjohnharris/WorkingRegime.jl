#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate :WRCircuit
using Bootstrap
using JLD2
using LinearAlgebra
using Optim
using ForwardDiff # activates TimeseriesTools OptimExt (MAPPLE fit!)
using MoreMaps
using StatsBase: Histogram, merge!   # `fit` comes in via @preamble (shared StatsAPI.fit generic)
using Logging, TerminalLoggers       # for the scoped progress-bar logger (with_logger) in the simulate block
WRCircuit.@preamble
set_theme!(fathom(:physics))
outfile = datadir("demo_run.jld2")

begin
    model = WRCircuit.models.Spatial
    begin # FNS parameters
        rho = 20000
        dx = 0.5
        delta = 3.5
        sigma_ee = 0.06  # from decay=7.5
        sigma_ei = 0.07  # from decay=9.5
        sigma_ie = 0.14  # from decay=19
        sigma_ii = 0.14  # from decay=19
        K_ee = 260
        K_ei = 340
        K_ie = 225
        K_ii = 290
        nu = 10.0
        n_ext = 100
        Delta_g_K = 0.002
    end
end

begin
    tmax = 25u"s" #55u"s"
    tmin = 5u"s" # The transient. Simulations always begin at 0
    fixed_params = (;
        rho,
        dx,
        delta,
        sigma_ee,
        sigma_ei,
        sigma_ie,
        sigma_ii,
        K_ee,
        K_ei,
        K_ie,
        K_ii,
        nu,
        n_ext,
        Delta_g_K,
        key = WRCircuit.PRNGKey(52),
    )
end

begin # * Run simulation
    m = model(; fixed_params...)
    sol = with_logger(TerminalLogger()) do
        simulate(
            m, tmax; populations = [:E, :I],
            vars = (; E = [:spike, :V, :input], I = [:spike])
        )
    end
    x = bpformat(sol; populations = [:E], vars = [:spike, :V, :input], transient = tmin)
    xI = bpformat(sol; populations = [:I], vars = [:spike], transient = tmin)
    epositions = [collect(pos) for pos in sol[:E].positions]
    ipositions = [collect(pos) for pos in sol[:I].positions]
end


# * Mean spectrum/MAD fits (median across neurons)
function fit_spectrum(s; components, peaks, f_range)
    negdims = [i for i in 1:ndims(s) if i != dimnum(s, 𝑓)] |> Tuple
    original_s = deepcopy(s)
    original_s = ustripall(original_s)
    original_s = median(original_s, dims = negdims)
    original_s = dropdims(original_s, dims = negdims)

    s = s[𝑓 = f_range] |> ustripall
    s = median(s, dims = negdims)
    s = dropdims(s, dims = negdims)
    _s = logsample(s)
    m = fit(MAPPLE, _s; components, peaks)
    fit!(m, _s)
    fitted_s = predict(m, s)
    return (; m, s = original_s, fitted_s, _s)
end
function fit_mad(s; components, peaks, tau_range)
    negdims = [i for i in 1:ndims(s) if i != dimnum(s, 𝑡)] |> Tuple
    s = s[𝑡 = tau_range] |> ustripall
    s = median(s, dims = negdims)
    s = dropdims(s, dims = negdims)
    m = fit(MAPPLE, s; components, peaks)
    fit!(m, s)
    fitted_s = predict(m, s)
    return (; m, s, fitted_s)
end

# * Per-neuron fits
function fit_spectrums(s::AbstractVector; components, peaks, f_range)
    s = s[𝑓 = f_range] |> ustripall
    _s = logsample(s)
    m = fit(MAPPLE, _s; components, peaks)
    fit!(m, _s)
    fitted_s = predict(m, s)
    return (; m, s, fitted_s, _s)
end
function fit_spectrums(s::AbstractMatrix; kwargs...)
    return map(eachcol(s)) do v
        fit_spectrums(v; kwargs...)
    end
end
function fit_mads(s::AbstractVector; components, peaks, tau_range)
    s = s[𝑡 = tau_range] |> ustripall
    m = fit(MAPPLE, s; components, peaks)
    fit!(m, s)
    fitted_s = predict(m, s)
    return (; m, s, fitted_s)
end
function fit_mads(s::AbstractMatrix; kwargs...)
    return map(eachcol(s)) do v
        try
            fit_mads(v; kwargs...)
        catch
            return NaN
        end
    end
end

function stream_hist(chunks, edges)
    H = nothing
    for c in chunks
        h = fit(Histogram, c, edges)
        H = H === nothing ? h : merge!(H, h)
    end
    centers = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
    return ToolsArray(H.weights ./ (sum(H.weights) * step(edges)), Dim{:bin}(centers))
end

begin # * Fano factor
    @info "Calculating Fano factor"
    spikes = x[Population = At(:E), Var = At(:spike)]
    dt = step(spikes)
    τs = logrange(dt * 10 |> ustrip, dt * 10000 |> ustrip, length = 200) # ms
    fano = fano_factor(ustripall(spikes), τs)

    mfano = map(Chart(ProgressLogger(), Threaded()), eachcol(fano)) do x
        ma = fit(MAPPLE, x; components = 3, peaks = 0)
        fit!(ma, x)
        return ma.params.components.β |> maximum
    end
end

begin # * Build LFP, membrane potential, and input traces
    V = x[Population = At(:E), Var = At(:V)]
    V = set(V, 𝑡 => convert2(u"s", times(V)))

    N = lookup(V, Neuron) |> length |> sqrt |> Int
    _V = reshape(V, (size(V, 1), N, N))

    block_size = 10
    num_row_blocks = div(size(_V, 2), block_size)
    num_col_blocks = div(size(_V, 3), block_size)

    idxs = [
        (
                (i * block_size + 1):((i + 1) * block_size),
                (j * block_size + 1):((j + 1) * block_size),
            )
            for i in 0:(num_row_blocks - 1), j in 0:(num_col_blocks - 1)
    ]

    LFP = map(idxs) do (i, j)
        m = _V[:, i, j] # * Get local patch
        m = mean(m, dims = (2, 3))
        m = ToolsArray(vec(m), dims(V, 𝑡))
    end
    LFP = ToolsArray(LFP[:], Obs(1:length(LFP))) |> stack

    input = x[Population = At(:E), Var = At(:input)]
    input = set(input, 𝑡 => convert2(u"s", times(V)))
end

begin # * Calculate spectra and MAD
    vars = (; V = V[:, 1:10:end], LFP = LFP[:, 1:10:end], input = input[:, 1:10:end])

    @info "Calculating spectra"
    spectra = map(Chart(Threaded()), vars) do v
        spectrum(v .- mean(v, dims = 𝑡), 1.0u"Hz", padding = 5000)
    end
    @info "Calculating MADs"
    mads = map(Chart(Threaded()), vars) do v
        madev(v, round.(Int, logrange(10, 10000, length = 100) |> unique) .* step(v))
    end
end

begin # * Fits
    f_range = 10u"Hz" .. 1000u"Hz"
    tau_range = 0u"s" .. 1u"s"
    @info "Fitting spectra"
    spectrum_fit = map(Chart(Threaded(), ProgressLogger()), spectra) do s
        fit_spectrum(s; components = 1, peaks = 1, f_range)
    end
    spectrum_fits = map(Chart(Threaded(), ProgressLogger()), spectra) do s
        fit_spectrums(s; components = 1, peaks = 1, f_range)
    end
    @info "Fitting MADs"
    mad_fit = map(Chart(Threaded(), ProgressLogger()), mads) do m
        fit_mad(m; components = 2, peaks = 0, tau_range)
    end
    mad_fits = map(Chart(Threaded(), ProgressLogger()), mads) do m
        fit_mads(m; components = 2, peaks = 0, tau_range)
    end
end

begin # * Fit input current distribution (Stable, per neuron)
    @info "Fitting input distributions"
    ds = map(Chart(Threaded()), eachslice(input[1:10:end, :], dims = Neuron)) do v
        fit(Stable, v)
    end
    αs = getfield.(ds, :α)
    βs = getfield.(ds, :β)
    μs = getfield.(ds, :μ)
    σs = getfield.(ds, :σ)
end

begin # * Precompute plot scalars and distributions (over the full trace, before we drop it)
    @info "Precomputing scalars and distributions"
    mean_V = mean(ustripall(V)[1:50:end, :])                                          # mean membrane potential (mV)
    nu = sum(spikes) ./ size(spikes, 2) ./ uconvert(u"s", duration(spikes)) |> ustrip # E firing rate (Hz)
    V_hist = stream_hist((ustrip.(c) for c in eachcol(V)), -70:0.1:-50)                # membrane-potential density (all data, streamed per neuron)
    dI_hist = stream_hist((abs.(diff(ustrip.(c))) for c in eachcol(input)), 0:0.1:4)   # |ΔI| step-size density (all neurons, full time, streamed)
end

begin # * Save the last 5 s of raw data (traces, input field, E/I raster) --- not the full trace
    @info "Saving raw data (last 5 s) to $(outfile)"
    Espike = x[Population = At(:E), Var = At(:spike)]
    t_end = maximum(times(Espike))
    win = (t_end - 5u"s") .. t_end
    tagsave(
        outfile,
        Dict(
            "E_spike" => Espike[𝑡 = win],
            "E_V" => x[Population = At(:E), Var = At(:V)][𝑡 = win],
            "E_input" => x[Population = At(:E), Var = At(:input)][𝑡 = win],
            "I_spike" => xI[Population = At(:I), Var = At(:spike)][𝑡 = win],
            "epositions" => epositions,
            "ipositions" => ipositions,
            "fixed_params" => fixed_params,
            "N" => N,
            "mean_V" => mean_V,
            "nu" => nu,
        ), safe = true
    )
end

begin # * Save derived statistics
    statsfile = datadir("demo_run_stats.jld2")
    @info "Saving derived statistics to $(statsfile)"
    jldsave(
        statsfile;
        fano, mfano, spectra, mads,
        spectrum_fit, spectrum_fits, mad_fit, mad_fits,
        αs, βs, μs, σs, V_hist, dI_hist
    )
end
