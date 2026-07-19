#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
using LinearAlgebra
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin
    model = WRCircuit.models.Nonspatial
    begin
        N_e = 2000
        nu = 10.0
        K_ee = 100
        K_ei = 150
        K_ie = 80
        K_ii = 90
        Delta_g_K = 0.00
    end
end

begin
    tmax = 7u"s" # * Bump up
    tmin = 2u"s" # The transient. Simulations always begin at 0
    fixed_params = (; N_e, nu, K_ee, K_ei, K_ie, K_ii, Delta_g_K)
end

begin # * Run simulation
    m = model(; fixed_params...)
    x = bpsolve(m, tmax; populations = [:E, :I], vars = [:spike, :V],
                transient = tmin)

    ispikes = x[Population = At(:I), Var = At(:spike)]
    ispiketimes = map(eachslice(ispikes, dims = Neuron)) do s
        sts = times(s)[findall(s)]
    end

    spikes = x[Population = At(:E), Var = At(:spike)]
    spiketimes = map(eachslice(spikes, dims = Neuron)) do s
        sts = times(s)[findall(s)]
    end
end

begin # * Spike raster
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Time (ms)", ylabel
              = "Neuron index", title = "Spike raster")
    hideydecorations!(ax)

    intrvl = 5000u"ms" .. 5200u"ms"
    for (i, s) in enumerate(spiketimes)
        idxs = s .∈ [intrvl]
        scatter!(ax, ustripall(s[idxs]), i * ones(sum(idxs)), color = :black,
                 markersize = 3)
    end

    ax2 = Axis(f[2, 1]; xlabel = "Time (ms)", ylabel
               = "Neuron index", title = "Spike raster (inhibitory)")
    for (i, s) in enumerate(ispiketimes)
        idxs = s .∈ [intrvl]
        scatter!(ax2, ustripall(s[idxs]), i * ones(sum(idxs)), color = :black,
                 markersize = 3)
    end
    hideydecorations!(ax2)
    linkxaxes!(ax, ax2)
    display(f)
end

begin # * Membrane potential
    V = x[Population = At(:E), Var = At(:V)][1:10:end, :]
    V = set(V, 𝑡 => uconvert.(u"s", times(V)))
    V = rectify(V, dims = 𝑡)
    lines(V[1:300, 1000]) |> display
end

# begin # * Unit spectrum
#     s = spectrum(V .- mean(V, dims = 𝑡), 0.1)
#     s = mean(s, dims = Neuron)
#     s = ustripall(dropdims(s, dims = Neuron))[𝑓 = 0.1 .. 200]
#     lines(s; axis = (; xscale = log10, yscale = log10)) |> display
# end

begin # * MUA spectrum
    rates = WRCircuit.compute_rates(spikes, 50u"ms")
    mdt = 3.0u"ms"
    mua = groupby(spikes, 𝑡 => Base.Fix2(WRCircuit.group_dt, mdt))
    mua = map(mua) do r
        dropdims(sum(r, dims = 𝑡), dims = 𝑡) ./ uconvert(u"s", mdt)
    end |> stack
    mua = permutedims(mua, (𝑡, Neuron))
    mua = rectify(mua, dims = 𝑡)
    N = lookup(mua, Neuron) |> length |> Int
    ts = dims(mua, 𝑡)
    muas = sum(mua, dims = Neuron)
    muas = dropdims(muas, dims = Neuron) .- mean(muas)
    muas = set(muas, 𝑡 => uconvert.(u"s", times(muas)))
    muas = rectify(muas, dims = 𝑡)
    muas = spectrum(muas, 0.5u"s")
    lines(ustripall(muas)[𝑓 = 1 .. 200]; axis = (; xscale = log10, yscale = log10)) |>
    display
end
begin # * Plot inter-spike interval distribution
    isis = map(spiketimes) do sts
        diff(sts)
    end
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Inter-spike interval (ms)",
              ylabel = "Density", limits = ((0, 500), nothing))
    hist!(ax, Iterators.flatten(isis) |> collect |> ustrip, normalization = :pdf,
          bins = range(0.0, 500, step = 1))
    display(f)
end
