#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
WRCircuit.@preamble
set_theme!(foresight(:physics))
#! /bin/bash
# -*- mode: julia -*-
#=
exec $HOME/build/julia-1.11.2/bin/julia -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Load stats
    load_stats, sweep_parameters, fixed_parameters, metadata = WRCircuit.stats.load("nonspatial_sweep.pickle")

    tmax = metadata[:tmax].val |> convert2(Float32)
    tmin = metadata[:tmin].val |> convert2(Float32)
    mua_dt = metadata[:mua_dt].val |> convert2(Float32)
    mua_dt = mua_dt / 1000
end
begin
    begin # * Extract a ToolsArray of one statistic
        deltas = sweep_parameters["delta"] |> convert2(Array)
        deltas = reshape(deltas, (length(unique(deltas)),))
        rate = load_stats["rate"]["E.spike"] |> convert2(Array)
        rate = reshape(rate, size(deltas)..., size(rate, 2))
        meanrate = dropdims(mean(rate, dims = 2), dims = 2)
    end
    begin
        f = Figure()
        ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "mean rate (Hz)")
        lines!(ax, sort(unique(deltas)), meanrate[:]; alpha = 0.85)
        f |> display
    end
    begin # * Susceptibility
        susceptibility = load_stats["susceptibility"]["E.spike"] |> convert2(Array)
        susceptibility = reshape(susceptibility, size(deltas)..., size(susceptibility, 2))
        mean_susceptibility = dropdims(mean(susceptibility, dims = 2), dims = 2)
        f = Figure()
        ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "susceptibility")
        lines!(ax, sort(unique(deltas)), mean_susceptibility[:]; alpha = 0.85)
        f
    end
end

begin # * Load mua
    _mua = load_stats["mua"]["E.spike"] |> convert2(Array)
    ts = range(tmin, stop = tmax, step = mua_dt)[2:end] |> ustripall

    delta = sort(unique(deltas))
    nu = sort(unique(nus))

    _mua = reshape(_mua, length(delta), length(nu), size(_mua, 2))
    mua = ToolsArray(_mua, (Dim{:delta}(delta), Dim{:nu}(nu), 𝑡(ts)))
    mua = permutedims(mua, (3, 1, 2))
end
begin
    mumua = mua .- mean(mua, dims = 𝑡)
    spectra = map(x -> spectrum(x, 0.8, padding = 500), eachslice(mumua, dims = (2, 3))) |>
              stack
    plot(spectra[delta = Near(4.0), nu = Near(7)])
end
begin
    # * plot a heatmap for a given nu
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "frequency (Hz)", ylabel = "δ")
    x = spectra[nu = Near(6.5)]
    # x = x[nu = 1.0 .. 5.0]
    x = x[𝑓 = IntervalSets.OpenInterval(0, 200)]
    x = log10.(x)
    # x = x ./ maximum(x, dims = 2)
    colorrange = extrema(x)
    heatmap!(ax, x, colormap = (pelagic |> reverse))
    f
end
