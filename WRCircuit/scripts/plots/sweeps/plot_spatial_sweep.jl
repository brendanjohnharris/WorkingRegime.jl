#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
using DataInterpolations
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Load stats
    load_stats, sweep_parameters, fixed_parameters, metadata = WRCircuit.stats.load("spatial_sweep.pickle")

    tmax = metadata[:tmax].val |> convert2(Float32)
    tmin = metadata[:tmin].val |> convert2(Float32)
    mua_dt = metadata[:mua_dt].val |> convert2(Float32)
    mua_dt = mua_dt / 1000
end
function extract_statistic(stat, sweep_parameters)
    maybescalar(x) = length(x) == 1 ? only(x) : x
    maybecollect(x) = isa(x, Number) ? x : collect(x)
    stat = map(maybecollect, stat)
    paramnames = sweep_parameters |> collect .|> Symbol
    params = sweep_parameters.values() |> collect .|> convert2(Array) .|> eachrow

    paramdims = unique.(params) .|> sort
    paramgrid = Iterators.product(paramdims...) |> collect
    X = Array{Union{Missing, eltype(stat)}}(missing, length.(paramdims)...)

    for (i, param) in enumerate(zip(params...))
        idx = findfirst(==(param), paramgrid)
        X[idx] = stat[i] |> maybecollect
    end
    ddims = [Dim{Symbol(n)}(maybescalar.(p)) for (n, p) in zip(paramnames, paramdims)] |>
            Tuple
    return ToolsArray(X, ddims)
end

begin # * Set up figure
    fig = Figure(size = (600, 2000))
    gs = subdivide(fig, 4, 1)
end

begin # * Mean rate
    rate = load_stats["rate"]["E.spike"] |> convert2(Array) |> eachrow
    rate = extract_statistic(rate, sweep_parameters)
    meanrate = mean.(rate)
    _μ, (_σₗ, _σₕ) = bootstrapaverage(mean, meanrate, dims = (1, 2))
end
begin
    μ = upsample(_μ, 10)
    σₗ = upsample(_σₗ, 10)
    σₕ = upsample(_σₕ, 10)

    f = gs[1]
    ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "Mean rate (Hz)")
    band!(ax, lookup(σₗ)[1], collect.([σₗ, σₕ])...; alpha = 0.3)
    rangebars!(ax, lookup(_σₗ)[1], collect.([_σₗ, _σₕ])...; alpha = 0.3, linewidth = 1,
               whiskerwidth = 3)
    lines!(ax, μ; alpha = 0.85)
    scatter!(ax, _μ; alpha = 0.85)
    f
end

begin # * Susceptibility
    susceptibility = load_stats["susceptibility"]["E.spike"] |> convert2(Array)
    susceptibility = extract_statistic(susceptibility, sweep_parameters)
    _μ, (_σₗ, _σₕ) = bootstrapaverage(mean, susceptibility, dims = (1, 2))
end
begin
    μ = upsample(_μ, 10)
    σₗ = upsample(_σₗ, 10)
    σₕ = upsample(_σₕ, 10)

    f = gs[2]
    ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "Susceptibility")
    band!(ax, lookup(σₗ)[1], collect.([σₗ, σₕ])...; alpha = 0.3)
    rangebars!(ax, lookup(_σₗ)[1], collect.([_σₗ, _σₕ])...; alpha = 0.3, linewidth = 1,
               whiskerwidth = 3)
    lines!(ax, μ; alpha = 0.85)
    scatter!(ax, _μ; alpha = 0.85)
    f
end

begin # * Load mua
    mua = load_stats["mua"]["E.spike"] |> convert2(Array) |> eachrow
    mua = extract_statistic(mua, sweep_parameters)
    ts = range(tmin, stop = tmax, step = mua_dt)[2:end] |> ustripall
    ts = 𝑡(ts)
    mua = ToolsArray.(mua, ((ts,),))
end
begin # * Power spectrum
    S = map(mua) do x
        x = x .- mean(x)
        x = x[𝑡 = IntervalSets.OpenInterval(0, 250)]
        x = spectrum(x, 0.8, padding = 500)
    end
    S = stack(S)
    _μ, (_σₗ, _σₕ) = bootstrapaverage(mean, S, dims = (2, 3))
end
begin
    f = gs[3]
    ax = Axis(f[1, 1]; xlabel = "Frequency (Hz)", ylabel = "Spectral density",
              yscale = log10, xscale = log10, limits = ((nothing, 250), nothing))
    deltas = lookup(_μ, :delta)[1:2:end]
    map(deltas) do δ
        μ = _μ[delta = At(δ)]
        σₗ = _σₗ[delta = At(δ)]
        σₕ = _σₕ[delta = At(δ)]
        lines!(ax, μ; alpha = 0.85, color = δ, colorrange = extrema(deltas),
               colormap = sunrise |> reverse)
    end
    Colorbar(f[1, 2]; colormap = sunrise |> reverse, colorrange = extrema(deltas),
             label = "δ")
    f
end

begin # * Spectral tail exponent
    α = map(eachslice(S, dims = (:key, :nu, :delta))) do s
        # * Fit a fooof model, ensuring log-linear spacing
    end
end
begin # * Grand distribution
    G, bins = load_stats["grand_distribution"]["E.input"] .|> convert2(Array)
    G = extract_statistic(G |> eachrow, sweep_parameters)
    bins = extract_statistic(bins |> eachrow, sweep_parameters)
    b = mean(bins)
    G = map(G) do g
        ToolsArray(g, (Dim{:bin}(b),))
    end
    G = stack(G)
    G = mean(G, dims = (2, 3))
    G = dropdims(G, dims = (2, 3))
end
begin
    deltas = lookup(G, :delta)[1:4:end]
    f = gs[4]
    ax = Axis(f[1, 1]; xlabel = "Input", ylabel = "Frequency", yscale = log10,
              xscale = Makie.pseudolog10)
    map(deltas) do δ
        g = G[delta = At(δ)]
        lines!(ax, g; alpha = 0.85, color = δ, colorrange = extrema(deltas),
               colormap = sunrise |> reverse)
    end
    Colorbar(f[1, 2]; colormap = sunrise |> reverse, colorrange = extrema(deltas),
             label = "δ")
    f
end
display(fig)
save(plotdir("plot_spatial_sweep", "spatial_sweep.pdf"), fig)

# begin # * Load mua
#     _mua = load_stats["mua"]["E.spike"] |> convert2(Array)
#     ts = range(tmin, stop = tmax, step = mua_dt)[2:end] |> ustripall

#     delta = sort(unique(deltas))
#     nu = sort(unique(nus))

#     _mua = reshape(_mua, length(delta), length(nu), size(_mua, 2))
#     mua = ToolsArray(_mua, (Dim{:delta}(delta), Dim{:nu}(nu), 𝑡(ts)))
#     mua = permutedims(mua, (3, 1, 2))
# end
# begin
#     mumua = mua .- mean(mua, dims = 𝑡)
#     spectra = map(x -> spectrum(x, 0.8, padding = 500), eachslice(mumua, dims = (2, 3))) |>
#               stack
#     plot(spectra[delta = Near(4.0), nu = Near(7)])
# end
# begin
#     # * plot a heatmap for a given nu
#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "frequency (Hz)", ylabel = "δ")
#     x = spectra[nu = Near(6.5)]
#     # x = x[nu = 1.0 .. 5.0]
#     x = x[𝑓 = IntervalSets.OpenInterval(0, 200)]
#     x = log10.(x)
#     # x = x ./ maximum(x, dims = 2)
#     colorrange = extrema(x)
#     heatmap!(ax, x, colormap = (pelagic |> reverse))
#     f
# end
