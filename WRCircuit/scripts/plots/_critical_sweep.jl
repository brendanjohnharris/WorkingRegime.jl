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
using Optim
using MoreMaps
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Load sweep parameters
    files = readdir(datadir("critical_sweep"), join = true)
    ps = map(files) do f
        fname = parse_savename(f; connector = string(connector))[2]
        if haskey(fname, "key")
            return nothing
        elseif haskey(fname, "Delta_g_K") && haskey(fname, "delta")
            return fname
        else
            return nothing
        end
    end
    filter!(!isnothing, ps)
    deltas = [p["delta"] for p in ps]
    Delta_g_Ks = [p["Delta_g_K"] for p in ps]
    parameter_grid = Iterators.product(
        Dim{:delta}(deltas |> unique),
        Dim{:Delta_g_K}(Delta_g_Ks |> unique)
    ) |>
        collect
    parameter_grid = map(parameter_grid) do (d, gk)
        idx = findfirst((deltas .== d) .& (Delta_g_Ks .== gk))
        if isnothing(idx)
            return idx
        else
            return files[idx]
        end
    end

    this_delta = 4.0
    this_Delta_g_K = 0.003
    parameter_grid = parameter_grid[delta = 3.3 .. 5]
    select_grid = parameter_grid[Delta_g_K = At(this_Delta_g_K)]
    @assert sum(isnothing, select_grid) == 0
end

begin # * Load PSD data
    psd = map(Chart(Threaded(), ProgressLogger()), select_grid) do f
        load(f, "inputs/psd")
    end
end

begin # * Load MAD data
    mad = map(Chart(Threaded(), ProgressLogger()), select_grid) do f
        load(f, "inputs/mad")
    end
end

# ? First, variation with delta

begin # * Plot Psd for one delta
    p = median(psd[delta = Near(4.5)], dims = Neuron)
    p = dropdims(p, dims = Neuron)

    f = Figure()
    ax = Axis(f[1, 1]; yscale = log10) #, xscale = log10)
    lines!(ax, ustripall(p)[𝑓 = eps() .. 150])
    display(f)
end

if false # * Plot psd heatmap over deltas
    f = Figure()
    p = median(stack(psd), dims = Neuron)
    p = dropdims(p, dims = Neuron)[2:5:end, :]
    p = p[𝑓 = 0u"Hz" .. 150u"Hz"]

    p = p ./ p[:, 1]

    ax = Axis(
        f[1, 1];
        xlabel = "Frequency (Hz)", ylabel = "Delta"
    )
    heatmap!(ax, ustripall(p); colorscale = log10)
    display(f)
end

begin # * Fit psds
    bs = map(psd) do x
        # p = median(p, dims = Neuron)
        a = map(eachcol(x[:, 1:50:end])) do y
            # * MAPPLE fit
            fmax = 1000
            _p = y[𝑓 = 10u"Hz" .. fmax * u"Hz"]
            _p = logsample(ustripall(_p))
            m = fit(MAPPLE, _p; components = 1, peaks = 0)
            fit!(m, _p)
            return m.params.components.β |> last
        end
        return mean(a)
    end
    lines(bs)
end

begin # * Plot mad for one delta
    p = median(mad[delta = Near(4.0)], dims = Neuron)
    p = dropdims(p, dims = Neuron) |> ustripall

    # * MAPPLE fit
    m = fit(MAPPLE, p; components = 2, peaks = 0)
    fit!(m, p)
    _m = predict(m, lookup(p, 𝑡))

    f = Figure()
    ax = Axis(f[1, 1]; xscale = log10, yscale = log10)
    lines!(ax, ustripall(p))
    lines!(ax, lookup(p, 𝑡), _m)
    display(f)
    display(m.params.components |> first)
end

begin # * Fit mads
    as = map(mad) do x
        # p = median(p, dims = Neuron)
        a = map(eachcol(x[:, 1:50:end])) do y
            # p = dropdims(p, dims = Neuron) |> ustripall
            y = ustripall(y)
            m = fit(MAPPLE, y; components = 2, peaks = 0)
            fit!(m, y)
            return m.params.components.β |> first
        end
        return mean(a)
    end
    lines(as)
end

if false # * Plot mad heatmap over deltas
    f = Figure()
    p = median(stack(mad), dims = Neuron)
    p = dropdims(p, dims = Neuron)[2:5:end, :]
    p = p[𝑓 = 0u"Hz" .. 150u"Hz"]

    p = p ./ p[:, 1]

    ax = Axis(
        f[1, 1];
        xlabel = "Frequency (Hz)", ylabel = "Delta"
    )
    heatmap!(ax, ustripall(p); colorscale = log10)
    display(f)
end

# * Now for sweep over Δg_K

# function susceptibility(x; dt = 100u"ms")
#     # * First bin into ms bins
#     X = groupby(x, 𝑡 => Base.Fix2(WRCircuit.group_dt, dt))

#     # * Active neurons
#     X = map(X) do x
#         sum(x, dims = 𝑡) .> 0
#     end

#     # * Fraction of active neurons at each time step
#     rho = mean.(X)

#     # * Susceptibility
#     chi = mean(rho .^ 2) - mean(rho)^2
# end

# ! Now, variation with Delta_g_K (fixed delta = 4.0)

begin # * Load sweep parameters for Delta_g_K sweep
    gk_parameter_grid = parameter_grid[delta = Near(this_delta)]
    @assert sum(isnothing, gk_parameter_grid) == 0
end

begin # * Load PSD data for Delta_g_K sweep
    gk_psd = map(Chart(Threaded(), ProgressLogger()), gk_parameter_grid) do f
        load(f, "inputs/psd")
    end
end

begin # * Load MAD data for Delta_g_K sweep
    gk_mad = map(Chart(Threaded(), ProgressLogger()), gk_parameter_grid) do f
        load(f, "inputs/mad")
    end
end

begin # * Fit psds over Delta_g_K
    gk_bs = map(gk_psd) do x
        a = map(eachcol(x[:, 1:50:end])) do y
            # * MAPPLE fit
            fmax = 1000
            _p = y[𝑓 = 10u"Hz" .. fmax * u"Hz"]
            _p = logsample(ustripall(_p))
            m = fit(MAPPLE, _p; components = 1, peaks = 0)
            fit!(m, _p)
            return m.params.components.β |> last
        end
        return mean(a)
    end
    lines(gk_bs)
end

begin # * Fit mads over Delta_g_K
    gk_as = map(gk_mad) do x
        a = map(eachcol(x[:, 1:50:end])) do y
            y = ustripall(y)
            m = fit(MAPPLE, y; components = 2, peaks = 0)
            fit!(m, y)
            return m.params.components.β |> first
        end
        return mean(a)
    end
    lines(gk_as)
end

begin # * Plot mad and PSD over Delta_g_K on same plot
    f = TwoPanel()

    ax1 = Axis(
        f[1, 1]; xlabel = "Delta", ylabel = "Diffusion exponent",
        xgridvisible = false,
        ygridvisible = false, title = "Δg_K = $(this_Delta_g_K)"
    )
    scatterlines!(ax1, as, color = :cornflowerblue, markersize = 10)
    ax2 = Axis(
        f[1, 1]; ylabel = rich("Spectral exponent", color = :crimson),
        yaxisposition = :right,
        xgridvisible = false, ygridvisible = false
    )
    scatterlines!(ax2, bs, color = :crimson, markersize = 10)
    display(f)

    gax1 = Axis(
        f[1, 2]; xlabel = "Δg_K", ylabel = "Diffusion exponent",
        xgridvisible = false,
        ygridvisible = false, title = "δ = $(this_delta)"
    )
    scatterlines!(gax1, gk_as, color = :cornflowerblue, markersize = 10)
    gax2 = Axis(
        f[1, 2]; ylabel = rich("Spectral exponent", color = :crimson),
        yaxisposition = :right,
        xgridvisible = false, ygridvisible = false
    )
    scatterlines!(gax2, gk_bs, color = :crimson, markersize = 10)

    linkyaxes!(ax1, gax1)
    linkyaxes!(ax2, gax2)
    display(f)
end

# * Try full grid
begin # * Load MAD data for full grid
    full_mad = map(Chart(Threaded(), ProgressLogger()), parameter_grid) do f
        isnothing(f) && return nothing
        try
            mad = load(f, "inputs/mad")
            return mad
        catch
            @warn "Failed to load MAD for $f"
            return nothing
        end
    end
end

begin # * Fit diffusion exponent over full grid
    full_as = map(full_mad) do x
        isnothing(x) && return NaN
        a = map(eachcol(x[:, 1:50:end])) do y
            y = ustripall(y)
            m = fit(MAPPLE, y; components = 2, peaks = 0)
            fit!(m, y)
            return m.params.components.β |> first
        end
        return mean(a)
    end
end

begin # * Heatmap of diffusion exponent over delta and Delta_g_K
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "Δg_K")
    h = heatmap!(
        ax, ustripall(full_as), colorrange = (0.5, 0.75), lowclip = :black,
        colormap = :turbo
    )
    Colorbar(f[1, 2], h; label = "Diffusion exponent")
    display(f)
end

# * Now plot spectral exponent as a heatmap

begin # * Load PSD and fit spectral exponent over full grid
    full_bs = map(Chart(Threaded(), ProgressLogger()), parameter_grid) do f
        isnothing(f) && return NaN
        psd = try
            load(f, "inputs/psd")
        catch
            @warn "Failed to load PSD for $f"
            return NaN
        end
        a = map(eachcol(psd[:, 1:50:end])) do y
            # * MAPPLE fit
            fmax = 1000
            _p = y[𝑓 = 10u"Hz" .. fmax * u"Hz"]
            _p = logsample(ustripall(_p))
            m = fit(MAPPLE, _p; components = 1, peaks = 0)
            fit!(m, _p)
            return m.params.components.β |> last
        end
        return mean(a)
    end
end

begin # * Heatmap of spectral exponent over delta and Delta_g_K
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "Δg_K")
    h = heatmap!(ax, ustripall(full_bs), colormap = :turbo)
    Colorbar(f[1, 2], h; label = "Spectral exponent")
    display(f)
end

begin # * Plot diffusion exponent vs spectral exponent
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Diffusion exponent", ylabel = "Spectral exponent")
    lines!(
        ax, collect(as), collect(bs), color = :cornflowerblue,
        label = "δ sweep (Δg_K = $(this_Delta_g_K))"
    )

    p = lines!(
        ax, collect(gk_as), collect(gk_bs), color = :crimson,
        label = "Δg_K sweep (δ = $(this_delta))"
    )
    axislegend(ax, position = :lt)
    display(f)
end
