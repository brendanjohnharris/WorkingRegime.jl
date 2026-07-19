#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
WRTheory.@preamble()
import FractionalNeuralSampling.Samplers: gen_lfsm_fns
set_theme!(foresight(:physics))

transient = 5000.0 # ms
η = 0.03
γ = 0.01
β = 0.8

begin # * Load parameters
    metadata = load(datadir("mean_field_sweep", "metadata.jld2"))
    @unpack named, unnamed = metadata
    params = values(metadata["params"])
    parameter_grid = Iterators.product(params...) |> collect
    parameter_grid = parameter_grid[η = At([η]), γ = At([0.0, γ])]

    files = map(Chart(Threaded(), ProgressLogger(1000)), parameter_grid) do p
        ps = Dict(name.(dims(parameter_grid)) .=> p)
        # hash = Base.hash(unnamed)
        filepath = savename((; ps..., named...), "tsv")
    end
    quality = map(Chart(Threaded()), files) do file
        isfile(datadir("mean_field_sweep", file))
    end
    @assert mean(quality) == 1.0

    @info "Loading spike times..."
    spikes = map(Chart(Threaded(), ProgressLogger()), files) do file
        file = datadir("mean_field_sweep", file)
        if filesize(file) == 0
            return []
        else
            s = readdlm(file) |> vec
            return s[s .> transient]
        end
    end
    @info "Loading fano factors..."
    fanos = map(Chart(Threaded(), ProgressLogger()), files) do file
        loadtimeseries(datadir("mean_field_sweep", "fano_$file"))
    end |> stack
    times(fanos) ./= 1000
    rates = map(spikes) do s
        rate = length(s) / (named.tspan - transient) * 1000
    end
end
begin # * Mapple fano curve fits
    fann = fanos[η = Near(η), γ = Near(γ)]
    fann = eachslice(fann, dims = setdiff(dims(fann), [dims(fann, 𝑡)]) |> Tuple)
    cs = map(Chart(ProgressLogger(), Threaded()), fann) do ff
        try
            m = fit(MAPPLE, ff; peaks = 0, components = 3)
            fit!(m, ff)
            c = maximum(m.params.components.β) # Max slope
        catch
            return NaN
        end
    end

    mcs = mapslices(cs, dims = Obs) do m
        if all(isnan, m)
            return NaN
        else
            return nansafe(median)(m)
        end
    end
    mcs = dropdims(mcs, dims = Obs)
end

begin
    f = Figure()
    fixeds = [Dim{:α}(Near(1.5)), Dim{:β}(Near(β)), Dim{:η}(At(η)), Dim{:γ}(At(0.0))]
    ff = fanos[fixeds...]
    ax = Axis(f[1, 1]; xscale = log10, yscale = log10,
              xlabel = "Time bin (s)", ylabel = "Fano factor",
              title = "Fano factor at $(fixeds)")
    # ff = Dropdims(mean)(ff, dims = Obs)
    ff = ff[:, 4]
    lines!(ax, ff)

    m = fit(MAPPLE, ff; peaks = 0, components = 3)
    fit!(m, ff)
    c = m.params.components.β[2] # Middle slope

    p = lines!(ax, lookup(ff, 𝑡), predict(m, lookup(ff, 𝑡)); label = "MAPPLE fit",
               color = :red, linestyle = :dash)

    display(f)
end

begin # * Figure
    f = TwoPanel(; size = (720, 250))
    # gs = subdivide(f, 1, 3)
    begin # * Firing rate phase diagram
        fixed = (Dim{:η}(At(η)), Dim{:γ}(At(γ)))
        x = rates[fixed...]
        x = median(x, dims = Obs)
        x = dropdims(x, dims = Obs)
        x = permutedims(x, (:α, :β))
        x[x .== 0.0] .= NaN

        ax = Axis(f[1, 1]; title = "Firing rate (Hz)", xlabel = "α", ylabel = "β",
                  backgroundcolor = :gray88)
        p = contourf!(ax, x; colormap = seethrough(:turbo), nan_color = :lightgray,
                      levels = 11)
        Colorbar(f[1, 2], p;)
        Label(f[1, 2, Top()], L"\nu"; valign = :bottom, halign = :center)
    end

    function namefixed(dims)
        dims = filter(x -> length(lookup(x)) == 1, dims)
        return Dict(name(d) => only(lookup(d)) for d in dims)
    end
    begin # * Plot fanos
        # fixed = (Dim{:η}(Near(0.01)), Dim{:γ}(Near(0.01)), Dim{:β}(Near(0.6)))
        # fixed = (Dim{:η}(Near(0.01)), Dim{:α}(Near(1.4)), Dim{:β}(Near(0.7)))
        ax = Axis(f[1, 3]; xscale = log10, yscale = log10, title = "Fano factor",
                  xlabel = "Time bin (s)",
                  #   ylabel = "Fano factor",
                  limits = ((1e-3, 1e0), nothing), xticks = LogTicks(WilkinsonTicks(4)))

        a = fanos[Dim{:α}(Near(1.5)), Dim{:β}(Near(β)), Dim{:η}(At(η)), Dim{:γ}(At(0.0))]
        a = Dropdims(mean)(a, dims = Obs)
        lines!(ax, a; label = "γ = 0")

        a = fanos[Dim{:α}(Near(1.5)), Dim{:β}(Near(β)), Dim{:η}(At(η)), Dim{:γ}(At(γ))]
        a = Dropdims(mean)(a, dims = Obs)
        lines!(ax, a; label = "γ = $γ")

        a = fanos[Dim{:α}(Near(1.5)), Dim{:β}(Near(1.0)), Dim{:η}(At(η)), Dim{:γ}(At(0.0))]
        a = Dropdims(mean)(a, dims = Obs)
        lines!(ax, a; label = "β=1, γ = 0")

        # # * Then add momentum
        # a = fanos[Dim{:α}(At(2.0)), Dim{:β}(At(1.0)), Dim{:η}(At(0.02)), Dim{:γ}(At(γ))]
        # a = Dropdims(mean)(a, dims = Obs)
        # lines!(ax, a; label = "BM, momentum")

        # # * Then add fbm
        # a = fanos[Dim{:α}(At(2.0)), Dim{:β}(Near(0.8)), Dim{:η}(At(0.02)), Dim{:γ}(At(γ))]
        # a = Dropdims(mean)(a, dims = Obs)
        # lines!(ax, a; label = "fBM, momentum")

        # # * Then add fbm
        # a = fanos[Dim{:α}(Near(1.5)), Dim{:β}(Near(0.8)), Dim{:η}(At(0.02)), Dim{:γ}(At(γ))]
        # a = Dropdims(mean)(a, dims = Obs)
        # lines!(ax, a; label = "bFNS, momentum")

        # x = fanos[fixed...]

        # x = mean(x, dims = Obs)
        # x = dropdims(x, dims = Obs)
        # p = traces!(ax, x)
        # Colorbar(gs[2][1, 2], p; label = "$(name(dims(x, 2)))")
        axislegend(ax; position = :lt, patchsize = (10, 10))
    end

    begin # * Plot fano fit variation over α and β
        colorrange = (0.0, 0.6)
        mcs = permutedims(mcs, (:α, :β))
        levels = range(colorrange...; length = 11)
        ax = Axis(f[1, 4]; title = "Variability exponent",
                  xlabel = "α", ylabel = "β", backgroundcolor = :gray88,
                  limits = ((1.2, 2.0), (0.2, 1.0)))
        colormap = darksunset
        p = contourf!(ax, mcs; colormap, nan_color = :lightgray, levels)
        Colorbar(f[1, 5], p; ticks = WilkinsonTicks(4; k_max = 4, k_min = 3))
        Label(f[1, 5, Top()], L"c"; valign = :bottom, halign = :center)
    end
    addlabels!(f, ["f)", "g)", "h)"])
    display(f)

    wsave(plotdir("mean_field_sweep", "mean_field_sweep.pdf"), f)
end

# begin
#     f = Figure()
#     # fixed = (Dim{:η}(Near(0.01)), Dim{:γ}(Near(0.01)), Dim{:β}(Near(0.6)))
#     fixed = (Dim{:η}(At(0.02)), Dim{:α}(Near(1.5)), Dim{:γ}(Near(0.01)))
#     ax = Axis(f[1, 1]; xscale = log10, yscale = log10, title = "$fixed",
#               xlabel = "Time bin (ms)", ylabel = "Fano factor")
#     x = fanos[fixed...]
#     x = mean(x, dims = Obs)
#     x = dropdims(x, dims = Obs)
#     p = traces!(ax, x)
#     Colorbar(f[1, 2], p; label = "$(name(dims(x, 2)))")
#     display(f)
# end
