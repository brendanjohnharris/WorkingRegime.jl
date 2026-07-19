#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
WRTheory.@preamble()
set_theme!(foresight(:physics))

begin # * Load data
    file = datadir("bFOLE_sweep.jld2")
    data = load(file)
end

begin # * Plot heatmap of spectral exponents
    η = 0.1
    spectral_exponents = data["spectral_exponent"][η = Near(η)]
    ms = mean(spectral_exponents, dims = Obs)
    ms = dropdims(ms, dims = Obs)
    f = Figure()
    ax = Axis(f[1, 1];
              xlabel = "α", ylabel = "β", title = "Spectral exponent")
    p = contourf!(ax, ms)
    Colorbar(f[1, 2], p)
    f
end
begin # * Plot heatmap of diffusion exponents
    η = 0.1
    diffusion_exponents = data["diffusion_exponent"][η = Near(η)]
    ms = mean(diffusion_exponents, dims = Obs)
    ms = dropdims(ms, dims = Obs)
    f = Figure()
    ax = Axis(f[1, 1];
              xlabel = "α", ylabel = "β", title = "Diffusion exponent")
    # p = heatmap!(ax, ms)
    p = contourf!(ax, ms)
    contour!(ax, ms; color = :black, levels = [-0.5], linestyle = :dash)
    Colorbar(f[1, 2], p)
    f
end
# begin # * Plot heatmap of sampling accuracy
#     η = 0.4
#     accuracy = data["accuracy"][η = Near(η)]
#     accuracy = map(accuracy) do a
#         if a === NaN
#             a
#         else
#             a[end] |> mean
#         end
#     end
#     ms = mean(accuracy, dims = Obs)
#     ms = dropdims(ms, dims = Obs)
#     f = Figure()
#     ax = Axis(f[1, 1])
#     # p = lines!(ax, ms[β = Near(1.0)])
#     p = heatmap!(ax, ms)
#     Colorbar(f[1, 2], p)
#     f
# end

# begin # * Sampling accuracy as a function of tau
#     η = 0.4
#     accuracy = data["accuracy"][η = Near(η)]
#     accuracy = map(accuracy) do a
#         if a === NaN
#             a
#         else
#             a .|> mean
#         end
#     end
#     accuracy = accuracy[Obs=Near(1)]
#     lines(accuracy[α = Near(1.5), β = Near(1.0)])
# end
