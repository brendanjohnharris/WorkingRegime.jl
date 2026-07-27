#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRExperiment"

using WRExperiment
using CairoMakie
using DimensionalData
using JLD2
using FileIO
using Foresight
CairoMakie.set_theme!(Foresight.foresight(:physics))

# Replots the hierarchy gradient of both exponents straight from the stored bootstrap, so nothing is
# refit. Filled markers are significant at PTHR, open markers are not.
gradients = (
    "Diffusion exponent" => (load(datadir("WRExperiment.jld2"), "diffusion_hierarchical"), :cornflowerblue),
    "Spectral exponent" => (load(datadir("WRExperiment.jld2"), "spectral_hierarchical"), :crimson),
)
mkpath(plotdir("madev"))

for stim in keys(first(gradients)[2][1])
    f = OnePanel()
    ax = Axis(
        f[1, 1]; xlabel = "Kendall's 𝜏", ylabel = "Cortical depth (%)",
        ytickformat = xs -> string.(round.(Int, 100 .* xs)),
        title = "Hierarchy gradient ($stim)", yreversed = true
    )
    vlines!(ax, 0; color = :gray, linewidth = 3, linestyle = :dash)

    for (label, (dict, color)) in gradients
        d = dict[stim]
        sig = collect(d.𝑝) .< PTHR
        band!(
            ax, Point2f.(collect(first.(d.σ)), d.unidepths),
            Point2f.(collect(last.(d.σ)), d.unidepths); color = (color, 0.2), label
        )
        scatter!(ax, collect(d.μ)[sig], d.unidepths[sig]; color, markersize = 10, label)
        scatter!(
            ax, collect(d.μ)[.!sig], d.unidepths[.!sig]; color = :transparent,
            strokecolor = color, strokewidth = 1, markersize = 10, label
        )
        @info "$stim, $label: $(count(sig))/$(length(sig)) depths significant at p < $PTHR"
    end

    axislegend(ax, position = :lb, merge = true, labelsize = 12)
    wsave(plotdir("madev", "hierarchy_tau_$(val_to_string(stim)).pdf"), f)
end
