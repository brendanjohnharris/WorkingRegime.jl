#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRExperiment"

using WRExperiment
import TimeseriesTools: freqs
using Peaks
using FileIO
using CairoMakie
using IntervalSets
using Foresight
using DimensionalData
using TimeseriesTools
using StatsBase
using Statistics
using DataFrames
using HypothesisTests
using MultipleTesting
using Distributed
using Random
using Unitful
import AllenNeuropixelsBase as AN
using AllenNeuropixels.AllenNeuropixelsBase
CairoMakie.set_theme!(Foresight.foresight(:physics))

stimuli = ["r\"Natural_Images\"", "spontaneous", "flash_250ms"]
xtickformat = terseticks
theta = Interval(THETA()...)
gamma = Interval(GAMMA()...)
alpha = 0.8
bandalpha = 0.2
mkpath(plotdir("fig2_reduced"))

data_file = savepath("mad_psd.jld2")(Dict())
plot_data = load(datadir("plots", data_file))

begin
    structure = "VISp"
    layer = 2
end

for stimulus in stimuli
    @info "Plotting spectra for $stimulus"
    @unpack S, layernames, layernums, layerints, meanlayers, S̄, oursessions, Q, mapple = plot_data[string(stimulus)]

    _b = map(2:5) do layer
        _b = map(structures) do structure
            a = mapple["χ"][Structure = At(structure)]
            S = plot_data[stimulus]["S"][Structure = At(structure)]
            struct_idx = findfirst(structures .== structure)
            ln = layernums[struct_idx]
            depths = lookup(ln, 1)
            mask = vec(any(parent(ln) .== layer, dims = 2))
            layermap = depths[mask]
            a = a[Depth = Near(layermap)]
            a = median(a, dims = Depth)
            a = .-dropdims(a, dims = Depth)
        end
        _b = ToolsArray(_b, (Structure(structures),)) |> stack
    end
    _b = ToolsArray(_b, (Dim{:layer}(2:5),)) |> stack

    begin
        filebase = stimulus == "spontaneous" ? "" : "_$(val_to_string(stimulus))"

        begin # * Median power spectrum. Bands show 1 S.D.
            f = OnePanel()
            axargs = (; xscale = log10, yscale = log10,
                      limits = ((3, 500), (exp10(-15), exp10(-8.5))),
                      yticks = (exp10.([-14, -12, -10]),
                                [
                                    rich("10", superscript("-14")),
                                    rich("10", superscript("-12")),
                                    rich("10", superscript("-10"))
                                ]),
                      xlabel = "Frequency (Hz)",
                      xgridvisible = true,
                      ygridvisible = true,
                      xtickformat,
                      xticks = [3, 10, 30, 100],
                      ylabel = "PSD",
                      title = "Power spectral density in $structure, L2/3")
            ax2 = Axis(f[1, 1]; axargs...) # For band annotations
            hideyaxis!(ax2)
            hidexaxis!(ax2)
            hidespines!(ax2)
            hidedecorations!(ax2)
            ax = Axis(f[1, 1]; axargs...)

            # # * Band annotations
            # vspan!(ax2, extrema(theta)..., color = (crimson, 0.22),
            #        label = "𝛉")
            # vlines!(ax2, [7.0], color = (crimson, 0.42), linestyle = :dash, linewidth = 4)
            # vspan!(ax2, extrema(gamma)..., color = (cornflowerblue, 0.22),
            #        label = "𝛄")
            # vlines!(ax2, [54.4], color = (cornflowerblue, 0.42), linestyle = :dash,
            #         linewidth = 4)

            s = S̄[Structure = At(structure)][layer = At([layer])][𝑓 = 3u"Hz" .. 500u"Hz"]
            # s = s ./ 10^((i - 1.5) / 2.5)
            yc = only(mean(s[𝑓 = Near(3u"Hz")]))
            ps = WRExperiment.plotspectrum!(ax, s; textposition = (3, yc),
                                            color = structurecolormap[structure],
                                            annotations = [:peaks],
                                            label = structure,
                                            mappleargs = (; color = crimson),
                                            domedian = true)

            mb = _b[Structure = At(structure), layer = At(layer)]
            mb, (sl, su) = bootstrapmedian(mb |> collect) # median(filter(!isnan, mb))
            @info "$stimulus spectral median: $mb, CI: ($sl, $su)"

            text!(ax, [100], [1e-10]; text = "b = $(round(mb; sigdigits=3))",
                  fontsize = 20)

            # ps, α = first.(psa), last.(psa)
            # α = round.(α, sigdigits = 3)
            # axislegend(ax2, position = :lb, labelsize = 12, backgroundcolor = :white,
            #            framevisible = true, padding = (5, 5, 5, 5))

            # axislegend(ax, position = :rt, labelsize = 12, merge = true,
            #            backgroundcolor = :white,
            #            framevisible = true, padding = (5, 5, 5, 5),
            #            nbanks = 3, patchsize = (15, 15), rowgap = 2)
            f
        end

        # begin # * Plot
        #     begin
        #         ax = Axis(f[1, 2]; ylabel = "Cortical layer",
        #                   xlabel = "Mean spectral exponent",
        #                   title = "Spectral exponent in VISp", yreversed = true,
        #                   yticks = (2:5, ["L2/3", "L4", "L5", "L6"]))

        #         vlines!(ax, 2.0; color = :gray, linewidth = 3, linestyle = :dash)
        #     end

        #     begin # * Do statistical test against value of 2.0
        #         𝑝 = map(eachslice(_b, dims = :layer)) do b
        #             𝑝 = HypothesisTests.SignedRankTest(collect(b) .- 2.0)
        #             𝑝 = HypothesisTests.pvalue(𝑝, tail = :left)
        #         end
        #         𝑝[:] .= MultipleTesting.adjust(collect(𝑝)[:], BenjaminiHochberg())
        #     end
        #     for l in lookup(_b, :layer)
        #         if l > 1
        #             x = _b[layer = At(l)]
        #             p = 𝑝[layer = At(l)]
        #             x = filter(!isnan, x)
        #             if p < PTHR
        #                 boxplot!(ax, fill(l, length(x)),
        #                          collect(x)[:];
        #                          orientation = :horizontal, show_outliers = false,
        #                          color = Foresight.colororder[l - 1],
        #                          strokecolor = Foresight.colororder[l - 1], strokewidth = 3)
        #             else
        #                 boxplot!(ax, fill(l, length(x)),
        #                          collect(x)[:];
        #                          orientation = :horizontal, show_outliers = false,
        #                          color = :transparent,
        #                          strokecolor = Foresight.colororder[l - 1], strokewidth = 3)
        #             end
        #         end
        #     end
        # end
        display(f)

        if true # * Inset variation across areas
            ssf = OnePanel()
            ax = Axis(ssf[1, 1];
                      yticks = (1:4, ["L2/3", "L4", "L5", "L6"]),
                      #   width = Relative(0.5),
                      #   height = Relative(0.6),
                      #   halign = 1,
                      #   valign = 0.0,
                      limits = ((-2.05, -1.2), nothing),
                      yreversed = true,
                      #   xaxisposition = :top
                      ylabel = "Cortical layer",
                      xlabel = "Spectral exponent",
                      ygridvisible = false)
            layers_to_plot = [2, 3, 4, 5]
            structure_offsets = [-0.375, -0.225, -0.075, 0.075, 0.225, 0.375] .* 0.9  # Offsets for each structure within a layer

            vlines!(ax, -2.0; color = :gray, linewidth = 3, linestyle = :dash)
            hlines!(ax, layers_to_plot[1:(end - 1)] .- 0.5; color = :gray, linewidth = 1)

            # Perform statistical tests for each structure-layer combination
            𝑝 = map(eachslice(_b, dims = (Structure, :layer))) do b
                𝑝 = HypothesisTests.SignedRankTest(collect(b) .+ 2.0)
                𝑝 = HypothesisTests.pvalue(𝑝, tail = :right)
            end
            𝑝[:] .= MultipleTesting.adjust(collect(𝑝)[:], BenjaminiHochberg())

            # Plot boxes for each layer and structure
            for (layer_idx, layer) in enumerate(layers_to_plot)
                for (struct_idx, structure) in enumerate(lookup(_b, Structure))
                    x = _b[Structure = At(structure), layer = At(layer)]
                    x = filter(!isnan, x)
                    p = 𝑝[Structure = At(structure), layer = At(layer)]

                    x_pos = layer_idx + structure_offsets[struct_idx]

                    if p < PTHR
                        boxplot!(ax, fill(x_pos, length(x)),
                                 collect(x)[:]; show_outliers = false,
                                 orientation = :horizontal,
                                 color = structurecolors[struct_idx],
                                 strokecolor = structurecolors[struct_idx],
                                 strokewidth = 1, width = 0.1, whiskerlinewidth = 0,
                                 medianlinewidth = 2, label = structure)
                    else
                        boxplot!(ax, fill(x_pos, length(x)),
                                 collect(x)[:]; show_outliers = false,
                                 color = :transparent, orientation = :horizontal,
                                 strokecolor = structurecolors[struct_idx],
                                 strokewidth = 1, width = 0.1, whiskerlinewidth = 0,
                                 medianlinewidth = 2, label = structure)
                    end
                end
            end
            axislegend(ax, position = :lt, merge = true)
        end
    end
    wsave(plotdir("fig2_reduced", "power_spectra$filebase.pdf"), f)
    wsave(plotdir("fig2_reduced", "supplemental_power_spectra$filebase.pdf"), ssf)
end
