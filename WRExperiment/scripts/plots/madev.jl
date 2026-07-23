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
import AllenNeuropixelsBase: Depth
CairoMakie.set_theme!(Foresight.foresight(:physics))

stimuli = [r"Natural_Images", "spontaneous", "flash_250ms"]
xtickformat = terseticks
theta = Interval(THETA()...)
gamma = Interval(GAMMA()...)
alpha = 0.8
bandalpha = 0.2
mkpath(plotdir("madev"))

plot_data, data_file = produce_or_load(
    Dict(), calcdir("plots");
    filename = savepath("wr_madev")
) do _
    session_table = load(calcdir("plots", "posthoc_session_table.jld2"), "session_table")
    oursessions = session_table.ecephys_session_id
    path = calcdir("madev")
    QQ = calcquality(path)
    plot_data = map(stimuli) do stimulus
        Q = QQ[
            stimulus = At(stimulus),
            Structure = At(structures),
            SessionID(At(oursessions)),
        ]
        @assert mean(Q) > 0.9

        begin # * Load data
            M = map(lookup(Q, Structure)) do structure
                out = map(lookup(Q, SessionID)) do sessionid
                    if Q[SessionID = At(sessionid), Structure = At(structure)] == 0
                        return nothing
                    end
                    filename = savepath(
                        (@strdict sessionid structure stimulus), "jld2",
                        path
                    )
                    mad = load(filename, "mad")
                    coeffs = load(filename, "coeffs")
                    S = load(filename, "S")
                    return mad, coeffs, S
                end
                out = filter(!isnothing, out)
                out = filter(out) do x # Remove sessions that don't have data to a reasonable depth
                    maximum(DimensionalData.metadata(x[1])[:streamlinedepths]) > 0.9
                end

                S = getindex.(out, 3)
                coeffs = getindex.(out, 2)
                out = getindex.(out, 1)

                m = DimensionalData.metadata.(out)
                sessions = getindex.(m, :sessionid)

                streamlinedepths = getindex.(m, :streamlinedepths)
                layerinfo = getindex.(m, :layerinfo)

                unidepths = commondepths(streamlinedepths)
                out = map(out, streamlinedepths, layerinfo) do o, s, l
                    o = set(o, Depth => s)
                    layernames = ToolsArray(l[1], (Depth(lookup(o, Depth)),))
                    layernums = ToolsArray(l[3], (Depth(lookup(o, Depth)),))
                    o = o[Depth(Near(unidepths))]
                    layernames = layernames[Depth(Near(unidepths))]
                    layernums = layernums[Depth(Near(unidepths))]
                    # @assert length(unique(lookup(o, Depth))) == length(unidepths)
                    @assert issorted(lookup(o, Depth))
                    push!(o.metadata, :layernames => layernames)
                    push!(o.metadata, :layernums => layernums)
                    o = set(o, Depth => unidepths)
                end
                coeffs = map(coeffs, streamlinedepths, layerinfo) do o, s, l
                    o = set(o, Depth => s)
                    layernames = ToolsArray(l[1], (Depth(lookup(o, Depth)),))
                    layernums = ToolsArray(l[3], (Depth(lookup(o, Depth)),))
                    o = o[Depth(Near(unidepths))]
                    layernames = layernames[Depth(Near(unidepths))]
                    layernums = layernums[Depth(Near(unidepths))]
                    # @assert length(unique(lookup(o, Depth))) == length(unidepths)
                    @assert issorted(lookup(o, Depth))
                    # push!(o.metadata, :layernames => layernames)
                    # push!(o.metadata, :layernums => layernums)
                    o = set(o, Depth => unidepths)
                end
                S = map(S, streamlinedepths, layerinfo) do o, s, l
                    o = set(o, Depth => s)
                    layernames = ToolsArray(l[1], (Depth(lookup(o, Depth)),))
                    layernums = ToolsArray(l[3], (Depth(lookup(o, Depth)),))
                    o = o[Depth(Near(unidepths))]
                    layernames = layernames[Depth(Near(unidepths))]
                    layernums = layernums[Depth(Near(unidepths))]
                    @assert issorted(lookup(o, Depth))
                    o = set(o, Depth => unidepths)
                end
                layernames = ToolsArray(
                    stack(
                        getindex.(
                            DimensionalData.metadata.(out),
                            :layernames
                        )
                    ),
                    (Dim{:depths}(unidepths), SessionID(sessions))
                )
                layernums = ToolsArray(
                    stack(
                        getindex.(
                            DimensionalData.metadata.(out),
                            :layernums
                        )
                    ),
                    (Dim{:depths}(unidepths), SessionID(sessions))
                )
                mad = stack(SessionID(sessions), out, dims = 3) .|> Float32
                coeffs = stack(SessionID(sessions), coeffs, dims = 2) .|> Float32
                S = stack(SessionID(sessions), S, dims = 3) .|> Float32
                layernums = parselayernum.(layernames)
                return mad, coeffs, S, layernames, layernums
            end
            M, coeffs, S, layernames, layernums = zip(M...)
        end
        begin # * Format layers
            meanlayers = map(layernums) do l
                round.(Int, mean(l, dims = 2))
            end
            M̄ = map(M, meanlayers) do m, l
                # s = set(s, Depth => Dim{:layer}(layernum2name.(parent(l)[:])))
                m = set(m, Depth => Dim{:layer}(parent(l)[:]))
                m = set(m, :layer => DimensionalData.Unordered)
            end
            M̄ = map(M̄) do s
                ss = map(unique(lookup(s, :layer))) do l
                    ls = s[Dim{:layer}(At(l))]
                    if hasdim(ls, :layer)
                        ls = mean(ls, dims = :layer)
                    end
                    ls
                end
                cat(ss..., dims = :layer)
            end
            M̄ = ToolsArray(M̄ |> collect, (Structure(lookup(Q, Structure)),))
            M = ToolsArray(M |> collect, (Structure(lookup(Q, Structure)),))

            S̄ = map(S, meanlayers) do s, l
                s = set(s, Depth => Dim{:layer}(parent(l)[:]))
                s = set(s, :layer => DimensionalData.Unordered)
            end
            S̄ = map(S̄) do s
                ss = map(unique(lookup(s, :layer))) do l
                    ls = s[Dim{:layer}(At(l))]
                    if hasdim(ls, :layer)
                        ls = mean(ls, dims = :layer)
                    end
                    ls
                end
                cat(ss..., dims = :layer)
            end
            S̄ = ToolsArray(S̄ |> collect, (Structure(lookup(Q, Structure)),))
            S = ToolsArray(S |> collect, (Structure(lookup(Q, Structure)),))
            # S̄ = map(S̄) do s
            #     N = UnitEnergy(s, dims = 1)
            #     N(s) .|> Float32
            # end # ! No normalization
            layerints = load(DrWatson.datadir("grand_unified_layers.jld2"), "layerints")
        end

        coeffs = ToolsArray(collect(coeffs), (Structure(structures),))
        begin # * Format layers
            coeffs_median = map(coeffs, meanlayers) do m, l
                # s = set(s, Depth => Dim{:layer}(layernum2name.(parent(l)[:])))
                m = set(m, Depth => Dim{:layer}(parent(l)[:]))
                m = set(m, :layer => DimensionalData.Unordered)
            end
            coeffs_median = map(coeffs_median) do s
                ss = map(unique(lookup(s, :layer))) do l
                    ls = s[Dim{:layer}(At(l))]
                    if hasdim(ls, :layer)
                        ls = median(ls, dims = :layer)
                    end
                    ls
                end
                cat(ss..., dims = :layer)
            end
            coeffs_median = ToolsArray(
                coeffs_median |> collect,
                (Structure(lookup(Q, Structure)),)
            )
        end

        plot_data = @strdict M M̄ S S̄ coeffs coeffs_median layernames layernums layerints meanlayers oursessions Q
        return plot_data
    end

    return Dict(string.(stimuli) .=> plot_data)
end

for stimulus in stimuli
    @info "Plotting madev for $stimulus"
    @unpack M, coeffs, coeffs_median, layernames, layernums, layerints, meanlayers, M̄, oursessions, Q = plot_data[string(stimulus)]
    begin
        filebase = stimulus == "spontaneous" ? "" : "_$(val_to_string(stimulus))"
        statsfile = plotdir("madev", "madev$filebase.txt")
        close(open(statsfile, "w")) # Create the file or clear it
        f = TwoPanel()

        begin # * Load the channel-wise fits
            χ = plot_data[string(stimulus)]["coeffs"]

            # * Ensure structures and sessions match
            coeffs = getindex.(coeffs, [SessionID(At(oursessions))])
            coeffs = coeffs[Structure = At(structures)]
        end

        begin # * Plot the intercept
            ax = Axis(
                f[1, 1]; ylabel = "Cortical depth (%)",
                xlabel = "Median diffusion exponent",
                limits = ((0.25, 0.75), (0, 1)), ytickformat = depthticks,
                title = "Diffusion exponent", yreversed = true
            )
            vlines!(ax, 0.5; color = :gray, linewidth = 3, linestyle = :dash)

            for (i, _b) in coeffs |> enumerate |> collect |> reverse
                μ, (σl, σh) = bootstrapmedian(
                    _b .+ eps() .* randn(size(_b)),
                    dims = SessionID
                )
                μ, σl, σh = upsample.((μ, σl, σh), 5)

                band!(
                    ax, Point2f.(collect(σl), lookup(μ, 1)),
                    Point2f.(collect(σh), lookup(μ, 1));
                    color = (structurecolors[i], 0.32), label = structures[i]
                )
                lines!(
                    ax, collect(μ), lookup(μ, 1); color = (structurecolors[i], alpha),
                    label = structures[i]
                )
            end
            l = axislegend(
                ax, position = :lb, nbanks = 2, labelsize = 12, merge = true,
                margin = (30, 0, 10, 0)
            )
            reverselegend!(l)
            plotlayerints!(ax, layerints; axis = :y, newticks = false)
        end

        begin # * Is the exponent correlated to depth?
            tps = [mediankendallpvalue(lookup(x, Depth), x) for x in χ]
            τs = cat(first.(tps), last.(tps); dims = Var([:kendall, :pvalue]))
            mtau = median(τs[2:end, 1]) # Median excluding VISp
            open(statsfile, "a+") do file
                write(file, "\n## Diffusion exponent\n")
                show(file, DataFrame(DimTable(τs; layersfrom = Var)))
                write(file, "\nMedian τ (no VISp) = $mtau\n")
            end
        end

        begin # * Plot the hierarchical correlation across layers
            N = 10000
            method = :group
            unidepths = commondepths(lookup.(χ, [Depth]))
            x = getindex.([hierarchy_scores], structures)

            unichi = getindex.(coeffs, [Depth(Near(unidepths))])
            unichi = set.(unichi, [Depth => unidepths])
            y = stack(Structure(structures), unichi)

            μ, σ, 𝑝 = hierarchicalkendall(x, y, method; N)
        end
        begin
            # μ[𝑝 .> PTHR] .= NaN
            # μt[𝑝t .> PTHR] .= NaN
            # μg[𝑝g .> PTHR] .= NaN

            markersize = 10

            ax = Axis(
                f[1, 2]; ylabel = "Cortical depth (%)",
                xlabel = "Kendall's 𝜏",
                ytickformat = depthticks,
                xtickformat,
                title = "Diffusion exponent gradient",
                limits = ((-0.6, 0.4), (0, 1)),
                yreversed = true,
                yticklabelsvisible = false
            )

            vlines!(ax, 0; color = :gray, linewidth = 3, linestyle = :dash)

            band!(
                ax, Point2f.(collect(first.(σ)), unidepths),
                Point2f.(collect(last.(σ)), unidepths);
                color = (:cornflowerblue, bandalpha),
                label = "Diffusion exponent"
            )
            # lines!(ax, unidepths, collect(μ); alpha = bandalpha,
            #        label = "1/𝑓 exponent", color = crimson)
            scatter!(
                ax, collect(μ[𝑝 .< PTHR]), unidepths[𝑝 .< PTHR];
                label = "Diffusion exponent", color = :cornflowerblue, markersize
            )
            scatter!(
                ax, collect(μ[𝑝 .≥ PTHR]), unidepths[𝑝 .≥ PTHR]; color = :transparent,
                strokecolor = :cornflowerblue,
                strokewidth = 1, markersize
            )

            # axislegend(ax, position = :lb, merge = true, labelsize = 12, nbanks = 1)

            plotlayerints!(ax, layerints; axis = :y, newticks = false)
        end
        # addlabels!(f, ["a", "d", "f", "e", "c", "b", "g", "h"])
        f |> display
    end

    begin # * VISp plot
        begin # * Plot mean MAD
            sf = OnePanel()
            m = M̄[Structure = At("VISp")][layer = At(2)]
            if hasdim(m, :layer)
                m = median(m, dims = :layer)
                m = dropdims(m, dims = :layer)
            end
            ax = Axis(
                sf[1, 1]; ylabel = "MAD",
                xlabel = "Time lag (s)",
                title = "Mean absolute deviation in VISp L2/3", xscale = log10,
                yscale = log10,
            )

            # * Fit exponent
            _m = m[𝑡 = 1.0e-3 .. 1.0e-2]
            t = log10.(times(_m))
            s = log10.(parent(_m))
            coeff = hcat(ones(length(t)), t) \ s
            intercept, slope = eachrow(coeff)
            meanintercept = median(intercept)
            # meanslope = median(slope)
            meanslope, (sl, su) = bootstrapmedian(slope) # median(filter(!isnan, slope))
            @info "mad median: $meanslope, CI: ($sl, $su)"
            # slopeerror = std(slope) / 2 #quantile.([slope], [0.25, 0.75])

            text!(
                ax, [1.0e-3], [10^(-4.25)];
                text = "a = $(round(meanslope, sigdigits = 2))",
                align = (:left, :top), fontsize = 20
            )

            mu = median(m, dims = SessionID)
            mu = dropdims(mu, dims = SessionID)
            σl = mapslices(x -> quantile(x, 0.25), m; dims = SessionID)
            σh = mapslices(x -> quantile(x, 0.75), m; dims = SessionID)
            σl = dropdims(σl, dims = SessionID)
            σh = dropdims(σh, dims = SessionID)
            band!(ax, lookup(mu, 𝑡), σl, σh, alpha = 0.5)

            lines!(ax, mu)

            lines!(
                ax, lookup(_m, 𝑡),
                exp10.(meanintercept .+ meanslope .* log10.(times(_m))),
                linestyle = :dash, color = crimson
            )
        end

        if true # * Inset variation across areas
            ssf = OnePanel()
            ax = Axis(
                ssf[1, 1];
                yticks = (1:4, ["L2/3", "L4", "L5", "L6"]),
                #   width = Relative(0.5),
                #   height = Relative(0.6),
                #   halign = 1,
                #   valign = 0.0,
                limits = ((0.35, 0.65), nothing),
                yreversed = true,
                #   xaxisposition = :top
                ylabel = "Cortical layer",
                xlabel = "Diffusion exponent",
                ygridvisible = false
            )
            layers_to_plot = [2, 3, 4, 5]
            structure_offsets = [-0.375, -0.225, -0.075, 0.075, 0.225, 0.375] .* 0.9  # Offsets for each structure within a layer

            vlines!(ax, 0.5; color = :gray, linewidth = 3, linestyle = :dash)
            hlines!(ax, layers_to_plot[1:(end - 1)] .- 0.5; color = :gray, linewidth = 1)

            # Extract data for all layers
            _a = stack(coeffs_median)

            # Perform statistical tests for each structure-layer combination
            𝑝 = map(eachslice(_a, dims = (Structure, :layer))) do b
                𝑝 = HypothesisTests.SignedRankTest(collect(b) .- 0.5)
                𝑝 = HypothesisTests.pvalue(𝑝, tail = :right)
            end
            𝑝[:] .= MultipleTesting.adjust(collect(𝑝)[:], BenjaminiHochberg())

            # Plot boxes for each layer and structure
            for (layer_idx, layer) in enumerate(layers_to_plot)
                for (struct_idx, structure) in enumerate(lookup(_a, Structure))
                    x = _a[Structure = At(structure), layer = At(layer)]
                    p = 𝑝[Structure = At(structure), layer = At(layer)]

                    x_pos = layer_idx + structure_offsets[struct_idx]

                    if p < PTHR
                        boxplot!(
                            ax, fill(x_pos, size(_a, SessionID)),
                            collect(x)[:]; show_outliers = false,
                            orientation = :horizontal,
                            color = structurecolors[struct_idx],
                            strokecolor = structurecolors[struct_idx],
                            strokewidth = 1, width = 0.1, whiskerlinewidth = 0,
                            medianlinewidth = 2, label = structure
                        )
                    else
                        boxplot!(
                            ax, fill(x_pos, size(_a, SessionID)),
                            collect(x)[:]; show_outliers = false,
                            color = :transparent, orientation = :horizontal,
                            strokecolor = structurecolors[struct_idx],
                            strokewidth = 1, width = 0.1, whiskerlinewidth = 0,
                            medianlinewidth = 2, label = structure
                        )
                    end
                end
            end
            axislegend(ax, position = :lb, merge = true)
        end

        # # !!! Layers
        # begin
        #     ax = Axis(sf[1, 2]; ylabel = "Cortical layer",
        #               xlabel = "Median diffusion exponent",
        #               title = "Diffusion exponent in VISp", yreversed = true,
        #               yticks = (2:5, ["L2/3", "L4", "L5", "L6"]))

        #     vlines!(ax, 0.5; color = :gray, linewidth = 3, linestyle = :dash)

        #     _b = coeffs_median[Structure = At("VISp")]
        # end
        # begin # * Do statistical test against value of 0.5
        #     𝑝 = map(eachslice(_b, dims = :layer)) do b
        #         𝑝 = HypothesisTests.SignedRankTest(collect(b) .- 0.5)
        #         𝑝 = HypothesisTests.pvalue(𝑝, tail = :right)
        #     end
        #     𝑝[:] .= MultipleTesting.adjust(collect(𝑝)[:], BenjaminiHochberg())
        # end
        # for l in lookup(_b, :layer)
        #     if l > 1
        #         x = _b[layer = At(l)]
        #         p = 𝑝[layer = At(l)]
        #         if p < SpatiotemporalMotifs.PTHR
        #             boxplot!(ax, fill(l, size(_b, SessionID)),
        #                      collect(x)[:];
        #                      orientation = :horizontal, show_outliers = false,
        #                      color = Foresight.colororder[l - 1],
        #                      strokecolor = Foresight.colororder[l - 1], strokewidth = 3)
        #         else
        #             boxplot!(ax, fill(l, size(_b, SessionID)),
        #                      collect(x)[:];
        #                      orientation = :horizontal, show_outliers = false,
        #                      color = :transparent,
        #                      strokecolor = Foresight.colororder[l - 1], strokewidth = 3)
        #         end
        #     end
        # end

        display(sf)
        wsave(plotdir("madev", "reduced_madev$filebase.pdf"), sf)
        display(ssf)
        wsave(plotdir("madev", "supplemental_madev_$filebase.pdf"), ssf)
    end

    wsave(plotdir("madev", "madev$filebase.pdf"), f)
end
