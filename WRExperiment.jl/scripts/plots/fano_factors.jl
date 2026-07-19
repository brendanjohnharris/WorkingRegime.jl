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
using Random
using Distributed
WRExperiment.@preamble
set_theme!(foresight(:physics))

stimuli = [r"Natural_Images", "spontaneous", "flash_250ms"]
mkpath(plotdir("fano_factor"))

plot_data, data_file = produce_or_load(Dict(), calcdir("plots");
                                       filename = savepath("fano_factor")) do _
    session_table = load(calcdir("plots", "posthoc_session_table.jld2"), "session_table")
    oursessions = session_table.ecephys_session_id
    path = calcdir("madev")
    QQ = calcquality(path)
    plot_data = map(stimuli) do stimulus
        Q = QQ[stimulus = At(stimulus),
               Structure = At(structures),
               SessionID(At(oursessions))]
        @assert mean(Q) > 0.9

        begin # * Load data
            unitdepths = map(lookup(Q, Structure)) do structure
                out = map(lookup(Q, SessionID)) do sessionid
                    if Q[SessionID = At(sessionid), Structure = At(structure)] == 0
                        return nothing
                    end
                    filename = savepath((@strdict sessionid structure stimulus), "jld2",
                                        path)
                    jldopen(filename, "r") do f
                        unitdepths = f["unitdepths"]
                        mad = f["mad"]
                        layermap = mad.metadata[:layerinfo]
                        layermap = ToolsArray(layermap[3], (Depth(layermap[2]),))

                        unitlayers = map(unitdepths.probedepth) do depth
                            layermap[Depth = Near(depth)]
                        end
                        unitdepths.layer = unitlayers
                        return unitdepths
                    end
                end
                out = filter(!isnothing, out)
                out = filter(!isempty, out)
                # out = filter(out) do x # Remove sessions that don't have data to a reasonable depth
                #     maximum(DimensionalData.metadata(x[1])[:streamlinedepths]) > 0.90
                # end
            end
        end

        plot_data = @strdict unitdepths
        return plot_data
    end

    return Dict(string.(stimuli) .=> plot_data)
end

begin
    fanorange = 10^(1.5) .. 1e3
    stimulus = "spontaneous"
    structure = "VISp"
end

begin # * Plot fano factor for VISp, layer 2/3, spontaneous
    unitdepths = plot_data[stimulus]["unitdepths"][findfirst(structures .== structure)]
    unitdepths = map(unitdepths) do units
        filter(units) do unit
            unit.layer == 2 # Choose layer 2/3
        end
    end
    unitdepths = filter(!isempty, unitdepths)
    oursessions = map(unitdepths) do u
        only(unique(u.ecephys_session_id))
    end
    # end
    # begin # * Plot mean fano factor for 1 subject
    f = OnePanel()
    ax = Axis(f[1, 1];
              xlabel = "Time lag (ms)",
              ylabel = "Fano factor",
              title = "Fano factor in VISp L2/3",
              xscale = log10, yscale = log10)
    fanos = map(unitdepths) do units
        fanos = units.fano_factor
        fanos = ToolsArray(fanos, (Unit(units.ecephys_unit_id),)) |> stack
        mean_fano = median(fanos, dims = Unit)
        mean_fano = dropdims(mean_fano, dims = Unit)
    end
    fanos = ToolsArray(fanos, (SessionID(oursessions),)) |> stack
    fanos = fanos[𝑡 = 1 .. 1000]
    mu = median(fanos, dims = SessionID)
    mu = dropdims(mu, dims = SessionID)
    # s = std(fanos, dims = SessionID)
    sl = mapslices(x -> quantile(x, 0.25), fanos; dims = SessionID)
    sl = dropdims(sl, dims = SessionID)
    su = mapslices(x -> quantile(x, 0.75), fanos; dims = SessionID)
    su = dropdims(su, dims = SessionID)

    mfanos = map(eachcol(fanos)) do mu
        fano_range = mu[fanorange] .|> log10
        t_range = times(fano_range) .|> log10
        xx = hcat(ones(length(t_range)), t_range)
        m = xx \ fano_range
    end
    mintercept = first.(mfanos)
    mslope = last.(mfanos)
    mintercept, (slintercept, suintercept) = bootstrapmedian(mintercept |>
                                                                                  collect) # median(filter(!isnan, mfanos))
    mslope, (slslope, suslope) = bootstrapmedian(mslope |> collect) # median(filter(!isnan, mfanos))
    @info "$stimulus fano median: $mslope, CI: ($slslope, $suslope)"

    # for x in fanos
    #     lines!(ax, x, color = cornflowerblue, alpha = 0.3, linewidth = 1)
    # end
    band!(ax, times(mu), sl, su, color = cornflowerblue, alpha = 0.3)

    lines!(ax, mu, color = cornflowerblue, linewidth = 3, label = "Mean Fano Factor")

    lines!(ax, exp10.(t_range), exp10.(mintercept .+ mslope .* t_range);
           color = :red, linestyle = :dash, linewidth = 3)

    text!(ax, [10^0.5], [10^0.4]; text = "c = $(round(mslope, digits=2))",
          align = (:left, :center), fontsize = 20)
    display(f)
end

begin # * Now do slope variation over layers boxplot.
    _c = map(Dim{:layer}(2:5)) do l# L2/3 to L6
        structureslopes = map(structures) do structure
            unitdepths = plot_data[stimulus]["unitdepths"][findfirst(structures .==
                                                                     structure)]
            _c = map(unitdepths) do units
                units = subset(units, :layer => ByRow(==(l)))
                if isempty(units)
                    return nothing
                else
                    slopes = map(units.fano_factor) do fano
                        fano_range = fano[fanorange] .|> log10
                        t_range = times(fano_range) .|> log10
                        X = hcat(ones(length(t_range)), t_range)
                        m = X \ fano_range

                        return last(m) # Fano slope
                    end

                    return ToolsArray(slopes, (Unit(units.ecephys_unit_id),)) |> stack
                end
            end
            idxs = findall(!isnothing, _c)
            _c = _c[idxs]
            oursessions = map(unitdepths) do ls
                only(unique(ls.ecephys_session_id))
            end
            oursessions = oursessions[idxs]
            _c = mean.(_c, dims = Unit)
            _c = dropdims.(_c, dims = Unit)
            _c = ToolsArray(_c, (SessionID(oursessions),)) |> stack
        end
        ToolsArray(structureslopes, (Structure(structures),))
    end
    _c = ToolsArray(_c, (Dim{:layer}(2:5),)) |> stack
end
if true # * Inset variation across areas
    ssf = OnePanel()
    ax = Axis(ssf[1, 1];
              yticks = (1:4, ["L2/3", "L4", "L5", "L6"]),
              #   width = Relative(0.5),
              #   height = Relative(0.6),
              #   halign = 1,
              #   valign = 0.0,
              limits = ((0.15, 0.4), nothing),
              yreversed = true,
              #   xaxisposition = :top
              ylabel = "Cortical layer",
              xlabel = "Variability exponent",
              ygridvisible = false)
    layers_to_plot = [2, 3, 4, 5]
    structure_offsets = [-0.375, -0.225, -0.075, 0.075, 0.225, 0.375] .* 0.9  # Offsets for each structure within a layer

    vlines!(ax, 0.00; color = :gray, linewidth = 3, linestyle = :dash)
    hlines!(ax, layers_to_plot[1:(end - 1)] .- 0.5; color = :gray, linewidth = 1)

    # Perform statistical tests for each structure-layer combination
    𝑝 = map(_c) do b
        𝑝 = HypothesisTests.SignedRankTest(collect(b))
        𝑝 = HypothesisTests.pvalue(𝑝, tail = :right)
    end
    𝑝[:] .= MultipleTesting.adjust(collect(𝑝)[:], BenjaminiHochberg())

    # Plot boxes for each layer and structure
    for (layer_idx, layer) in enumerate(layers_to_plot)
        for (struct_idx, structure) in enumerate(lookup(_c, Structure))
            x = _c[Structure = At(structure), layer = At(layer)]
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
    axislegend(ax, position = :lb, merge = true)
    display(ssf)
end
# begin # * plot
#     begin
#         ax = Axis(f[1, 2]; ylabel = "Cortical layer",
#                   xlabel = "Median fano factor exponent",
#                   title = "Fano factor exponent", yreversed = true,
#                   yticks = (2:5, ["L2/3", "L4", "L5", "L6"]))

#         _b = _c
#     end

#     begin # * Do statistical test against value of 0.5
#         𝑝 = map(_b) do b
#             𝑝 = HypothesisTests.SignedRankTest(collect(b))
#             𝑝 = HypothesisTests.pvalue(𝑝, tail = :right)
#         end
#         𝑝[:] .= MultipleTesting.adjust(collect(𝑝)[:], BenjaminiHochberg())
#     end
#     for l in lookup(_b, :layer)
#         if l > 1
#             x = _b[layer = At(l)]
#             p = 𝑝[layer = At(l)]
#             _x = filter(!isnan, collect(x)[:])
#             if p < PTHR
#                 boxplot!(ax, fill(l, length(_x)), _x;
#                          orientation = :horizontal, show_outliers = false,
#                          color = Foresight.colororder[l - 1],
#                          strokecolor = Foresight.colororder[l - 1], strokewidth = 3)
#             else
#                 boxplot!(ax, fill(l, length(_x)), _x;
#                          orientation = :horizontal, show_outliers = false,
#                          color = :transparent,
#                          strokecolor = Foresight.colororder[l - 1], strokewidth = 3)
#             end
#         end
#     end
# end
display(f)
wsave(plotdir("fano_factor", "fano_factors.pdf"), f)
wsave(plotdir("fano_factor", "supplemental_fano_factors.pdf"), ssf)
