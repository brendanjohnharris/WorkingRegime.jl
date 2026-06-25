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
using MoreMaps
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Load manifest
    parameter_grid, hash_grid, default_params = load(datadir("spatial_sampling",
                                                             "parameter_grid.jld2"),
                                                     "parameter_grid", "hash_grid",
                                                     "default_params")
    # dx = default_params[:parameters][:dx]
end
begin # * Check file quality
    computed_hashes = readdir(datadir("spatial_sampling")) .|> splitext .|> first
    computed_hashes = filter(x -> all(isdigit, x), computed_hashes)
    computed_hashes = map(Base.Fix1(parse, UInt), computed_hashes)
    Q = map(x -> false, hash_grid)
    for h in computed_hashes
        if h in hash_grid
            Q[findfirst(==(h), hash_grid)] = true
        end
    end
end
begin # * Loop through files and plot
    @info "Processing $(sum(Q)) files"
    Q = map(Chart(ProgressLogger(100)), hash_grid[Q]) do hash
        file = string(hash) * ".jld2"

        parameters = load(datadir("spatial_sampling", file), "parameters")
        title = parameters[[:K_ee, :K_ei, :K_ie, :K_ii, :nu, :J_ei]]
        title = map(keys(title), values(title)) do k, v
                    k => round(v, sigdigits = 3)
                end |> NamedTuple |> WRCircuit.sortparams |> string

        spikes = load(datadir("spatial_sampling", file), "monitor")["E.spike"]

        if isfile(datadir("spatial_sampling", file))
            animation_file = plotdir("spatial_sampling", "animation", "$hash.mp4")
            if isfile(animation_file)
                @info "Animation for $hash already exists"
            else
                dx = parameters[:dx]
                rates = WRCircuit.compute_rates(spikes, 50u"ms")
                WRCircuit.animate_rates(rates, dx;
                                        filename = animation_file,
                                        axis = (; title, titlesize = 12))
            end

            # * Now do MUA spectrum from local patch
            mua_file = plotdir("spatial_sampling", "mua_spectrum", "$hash.png")
            if isfile(mua_file)
                @info "MUA spectrum for $hash already exists"
            else
                # * MUA in local patch
                dx = parameters[:dx]
                rho = parameters[:rho]
                dn = round(Int, sqrt(rho * dx^2))
                positions = WRCircuit.positions.GridPositions((dx, dx))((dn, dn))  # Maybe think about capturing this somehow
                positions = map(positions) do pos
                    map(pos) do p
                        p.tolist() |> convert2(Float32)
                    end
                end

                radius = 0.1 # mm
                origin = [dx / 2, dx / 2]
                mask = map(positions) do pos
                    dp = abs.(pos .- origin)
                    dp = min.(dp, dx .- dp)
                    norm(dp) < radius
                end
                local_idxs = findall(mask)

                mua = WRCircuit.compute_rates(spikes[:, local_idxs], 2u"ms")
                mua = mean(mua, dims = Neuron)
                mua = dropdims(mua, dims = Neuron)
                mua = Float64.(mua)

                mua = mua .- mean(mua)
                mua = set(mua, 𝑡 => uconvert.(u"s", times(mua)))
                mua = rectify(mua, dims = 𝑡, tol = 2)
                s = spectrum(mua, 0.5u"Hz")
                limits = ((1, 250), (extrema(s) .|> ustrip))
                f = Figure()
                ax = Axis(f[1, 1]; title = title, titlesize = 12, xscale = log10,
                          yscale = log10, limits)
                plotspectrum!(ax, s)

                # * Inset firing rate distributions
                rates = mean(spikes, dims = 𝑡) ./ step(spikes)
                axx = Axis(f[1, 1];
                           width = Relative(0.3),
                           height = Relative(0.3),
                           halign = 0.05,
                           valign = 0.05,
                           xticklabelsize = 12,
                           xgridvisible = false,
                           ygridvisible = false,
                           backgroundcolor = :white,
                           xaxisposition = :top)
                hideydecorations!(axx)
                hist!(axx, rates[:] |> ustrip, normalization = :pdf)
                wsave(mua_file, f)
            end

            return true
        else
            @warn "File $file not found but quality check indicated it should be present"
            return false
        end
    end
end
