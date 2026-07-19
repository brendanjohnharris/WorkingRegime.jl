#! /bin/bash
#=
exec julia +1.12 -t auto "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.quickactivate("WRExperiment")
using Distributed

begin
    # collect is single-process now that the per-channel fits moved to send_madev --- no workers needed.
    expr = quote
        using DrWatson
        using WRExperiment
        using AcademicClusters
        using Distributed
        using ProgressLogging
        using DimensionalData
        using Peaks
        using FileIO
        using JLD2
        using IntervalSets
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
        import TimeseriesTools: freqs
        calcdir = DrWatson.datadir
    end
    @eval $expr
end

# * Collect plot data
begin
    stimuli = [r"Natural_Images", "spontaneous", "flash_250ms"]

    # ============================================================================ #
    # Step 1: MAD / Diffusion exponent data (from fig2_mad.jl / diffusion_exponent.jl)
    # ============================================================================ #
    madev_data, _ = produce_or_load(
        Dict(), calcdir("plots");
        filename = savepath("mad_psd")
    ) do _
        session_table = load(
            calcdir("plots", "posthoc_session_table.jld2"),
            "session_table"
        )
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
                        chi = load(filename, "chi")
                        return mad, coeffs, S, chi
                    end
                    out = filter(!isnothing, out)
                    out = filter(out) do x # Remove sessions that don't have data to a reasonable depth
                        maximum(DimensionalData.metadata(x[1])[:streamlinedepths]) > 0.9
                    end

                    S = getindex.(out, 3)
                    chi = getindex.(out, 4)
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
                        @assert issorted(lookup(o, Depth))
                        o = set(o, Depth => unidepths)
                    end
                    chi = map(chi, streamlinedepths, layerinfo) do o, s, l
                        o = set(o, Depth => s)
                        layernames = ToolsArray(l[1], (Depth(lookup(o, Depth)),))
                        layernums = ToolsArray(l[3], (Depth(lookup(o, Depth)),))
                        o = o[Depth(Near(unidepths))]
                        layernames = layernames[Depth(Near(unidepths))]
                        layernums = layernums[Depth(Near(unidepths))]
                        @assert issorted(lookup(o, Depth))
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
                    chi = stack(SessionID(sessions), chi, dims = 2) .|> Float32
                    S = stack(SessionID(sessions), S, dims = 3) .|> Float32
                    layernums = parselayernum.(layernames)
                    return mad, coeffs, S, chi, layernames, layernums
                end
                M, coeffs, S, chi, layernames, layernums = zip(M...)
            end
            begin # * Format layers
                meanlayers = map(layernums) do l
                    round.(Int, mean(l, dims = 2))
                end
                M̄ = map(M, meanlayers) do m, l
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
                layerints = load(calcdir("plots", "grand_unified_layers.jld2"), "layerints")
            end

            coeffs = ToolsArray(collect(coeffs), (Structure(structures),))
            begin # * Format layers
                coeffs_median = map(coeffs, meanlayers) do m, l
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

            # Spectral χ is now fit per-channel in send_madev and loaded/stacked above; just wrap per structure.
            χ = ToolsArray(collect(chi), (Structure(structures),))
            mapple = Dict("χ" => χ)

            plot_data = @strdict M M̄ S S̄ coeffs coeffs_median layernames layernums layerints meanlayers oursessions Q mapple
            return plot_data
        end
        @info "Saving to $filename"
        return Dict(string.(stimuli) .=> plot_data)
    end
end
begin
    # ============================================================================ #
    # Step 2: Fano factor data (from fano_factors.jl)
    # ============================================================================ #
    fano_data, _ = produce_or_load(
        Dict(), calcdir("plots");
        filename = savepath("fano_factor")
    ) do _
        session_table = load(
            calcdir("plots", "posthoc_session_table.jld2"),
            "session_table"
        )
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
                unitdepths = map(lookup(Q, Structure)) do structure
                    out = map(lookup(Q, SessionID)) do sessionid
                        if Q[SessionID = At(sessionid), Structure = At(structure)] == 0
                            return nothing
                        end
                        filename = savepath(
                            (@strdict sessionid structure stimulus), "jld2",
                            path
                        )
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
                end
            end

            plot_data = @strdict unitdepths
            return plot_data
        end

        return Dict(string.(stimuli) .=> plot_data)
    end
end
begin
    # ============================================================================ #
    # Step 3: Spectral exponent data (from fig2_reduced.jl / spectral_exponent.jl)
    # ============================================================================ #
    stimuli_str = ["r\"Natural_Images\"", "spontaneous", "flash_250ms"]

    data_file = savepath("mad_psd.jld2")(Dict())
    fig2_data = load(datadir("plots", data_file))

    spectral_exponents = Dict{String, Any}()
    spectral_curves = Dict{String, Any}()

    for stimulus in stimuli_str
        @info "Extracting spectral data for $stimulus"
        @unpack S, S̄, mapple, layernums = fig2_data[string(stimulus)]

        _b = map(2:5) do layer
            _b = map(structures) do structure
                a = mapple["χ"][Structure = At(structure)]
                S_struct = S[Structure = At(structure)]
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

        spectral_exponents[string(stimulus)] = _b

        # Pre-compute PSD plot data per structure (reproduces plotspectrum! output)
        psd_per_structure = Dict{String, Any}()
        psdrange = WRExperiment.PSD_RANGE[1] * u"Hz" .. WRExperiment.PSD_RANGE[2] * u"Hz"
        for structure in structures
            s = S̄[Structure = At(structure)][layer = At([2])][𝑓 = psdrange]
            μ = median(s, dims = (SessionID, :layer))
            μ = dropdims(μ, dims = (SessionID, :layer)) |> ustripall
            σl = map(x -> quantile(x[:], 0.25), eachslice(s, dims = 𝑓)) |> ustripall
            σh = map(x -> quantile(x[:], 0.75), eachslice(s, dims = 𝑓)) |> ustripall
            f = collect(freqs(μ))

            # Peaks
            pks, proms = findpeaks(μ, 2; N = 2)
            peak_freqs = collect(freqs(pks))
            peak_vals = collect(pks)

            # MAPPLE fit
            ff, ps = WRExperiment.mapple_fit(μ)
            mapple = collect(ff.(f))

            # Spectral exponent for this structure, L2/3
            mb = _b[Structure = At(structure), layer = At(2)]
            spectral_exponent_median = median(mb)

            psd_per_structure[structure] = (;
                f, μ = collect(μ), σl = collect(σl),
                σh = collect(σh),
                peak_freqs, peak_vals, mapple,
                spectral_exponent_median,
            )
        end
        spectral_curves[string(stimulus)] = psd_per_structure
        @info "Extracted spectral data for $stimulus: size = $(size(_b))"
    end

    # ============================================================================ #
    # Step 4: Compute fano factor slopes per layer (from fano_factors.jl)
    # ============================================================================ #
    fanorange = 10^(1.5) .. 1.0e3
    fano_slopes = Dict{String, Any}()

    for stimulus in stimuli
        stim_str = string(stimulus)
        unitdepths_all = fano_data[stim_str]["unitdepths"]

        _c = map(Dim{:layer}(2:5)) do l
            structureslopes = map(structures) do structure
                struct_idx = findfirst(structures .== structure)
                unitdepths = unitdepths_all[struct_idx]
                _c = map(unitdepths) do units
                    units = subset(units, :layer => ByRow(==(l)))
                    if isempty(units)
                        return nothing
                    else
                        slopes = map(units.fano_factor) do fano
                            if (fano isa Number) && isnan(fano)
                                return NaN
                            end
                            fano_range = fano[fanorange] .|> log10
                            t_range = times(fano_range) .|> log10
                            X = hcat(ones(length(t_range)), t_range)
                            m = X \ fano_range
                            return last(m)
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

        fano_slopes[stim_str] = _c
        @info "Computed fano slopes for $stim_str: size = $(size(_c))"
    end

    # ============================================================================ #
    # Step 5: Compute diffusion exponent hierarchical correlations (from fig2_mad.jl)
    # ============================================================================ #
    diffusion_hierarchical = Dict{String, Any}()

    for stimulus in stimuli
        stim_str = string(stimulus)
        @unpack coeffs, oursessions = madev_data[stim_str]

        N = 10000
        method = :group
        χ = coeffs
        coeffs_indexed = getindex.(coeffs, [SessionID(At(oursessions))])
        coeffs_indexed = coeffs_indexed[Structure = At(structures)]

        unidepths = commondepths(lookup.(χ, [Depth]))
        x = getindex.([hierarchy_scores], structures)

        unichi = getindex.(coeffs_indexed, [Depth(Near(unidepths))])
        unichi = set.(unichi, [Depth => unidepths])
        y = stack(Structure(structures), unichi)

        μ, σ, 𝑝 = hierarchicalkendall(x, y, method; N)

        diffusion_hierarchical[stim_str] = (; μ, σ, 𝑝, unidepths)
        @info "Computed hierarchical correlation for $stim_str"
    end

    # ============================================================================ #
    # Step 6: Pre-compute MAD curves for plotting (from fig2_mad.jl VISp L2/3 panel)
    # ============================================================================ #
    mad_curves = Dict{String, Any}()

    for stimulus in stimuli
        stim_str = string(stimulus)
        M̄ = madev_data[stim_str]["M̄"]
        m = M̄[Structure = At("VISp")][layer = At(2)]
        if hasdim(m, :layer)
            m = median(m, dims = :layer)
            m = dropdims(m, dims = :layer)
        end

        # Fit exponent in 1ms-10ms range
        _m = m[𝑡 = 1.0e-3 .. 1.0e-2]
        t = log10.(times(_m))
        s = log10.(parent(_m))
        coeff = hcat(ones(length(t)), t) \ s
        intercept, slope = eachrow(coeff)
        meanintercept = median(intercept)
        meanslope = median(slope)

        # Median + IQR
        mu = median(m, dims = SessionID)
        mu = dropdims(mu, dims = SessionID)
        σl = mapslices(x -> quantile(x, 0.25), m; dims = SessionID)
        σh = mapslices(x -> quantile(x, 0.75), m; dims = SessionID)
        σl = dropdims(σl, dims = SessionID)
        σh = dropdims(σh, dims = SessionID)

        mad_curves[stim_str] = (;
            t_all = collect(lookup(mu, 𝑡)),
            mu = collect(mu),
            σl = collect(σl),
            σh = collect(σh),
            fit_t = collect(lookup(_m, 𝑡)),
            meanintercept,
            meanslope,
            intercept = collect(intercept),
            slope = collect(slope),
        )
        @info "Pre-computed MAD curve for $stim_str: slope = $meanslope"
    end

    # ============================================================================ #
    # Step 7: Pre-compute Fano factor curves for plotting (from fano_factors.jl VISp L2/3 panel)
    # ============================================================================ #
    fano_curves = Dict{String, Any}()

    for stimulus in stimuli
        stim_str = string(stimulus)
        structure = "VISp"
        unitdepths = fano_data[stim_str]["unitdepths"][findfirst(structures .== structure)]
        unitdepths = map(unitdepths) do units
            filter(units) do unit
                unit.layer == 2
            end
        end
        unitdepths = filter(!isempty, unitdepths)
        oursessions = map(unitdepths) do u
            only(unique(u.ecephys_session_id))
        end

        fanos = map(unitdepths) do units
            fanos = units.fano_factor
            idxs = map(Base.Fix2(isa, AbstractVector), fanos) # Remove NaN fano curves
            fanos = ToolsArray(fanos[idxs], (Unit(units.ecephys_unit_id[idxs]),)) |> stack
            median_fano = median(fanos, dims = Unit)
            median_fano = dropdims(median_fano, dims = Unit)
        end
        fanos = ToolsArray(fanos, (SessionID(oursessions),)) |> stack
        fanos = fanos[𝑡 = 1 .. 1000]
        mu = median(fanos, dims = SessionID)
        mu = dropdims(mu, dims = SessionID)
        sl = mapslices(x -> quantile(x, 0.25), fanos; dims = SessionID)
        sl = dropdims(sl, dims = SessionID)
        su = mapslices(x -> quantile(x, 0.75), fanos; dims = SessionID)
        su = dropdims(su, dims = SessionID)

        t_range = times(first(eachcol(fanos))[fanorange]) .|> log10
        mfanos = map(eachcol(fanos)) do mu
            fano_range = mu[fanorange] .|> log10
            xx = hcat(ones(length(t_range)), t_range)
            m = xx \ fano_range
        end
        mintercept = median(first.(mfanos))
        mslope = median(last.(mfanos))

        fano_curves[stim_str] = (;
            t_all = collect(times(mu)),
            mu = collect(mu),
            sl = collect(sl),
            su = collect(su),
            fit_t_range = collect(t_range),
            mintercept,
            mslope,
        )
        @info "Pre-computed Fano curve for $stim_str: slope = $mslope"
    end

    # * Check fano factors are unique across stimuli
    let keys_ = collect(keys(fano_curves))
        for i in 1:length(keys_), j in (i + 1):length(keys_)
            a, b = fano_curves[keys_[i]].mu, fano_curves[keys_[j]].mu
            @assert a != b "Fano curves for $(keys_[i]) and $(keys_[j]) are identical!"
            @info "Fano mu diff $(keys_[i]) vs $(keys_[j]): max|Δ| = $(maximum(abs.(a .- b)))"
        end
    end

    # ============================================================================ #
    # Step 8: Save all plot data
    # ============================================================================ #
    outpath = datadir("plots", "WRExperiment.jld2")
    mkpath(dirname(outpath))
    @info "Saving all plot data to $outpath"

    jldsave(
        outpath;
        madev_data,
        fano_data,
        spectral_curves,
        spectral_exponents,
        fano_slopes,
        diffusion_hierarchical,
        mad_curves,
        fano_curves
    )

    @info "Done. Saved plot data to $outpath"
end
