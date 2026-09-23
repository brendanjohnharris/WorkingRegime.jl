#! /bin/bash
#=
exec julia +1.12 -t auto "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.quickactivate("WRExperiment")
using Distributed
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
import TimeseriesSurrogates: RandomFourier, surrogenerator
using Clustering
using LinearAlgebra

# * Figure 1 inputs. Both blocks below were folded in from standalone scripts (`scripts/plots/traces.jl`
# and the increment-histogram script) so that every Figure 1 input is produced here and read from disk
# by `scripts/plots/Fig1_combined_curves.jl`. Each is cached independently, so re-running this script
# does not repeat their LFP loads.

begin # * Example LFP trace + spike raster (Fig 1b)
    traces_data, _ = produce_or_load(
        Dict(), DrWatson.datadir(); filename = savepath("traces")
    ) do _
        sessionid = DEFAULT_SESSION_ID
        structure = "VISp"
        stimulus = "spontaneous"
        chan = 8
        session = AN.Session(sessionid)
        LFP = AN.formatlfp(session; rectify = false, epoch = :longest, structure, stimulus)
        spikes = AN.getspiketimes(session, structure)

        t0lfp = minimum(times(LFP))
        window = (t0lfp + 139) .. (t0lfp + 140)       # a 1 s window well inside the epoch
        y = map(collect(values(spikes))) do s
            s[findall(s .∈ [window])]
        end
        y = filter(x -> length(x) > 5, y)             # only neurons with enough spikes to order
        # Order neurons by spike-train similarity so the raster shows its correlation structure rather
        # than an arbitrary unit ordering. `stoic` returns the similarity matrix; invert for a distance.
        h = hclust(Symmetric(1.0 ./ TimeseriesTools.stoic(y)))
        y = y[h.order]

        x = LFP[𝑡 = window][:, chan] |> ustripall
        t0 = minimum(times(x))
        Dict(
            "t" => collect(times(x)) .- t0,
            "lfp" => collect(x),
            "spikes" => [collect(s) .- t0 for s in y],
            "sessionid" => sessionid, "structure" => structure,
            "stimulus" => stimulus, "channel" => chan
        )
    end
end

begin # * Pooled L2/3 increment distribution against its FT surrogate null (Fig 1, far right)
    increment_histograms, _ = produce_or_load(
        Dict(), DrWatson.datadir(); filename = savepath("increment_histograms")
    ) do _
        n_sessions = nothing                           # nothing = every QC-passing session; an integer takes a draft subset
        structure, stimulus = "VISp", "spontaneous"
        edges = range(-15, 15, length = 601)           # standard deviations, 0.05 SD bins
        sessions = load(
            DrWatson.datadir("session_table.jld2"), "session_table"
        ).ecephys_session_id
        isnothing(n_sessions) || (sessions = sessions[1:min(n_sessions, length(sessions))])

        counts, counts_surr = zeros(Int, length(edges) - 1), zeros(Int, length(edges) - 1)
        kurt, kurt_surr, nchan = Float64[], Float64[], 0   # per channel, pooled across sessions
        kurt_sess, kurt_sess_surr = Float64[], Float64[]   # per session: median over its L2/3 channels
        for (i, sessionid) in enumerate(sessions)
            @info "[$i/$(length(sessions))] increment histogram, session $sessionid"
            try
                session = AN.Session(sessionid)
                LFP = AN.formatlfp(
                    session; tol = 3, sessionid, epoch = :longest, band = (1.0e-3, 1.0e-2),
                    pass = (1, 625), stimulus, structure
                )
                X = Float64.(parent(ustripall(LFP)))
                lnum = parselayernum.(
                    string.(last(AN.getchannellayers(session, collect(lookup(LFP, AN.Chan)))))
                )
                k_this, ks_this = Float64[], Float64[] # this session's per-channel values
                for j in findall(lnum .== 2)           # L2/3
                    d = diff(@view X[:, j])
                    s = diff(surrogenerator(collect(@view X[:, j]), RandomFourier(), Xoshiro(j))())
                    counts .+= StatsBase.fit(Histogram, d ./ std(d), edges).weights
                    counts_surr .+= StatsBase.fit(Histogram, s ./ std(s), edges).weights
                    push!(k_this, kurtosis(d))         # excess kurtosis, as the surrogate sweep computes it
                    push!(ks_this, kurtosis(s))
                    nchan += 1
                end
                append!(kurt, k_this)
                append!(kurt_surr, ks_this)
                # Session-level value: median over this session's L2/3 channels, matching `save_surrogate_statistics` (Fig 4).
                kk, kks = filter(!isnan, k_this), filter(!isnan, ks_this)
                isempty(kk) || push!(kurt_sess, median(kk))
                isempty(kks) || push!(kurt_sess_surr, median(kks))
            catch e
                @warn "Skipping $sessionid" e
            end
        end
        density(c) = c ./ (sum(c) * step(edges))
        Dict(
            "edges" => collect(edges),
            "centres" => collect(edges)[1:(end - 1)] .+ step(edges) / 2,
            "density" => density(counts), "density_surrogate" => density(counts_surr),
            "counts" => counts, "counts_surrogate" => counts_surr,
            "kurtosis" => kurt, "kurtosis_surrogate" => kurt_surr,
            "kurtosis_session" => kurt_sess, "kurtosis_surrogate_session" => kurt_sess_surr,
            "nchannels" => nchan, "sessions" => sessions,
            "structure" => structure, "stimulus" => stimulus
        )
    end
end


# * Collect plot data
begin
    stimuli = [r"Natural_Images", "spontaneous", "flash_250ms"]


    madev_data, _ = produce_or_load(
        Dict(), DrWatson.datadir();
        filename = savepath("mad_psd")
    ) do _
        session_table = load(
            DrWatson.datadir("session_table.jld2"),
            "session_table"
        )
        oursessions = session_table.ecephys_session_id
        path = DrWatson.datadir("calculations")
        QQ = WRExperiment.calcquality(path)
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
                        p = sortperm(s) # streamline depths are per-channel, not depth-ordered; sort so `Near` works
                        o = set(o[Depth(p)], Depth => s[p])
                        layernames = ToolsArray(l[1][p], (Depth(lookup(o, Depth)),))
                        layernums = ToolsArray(l[3][p], (Depth(lookup(o, Depth)),))
                        o = o[Depth(Near(unidepths))]
                        layernames = layernames[Depth(Near(unidepths))]
                        layernums = layernums[Depth(Near(unidepths))]
                        @assert issorted(lookup(o, Depth))
                        push!(o.metadata, :layernames => layernames)
                        push!(o.metadata, :layernums => layernums)
                        o = set(o, Depth => unidepths)
                    end
                    coeffs = map(coeffs, streamlinedepths, layerinfo) do o, s, l
                        p = sortperm(s) # streamline depths are per-channel, not depth-ordered; sort so `Near` works
                        o = set(o[Depth(p)], Depth => s[p])
                        layernames = ToolsArray(l[1][p], (Depth(lookup(o, Depth)),))
                        layernums = ToolsArray(l[3][p], (Depth(lookup(o, Depth)),))
                        o = o[Depth(Near(unidepths))]
                        layernames = layernames[Depth(Near(unidepths))]
                        layernums = layernums[Depth(Near(unidepths))]
                        @assert issorted(lookup(o, Depth))
                        o = set(o, Depth => unidepths)
                    end
                    chi = map(chi, streamlinedepths, layerinfo) do o, s, l
                        p = sortperm(s) # streamline depths are per-channel, not depth-ordered; sort so `Near` works
                        o = set(o[Depth(p)], Depth => s[p])
                        layernames = ToolsArray(l[1][p], (Depth(lookup(o, Depth)),))
                        layernums = ToolsArray(l[3][p], (Depth(lookup(o, Depth)),))
                        o = o[Depth(Near(unidepths))]
                        layernames = layernames[Depth(Near(unidepths))]
                        layernums = layernums[Depth(Near(unidepths))]
                        @assert issorted(lookup(o, Depth))
                        o = set(o, Depth => unidepths)
                    end
                    S = map(S, streamlinedepths, layerinfo) do o, s, l
                        p = sortperm(s) # streamline depths are per-channel, not depth-ordered; sort so `Near` works
                        o = set(o[Depth(p)], Depth => s[p])
                        layernames = ToolsArray(l[1][p], (Depth(lookup(o, Depth)),))
                        layernums = ToolsArray(l[3][p], (Depth(lookup(o, Depth)),))
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
                    mad = cat(out...; dims = SessionID(sessions)) .|> Float32
                    coeffs = cat(coeffs...; dims = SessionID(sessions)) .|> Float32
                    chi = cat(chi...; dims = SessionID(sessions)) .|> Float32
                    S = cat(S...; dims = SessionID(sessions)) .|> Float32
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
                begin # * Grand unified layers (moved from filter_sessions_posthoc): each cortical layer's
                    # normalized-depth span, pooled as min..max across structures and good sessions.
                    layerints = map(WRExperiment.layers) do _l
                        spans = Float64[]
                        for ln in layernames # per structure: depths × (good) sessions of layer-name strings
                            ds = collect(lookup(ln)[1])
                            for j in axes(ln, 2)
                                mask = occursin.(_l, parent(ln)[:, j])
                                any(mask) && append!(spans, collect(extrema(ds[mask])))
                            end
                        end
                        minimum(spans) .. maximum(spans) # ponytail: assumes each layer appears in ≥1 good session
                    end
                    @assert length(layerints) == length(WRExperiment.layers)
                    if stimulus == r"Natural_Images" # save once; layer anatomy is stimulus-independent
                        tagsave(
                            DrWatson.datadir("grand_unified_layers.jld2"),
                            Dict("layerints" => layerints)
                        )
                    end
                end
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
    fano_data, _ = produce_or_load(
        Dict(), DrWatson.datadir();
        filename = savepath("fano_factor")
    ) do _
        session_table = load(
            DrWatson.datadir("session_table.jld2"),
            "session_table"
        )
        oursessions = session_table.ecephys_session_id
        path = DrWatson.datadir("calculations")
        QQ = WRExperiment.calcquality(path)
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


    stimuli_str = ["r\"Natural_Images\"", "spontaneous", "flash_250ms"]

    fig2_data = load(datadir("mad_psd.jld2")) # block 1 saved this under datadir() via produce_or_load

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
            common = intersect(lookup.(_b, SessionID)...) # structures keep different session subsets; align to shared
            _b = map(p -> p[SessionID = At(common)], _b)
            _b = cat(_b...; dims = Structure(structures))
        end
        _b = cat(_b...; dims = Dim{:layer}(2:5))

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

            # MAPPLE fit
            ff, ps = WRExperiment.mapple_fit(μ)
            mapple = collect(ff.(f))

            # Spectral exponent for this structure, L2/3
            mb = _b[Structure = At(structure), layer = At(2)]
            spectral_exponent_median = median(mb)

            psd_per_structure[structure] = (;
                f, μ = collect(μ), σl = collect(σl),
                σh = collect(σh),
                mapple,
                spectral_exponent_median,
            )
        end
        spectral_curves[string(stimulus)] = psd_per_structure
        @info "Extracted spectral data for $stimulus: size = $(size(_b))"
    end


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
                        # Drop the units with no fano curve BEFORE reducing over Unit. Without this the
                        # `mean` below propagates a single unit's NaN over the whole session: 11% of units
                        # carry the NaN sentinel, which NaN-ed 77% of the (session, layer) cells even though
                        # no cell has every unit missing. Mirrors the `isa AbstractVector` guard the
                        # fano_curves block already applies for the same reason.
                        keep = findall(!isnan, slopes)
                        isempty(keep) && return nothing
                        return ToolsArray(slopes[keep], (Unit(units.ecephys_unit_id[keep]),)) |> stack
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


    mad_curves = Dict{String, Any}()

    for stimulus in stimuli
        stim_str = string(stimulus)
        M̄ = madev_data[stim_str]["M̄"]
        m = M̄[Structure = At("VISp")][layer = At(2)]
        if hasdim(m, :layer)
            m = median(m, dims = :layer)
            m = dropdims(m, dims = :layer)
        end


        _m = m[𝑡 = WRExperiment.MAD_BAND[1] .. WRExperiment.MAD_BAND[2]]      # kept only for `fit_t`, the drawn line's x-axis
        fits = map(eachslice(m, dims = SessionID)) do col
            try
                diffusion_line(col)
            catch err
                @warn "diffusion_line failed for one session; dropping it" err
                (NaN, NaN)
            end
        end
        intercept = collect(first.(fits))
        slope = collect(last.(fits))
        meanintercept = median(filter(!isnan, intercept))
        meanslope = median(filter(!isnan, slope))

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

        # The variability exponent of the drawn median. Fit HERE rather than in Figure 1, so the
        # figure and the manuscript quote one cached number: the estimator draws its multistart
        # restarts from the global RNG (hence `VARIABILITY_SEED`), and a quantity refitted at draw
        # time has no saved value to check against. `mslope` above is the legacy fixed-band OLS and
        # is kept only for comparison.
        cfit = variability_exponent(collect(times(mu)), collect(mu))

        fano_curves[stim_str] = (;
            t_all = collect(times(mu)),
            mu = collect(mu),
            sl = collect(sl),
            su = collect(su),
            fit_t_range = collect(t_range),
            mintercept,
            mslope,
            cfit,
        )
        @info "Pre-computed Fano curve for $stim_str" mslope beta=cfit.β band=(cfit.lo, cfit.hi)
    end

    # * Check fano factors are unique across stimuli
    let keys_ = collect(keys(fano_curves))
        for i in 1:length(keys_), j in (i + 1):length(keys_)
            a, b = fano_curves[keys_[i]].mu, fano_curves[keys_[j]].mu
            @assert a != b "Fano curves for $(keys_[i]) and $(keys_[j]) are identical!"
            @info "Fano mu diff $(keys_[i]) vs $(keys_[j]): max|Δ| = $(maximum(abs.(a .- b)))"
        end
    end


    outpath = datadir("WRExperiment.jld2")
    mkpath(dirname(outpath))
    @info "Saving all plot data to $outpath"

    jldsave(
        outpath;
        madev_data,
        fano_data,
        spectral_curves,
        spectral_exponents,
        fano_slopes,
        mad_curves,
        fano_curves
    )

    @info "Done. Saved plot data to $outpath"
end
