using TimeseriesTools
import AllenNeuropixelsBase as AN
using Unitful
using Makie
using LinearAlgebra
using Fathom
using StatsBase

import DataFrames: DataFrame, innerjoin
import JLD2: jldopen
import ProgressLogging: @withprogress, @logprogress
import DimensionalData
import Dates: TimeType
import Bootstrap
import MultipleTesting: adjust, BenjaminiHochberg
import HypothesisTests: MannWhitneyUTest, SignedRankTest, pvalue

# Band over which the spectral exponent is fit, IN HERTZ. The upper edge is the acquisition
# anti-aliasing corner, NOT the Nyquist frequency: the Allen LFP is stored at 1250 Hz (Nyquist
# 625 Hz), but the LFP band is filtered at ~500 Hz before storage, so 505-625 Hz is filter rolloff
# falling to a flat noise floor (VISp spontaneous median: local slope -14 over 500-625 Hz, PSD down
# 20x by 550 Hz and flat thereafter). Fitting through it steepens the aperiodic slope by ~0.16
# (median curve: b = -1.49 over 3-500 Hz vs -1.65 over 3-625 Hz), so the cliff is excluded.
const PSD_RANGE = [3, 500]

# Band over which the diffusion exponent is fit, IN SECONDS (the LFP lag axis is in seconds; the
# circuit's `WRCircuit.MAD_BAND_MS` is in milliseconds, so the numeric values differ by 1000x even
# though they denote the SAME physical band).
#
# 0.8-8 ms: the first decade of the lag grid, from one sample at 1250 Hz to a decade later. The
# circuit is fit over this identical band, so the two exponents are the same quantity. A model-free
# knee (the lag at which the local slope falls to half its short-lag value) sits at 8.8 ms median
# across sessions, IQR 7.6-10.4, so the band ends essentially at the edge of the scaling regime.
# NB it is that model-free estimate the band is justified against, NOT `breakfrequencies` of a
# 2-component fit: the latter's threshold is (β₁+β₂)/2, which moves with the unstable fitted
# exponents and scatters over 4x, so it is not a usable knee on these curves.
const MAD_BAND = [0.0, 8.0e-3]

function produce_unitdepths(session::AN.AbstractSession)
    sessionid = AN.getid(session)
    @info "Computing depths for session $sessionid"
    probes = AN.getprobes(session)
    unitmetrics = AN.getunitmetrics(session)
    cdf = AN.getchannels(session)
    probedepths = Dict{Any, Any}()
    streamlinedepths = Dict{Any, Any}()
    for probe in probes.id
        _cdf = cdf[cdf.probe_id .== probe, :]
        channels = _cdf.id
        _probedepths = Dict(
            channels .=>
                AN._getchanneldepths(_cdf, channels; method = :probe)
        )
        _streamlinedepths = Dict(
            channels .=>
                AN._getchanneldepths(_cdf, channels; method = :streamlines)
        )
        p = Dict(channels .=> getindex.([_probedepths], channels))
        s = Dict(channels .=> getindex.([_streamlinedepths], channels))
        merge!(probedepths, p)
        merge!(streamlinedepths, s)
    end
    D = DataFrame(
        [
            keys(probedepths) |> collect, values(probedepths) |> collect,
            getindex.([streamlinedepths], keys(probedepths)),
        ],
        [:peak_channel_id, :probedepth, :streamlinedepth]
    )
    D = innerjoin(D, unitmetrics, on = :peak_channel_id)
    D = innerjoin(D, cdf, on = :peak_channel_id => :id)
    D.ecephys_session_id .= sessionid
    return D
end

# Spectral aperiodic fit via MAPPLE (replaces the FOOOF/AllenNeuropixels `aperiodicfit`). Mirrors WRCircuit's
# `spectral_exponents`: log-sample the PSD over PSD_RANGE, fit a 1-component/1-peak MAPPLE, read the aperiodic
# slope. χ is returned with FOOOF's sign (positive), so the pipeline's `-χ` gives the same (negative) spectral
# exponent and matches the circuit. Returns `(ff, Dict(:χ,:b,:k))` --- same shape callers expect from FOOOF.
# NOTE: MAPPLE's `fit!` lives in TimeseriesTools' OptimExt --- needs Optim + ForwardDiff loaded or it NaNs.
function mapple_fit(x; kwargs...)
    p = logsample(ustripall(x)[𝑓 = PSD_RANGE[1] .. PSD_RANGE[2]])
    p = p ./ maximum(p) # lift off eps(Float32): raw PSD (~1e-12 V²/Hz) underflows MAPPLE's _safelog10 floor; β is scale-invariant
    m = fit(MAPPLE, p; components = 1, peaks = 2)  # 2 peaks: absorb both the ~7 Hz and gamma bumps (FOOOF removed up to 8) so the aperiodic slope isn't biased
    fit!(m, p)
    β = last(m.params.components.β)
    ff = f -> only(mapple([float(f)], m.params))
    return ff, Dict(:χ => -β, :b => m.params.log_A, :k => 0.0)
end

# Diffusion exponent via MAPPLE (matches WRCircuit's `diffusion_exponents`): the first component of a
"""
    diffusion_fit(mad_col; band = MAD_BAND)

Diffusion exponent: the slope of a 1-component MAPPLE fit to the MAD curve restricted to `band`
(seconds). One component means the model is a single power law, so `β` IS the log-log slope of the
band, with no breakpoint and no crossfade for a second component to leak through.

The restriction is the point. A 2-component fit over the full 0.8-1000 ms curve returns a `first(β)`
that is the τ→0 asymptote of a segment blended with its neighbour, not a slope the data exhibits
anywhere (0.756 against a true band slope of ~0.52). The MAD saturates by ~60 ms, so two of the
three decades carry no exponent; `band` keeps the fit inside the scaling regime. Use
[`diffusion_knee`](@ref) to confirm per curve that the band ends before the knee.

`w = true` weights by log-spacing: after rounding to whole samples the short lags land on
consecutive integers (linearly spaced) while long lags stay log-spaced, so an unweighted fit would
be dominated by the log-denser upper end of the band. Requires TimeseriesTools with `logweights`.
"""
diffusion_fit(mad_col; band = MAD_BAND) = last(diffusion_line(mad_col; band))

"""
    diffusion_line(mad_col; band = MAD_BAND) -> (intercept, slope)

The same fit as [`diffusion_fit`](@ref), returned as the log-log LINE it is:
`log10(MAD) = intercept + slope·log10(τ)`. A 1-component, 0-peak MAPPLE is a single power law, so the
line IS the model rather than an approximation to it, and `slope` is exactly `diffusion_fit`'s
exponent. Use this wherever the fit is drawn, so the line on the figure and the exponent in the
caption cannot come from different estimators.
"""
function diffusion_line(mad_col; band = MAD_BAND)
    y = ustripall(mad_col)[𝑡 = band[1] .. band[2]]
    m = fit(MAPPLE, y; components = 1, peaks = 0)
    fit!(m, y; w = true)
    β = first(m.params.components.β)
    τ₀ = first(lookup(y, 𝑡))                       # any band lag pins the intercept; the model is a line
    return log10(only(predict(m, [τ₀]))) - β * log10(τ₀), β
end

"""
    diffusion_knee(mad_col)

Crossover lag (seconds) where the MAD curve leaves its scaling regime: the first breakpoint of an
unconstrained 2-component MAPPLE fit over the whole curve. Fit separately from
[`diffusion_fit`](@ref) and used only to locate the knee, never to read an exponent off; the free
breakpoint moves with the model order, so the segment slopes either side of it are not stable
quantities, whereas the knee itself is.
"""
function diffusion_knee(mad_col)
    y = ustripall(mad_col)
    m = fit(MAPPLE, y; components = 2, peaks = 0)
    fit!(m, y; w = true)
    return first(breakfrequencies(m))
end

# Warn if the exponent band runs past the knee on a meaningful fraction of curves: the band is only
# defensible while it stays inside the scaling regime, and that is an empirical claim per dataset.
function confirm_band_before_knee(knees, band = MAD_BAND; tol = 0.05)
    k = filter(isfinite, collect(knees))
    isempty(k) && return NaN
    bad = count(<(band[2]), k) / length(k)
    if bad > tol
        @warn "Diffusion band ends after the knee on $(round(100bad; digits = 1))% of curves; \
               the exponent is being read partly past the scaling regime" band median_knee = median(k)
    else
        @info "Diffusion band OK: knee median $(round(1000median(k); digits = 2)) ms vs band end \
               $(1000band[2]) ms ($(round(100bad; digits = 1))% of curves knee early)"
    end
    return median(k)
end

# 1-D connected components: id increments at each change (replaces ImageMorphology.label_connected_components).
function _runlabel(v)
    o = ones(Int, length(v))
    for i in 2:length(v)
        o[i] = v[i] == v[i - 1] ? o[i - 1] : o[i - 1] + 1
    end
    return o
end

# Replaces AllenNeuropixels.Plots._layerplot (that submodule went with AllenNeuropixels). Same body; only
# [1]=layers and [3]=unids are consumed downstream, the rest are kept for signature parity.
function _layerinfo(session, channels::AbstractVector{<:Int})
    channels = AN.sortbydepth(session, channels; method = :probe)
    depths = AN.getchanneldepths(session, channels; method = :probe)
    depths = Depth(depths)
    depths = first(rectify(depths))
    layerids, layers = AN.getchannellayers(session, channels)
    unids = _runlabel(Int.(indexin(layers, unique(layers))))
    dr = [0; diff(unids)] .> 0
    cs = [mean(depths[unids .== x]) for x in unique(unids)]
    return layers, depths, unids, dr, cs
end

function madev(x::AbstractVector, lags; p = 1)
    if !issorted(lags)
        throw(ArgumentError("Lags must be sorted"))
    end
    l = length(x)
    result = similar(x, length(lags))
    @inbounds for (i, k) in enumerate(lags)
        if k >= l
            result[i] = 0.0
        else
            n_pairs = l - k
            x1 = @view x[1:n_pairs]           # x[1:n-k]
            x2 = @view x[(k + 1):(k + n_pairs)]   # x[k+1:n]

            result[i] = norm(x1 .- x2, p) / n_pairs
        end
    end
    return result
end
function madev(x::UnivariateRegular, _lags; kwargs...)
    lags = round.(Int, _lags ./ TimeseriesTools.samplingperiod(x))
    return Timeseries(madev(parent(x), lags; kwargs...), _lags)
end
function madev(x::MultivariateRegular, _lags; kwargs...)
    lags = round.(Int, _lags ./ TimeseriesTools.samplingperiod(x))
    d = dims(x)[2:end]
    m = mapslices(x -> madev(x, lags; kwargs...), parent(x); dims = 1)
    return Timeseries(m, _lags, d...)
end

"""
    madev_taus(dt)

The pipeline's MAD lag grid: 50 log-spaced lags over 1 ms--1 s, reduced to DISTINCT whole-sample
lags at `dt` (in seconds). `madev` labels each point with the requested τ, so a grid finer than the
sample period would return tied values at a moving x; distinct integer lags make every point an
independent measurement at its true lag.
"""
madev_taus(dt) = filter(>=(1), unique(round.(Int, exp10.(range(-3, 0, 50)) ./ dt))) .* dt

function _count(spike_times, τ; bins = minimum(spike_times):τ:maximum(spike_times))
    return fit(Histogram, spike_times, bins).weights
end

function rates(spike_times, τ)
    bins = minimum(spike_times):τ:maximum(spike_times)
    if isempty(bins)
        return bins .* NaN
    else
        counts = _count(spike_times, τ; bins)
        return counts ./ τ
    end
end

function fano_factor(spike_times::AbstractVector{<:Number}, τ::Number)
    counts = _count(spike_times, τ)
    m = mean(counts)
    return var(counts, mean = m) / m
end

function fano_factor(
        spike_times::AbstractVector{<:Number},
        τ_values::AbstractVector = defaultfanobins(spike_times)
    )
    f = [fano_factor(spike_times, τ) for τ in τ_values]
    return Timeseries(f, τ_values)
end

function send_madev(
        sessionid, stimulus, structure;
        outpath = DrWatson.datadir("calculations"),
        # plotpath = calcdir("plots", "madev")
    )
    params = (;
        sessionid,
        epoch = :longest,
        band = (1.0e-3, 1.0e-2),
        pass = (1, 625),
    ) # * Right up the nyquist frequency

    ssession = []
    GC.safepoint()
    @info "Loading $(stimulus) LFP in $(structure) for session $(params[:sessionid])"
    _params = (; params..., stimulus, structure)
    if stimulus == r"Natural_Images"
        _params = (; _params..., epoch = (:longest, :active))
    end
    outfile = savepath(
        Dict(
            "sessionid" => params[:sessionid],
            "stimulus" => stimulus,
            "structure" => structure
        ), "jld2", outpath
    )
    @info outfile

    fstimulus = _params[:stimulus] isa Regex ? _params[:stimulus].pattern :
        _params[:stimulus]

    # plotfile = joinpath(
    #     plotpath, "$(_params[:sessionid])",
    #     "$(fstimulus)_$(_params[:structure]).pdf"
    # )
    # psdplotfile = joinpath(
    #     plotpath, "$(_params[:sessionid])_psd",
    #     "$(fstimulus)_$(_params[:structure]).pdf"
    # )

    if isempty(ssession) # Only initialize session if we have to
        session = AN.Session(params[:sessionid])
        push!(ssession, session)
    else
        session = ssession[1]
    end
    try
        probestructures = AN.getprobestructures(session)
        probestructures = unique(vcat(values(probestructures)...))
        if structure ∉ probestructures
            str = "Region error: structure $(structure) not found in $(params[:sessionid])"
            @warn str
            tagsave(outfile, Dict("error" => str))
            return
        end

        LFP = AN.formatlfp(session; tol = 3, _params...)u"V"

        LFP = set(LFP, 𝑡 => 𝑡((times(LFP))u"s"))
        channels = lookup(LFP, Chan)

        begin # * Full power spectra
            S = powerspectrum(LFP, 0.1; padding = 10000)
            S = S[𝑓(params[:pass][1] * u"Hz" .. params[:pass][2] * u"Hz")]
            depths = AN.getchanneldepths(session, LFP; method = :probe)
            S = set(S, Chan => Depth(depths))

            # begin
            #     f = Figure()
            #     colorrange = extrema(depths)
            #     ax = Axis(
            #         f[1, 1]; xscale = log10, yscale = log10,
            #         xtickformat = "{:.1f}",
            #         limits = (params[:pass], (nothing, nothing)),
            #         xgridvisible = true,
            #         ygridvisible = true, topspinevisible = true,
            #         title = "$(_params[:structure]), $(_params[:stimulus])",
            #         xminorticksvisible = true, yminorticksvisible = true,
            #         xminorgridvisible = true, yminorgridvisible = true,
            #         xminorgridstyle = :dash
            #     )
            #     p = traces!(
            #         ax, S[2:end, :]; colormap = cgrad(sunset, alpha = 0.4),
            #         linewidth = 3, colorrange
            #     )

            #     c = Colorbar(
            #         f[1, 2]; label = "Channel depth (μm)", colorrange,
            #         colormap = sunset
            #     )
            #     # rowsize!(f.layout, 1, Relative(0.8))
            #     mkpath(joinpath(plotpath, "$(_params[:sessionid])_psd"))
            #     wsave(psdplotfile, f)
            #     @info "Saved plot to `$psdplotfile`"
            # end
        end

        lfp = LFP |> ustripall            # dt from the same object madev sees, so the two agree by construction
        dt = TimeseriesTools.samplingperiod(lfp)
        taus = madev_taus(dt)             # distinct whole-sample lags; see the docstring
        mad = madev(lfp, taus)
        depths = AN.getchanneldepths(session, LFP; method = :probe)
        mad = set(mad, Chan => Depth(depths))
        # mad_fit = mad[𝑡 = pass[1] .. pass[2]]

        # * Mean fit (MAPPLE diffusion exponent, matching the circuit)
        coeff = diffusion_fit(median(mad, dims = Depth)[:, 1])

        # * Full fit (per-channel MAPPLE diffusion exponent)
        coeffs = map(diffusion_fit, eachslice(mad, dims = 2))

        # * Knee per channel, from an unconstrained 2-component fit. Saved as an observable in its
        #   own right (it is the crossover out of the scaling regime) and used to confirm the
        #   exponent band sits before it.
        knees = map(diffusion_knee, eachslice(mad, dims = 2))
        confirm_band_before_knee(knees)

        # * Spectral fit (per-channel MAPPLE aperiodic exponent; moved here from collect_calculations so a and
        #   b are both fit at the source, parallel across the session-level distributed jobs). Only χ is used.
        chi = map(s -> last(mapple_fit(s))[:χ], eachslice(S, dims = 2))

        mmad = mad ./ maximum(mad, dims = 1)
        # begin
        #     f = Figure()
        #     colorrange = extrema(depths)
        #     ax = Axis(
        #         f[1, 1]; xscale = log10, yscale = log10,
        #         xtickformat = "{:.1f}",
        #         xgridvisible = true,
        #         ygridvisible = true, topspinevisible = true,
        #         title = "$(_params[:structure]), $(_params[:stimulus])",
        #         xminorticksvisible = true, yminorticksvisible = true,
        #         xminorgridvisible = true, yminorgridvisible = true,
        #         xminorgridstyle = :dash
        #     )
        #     p = traces!(
        #         ax, mmad; colormap = cgrad(sunset, alpha = 0.4),
        #         linewidth = 3, colorrange
        #     )
        #     c = Colorbar(
        #         f[1, 2]; label = "Channel depth (μm)", colorrange,
        #         colormap = sunset
        #     )
        #     # rowsize!(f.layout, 1, Relative(0.8))
        #     mkpath(joinpath(plotpath, "$(_params[:sessionid])"))
        #     wsave(plotfile, f)
        #     @info "Saved plot to `$plotfile`"
        # end

        begin # * Fano factor calculations
            unitdepths = produce_unitdepths(session)
            unitdepths[!, :stimulus] .= stimulus

            @info "Calculating spike fano factors"
            spiketimes = AN.getspiketimes(session, structure)
            units = AN.getunitmetrics(session)

            Is = AN.stimulusepochs(session, stimulus).interval
            _, epoch = findmax(IntervalSets.width, Is) # Longest epoch
            I = Is[epoch]

            units = units[units.ecephys_unit_id .∈ [keys(spiketimes)], :]
            unitdepths = unitdepths[unitdepths.ecephys_unit_id .∈ [units.id], :]

            ffactor = fill(NaN, size(unitdepths, 1)) |> Vector{Any}
            unitdepths.fano_factor = ffactor

            τs = range(log10(1), log10(10000), length = 200) .|> exp10 # milliseconds
            for u in 1:size(unitdepths, 1)
                unitid = unitdepths[u, :].ecephys_unit_id
                spikes = spiketimes[unitid]
                spikes = filter(∈(I), spikes) # Select from stimulus
                if length(spikes) > 10 # * At least 10 spikes
                    unitdepths[u, :].fano_factor = fano_factor(spikes .* 1000, τs) # To milliseconds
                end
            end
        end

        # * Format data for saving
        streamlinedepths = AN.getchanneldepths(session, LFP; method = :streamlines)
        layerinfo = _layerinfo(session, channels)

        D = Dict(DimensionalData.metadata(LFP))
        @pack! D = channels, streamlinedepths, layerinfo
        mad = rebuild(mad; metadata = D)
        outD = Dict(
            "S" => S .|> Float32,
            "mad" => mad .|> Float32,
            "coeff" => coeff .|> Float32,
            "coeffs" => coeffs .|> Float32,
            "knees" => knees .|> Float32,  # crossover lag (s) per channel, unconstrained 2-comp fit
            "chi" => chi .|> Float32,      # spectral MAPPLE exponent per channel
            "unitdepths" => unitdepths,
            # "plotfiles" => relpath.([plotfile], [projectdir()])
        )
        tagsave(outfile, outD)

        @info "Data saved to `$outfile`"

        GC.safepoint()
        GC.gc()
    catch e
        GC.safepoint()
        GC.gc()
        @warn e
        tagsave(outfile, Dict("error" => sprint(showerror, e)))
    end
    # @info "Finished calculations"
    GC.safepoint()
    GC.gc()
    return outfile
end

# Vendored from SpatiotemporalMotifs (all unexported there) so the calc-quality machinery is SM-free.
const connector = "&"
const structures = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
const layers = ["1", "2/3", "4", "5", "6"]
const PTHR = 1.0e-2
const hierarchy_scores = Dict(
    "VISp" => -0.357, "VISl" => -0.093, "VISrl" => -0.059,
    "VISal" => 0.152, "VISpm" => 0.327, "VISam" => 0.441
)  # anatomical hierarchy, Siegle 2021
THETA() = (3, 10)   # SM used a preference-overridable getpref; WRExperiment just needs the defaults
GAMMA() = (30, 100)
CLUSTER() = get(ENV, "SM_CLUSTER", "false") == "true"   # are we running on the cluster?
const DEFAULT_SESSION_ID = 1140102579

function commondepths(depths)
    # A common range of 20 normalized depths approximating the collection (SM Utils.commondepths;
    # ponytail: dropped SM's dead `N`/filter lines --- the range only depends on the median bounds).
    lo = ceil(median(minimum.(depths)), sigdigits = 2)
    hi = floor(median(maximum.(depths)), sigdigits = 2)
    return range(lo, hi, length = 20)
end

function parselayernum(layername) # SM Utils.parselayernum: leading digits, 0 if none, merge layers 2 and 3
    m = match(r"\d+", layername)
    m = m === nothing ? 0 : parse(Int, m.match)
    return m > 2 ? m - 1 : m
end

# Trimmed replacement for SpatiotemporalMotifs' bulk-import `@preamble` macro: only the packages that are
# actually WRExperiment deps and used by the scripts (dropped the 7 unused heavy pkgs + SM internals the
# original pulled in). Expands in the calling script's scope, like the original.
macro preamble()
    return quote
        using Suppressor
        import AllenNeuropixelsBase as AN
        using Statistics, StatsBase, Random, LinearAlgebra, Unitful, FileIO
        using CairoMakie, DataFrames, DimensionalData, DrWatson, Fathom, Peaks
        using IntervalSets, Distributed, JLD2, HypothesisTests, MultipleTesting
        using ProgressLogging, TimeseriesTools
        using WRExperiment
    end
end
has_keys(D, required_keys) = all(haskey.([D], required_keys))

# bootstrapmedian is reused from TimeseriesTools (BCa CI via its BootstrapExt, active since we load Bootstrap);
# re-exported from WRExperiment. Returns (; average, confint=(; lower, upper)) --- destructures as μ,(σl,σh).

# Hierarchy correlation (Kendall τ vs anatomical hierarchy), vendored from SpatiotemporalMotifs. Used by
# collect_calculations (diffusion_hierarchical) and madev. Bootstrap.jl → BCa CIs; permutation p-values.
function hierarchicalkendall(
        x::AbstractVector{<:Real}, y::AbstractDimArray,
        mode::Symbol = :group; kwargs...
    )
    hasdim(y, Depth) || throw(ArgumentError("Argument 2 should have a Depth dimension"))
    hasdim(y, SessionID) || throw(ArgumentError("Argument 2 should have a SessionID dimension"))
    hasdim(y, Structure) || throw(ArgumentError("Argument 2 should have a Structure dimension"))
    return hierarchicalkendall(x, y, Val(mode); kwargs...)
end
function pairedkendall(xy)
    x = first.(xy)
    y = last.(xy)
    notnan = .!(isnan.(y) .| isnan.(x))
    return corkendall(x[notnan], y[notnan])
end
function _hierarchicalkendall(xx, yy; N = 10000, confint = 0.95)
    b = Bootstrap.bootstrap(pairedkendall, collect(zip(xx, yy)), Bootstrap.BalancedSampling(N))
    μ, σ... = only(Bootstrap.confint(b, Bootstrap.BCaConfInt(confint)))
    nsesh = hasdim(yy, SessionID) ? size(yy, SessionID) : 1
    μsur = map(1:N) do _
        idxs = stack(randperm(size(yy, Structure)) for _ in 1:nsesh)
        idxs = yy isa AbstractMatrix ? idxs' : idxs[:]
        ys = view(yy, idxs)   # a different shuffle per session
        @assert size(xx) == size(ys)
        pairedkendall(collect(zip(xx, ys)))
    end
    𝑝 = mean(abs.(μ) .< abs.(μsur))   # permutation test
    return μ, σ, 𝑝
end
function hierarchicalkendall(x::AbstractVector{<:Real}, y::AbstractDimArray, ::Val{:group}; kwargs...)
    y = permutedims(y, (Depth, SessionID, Structure))
    xx = repeat(x', size(y, SessionID), 1)
    ms = asyncmap(eachslice(y, dims = Depth)) do yy
        _hierarchicalkendall(xx, yy; kwargs...)
    end
    𝑝 = last.(ms)
    𝑝 = set(𝑝, adjust(collect(𝑝), BenjaminiHochberg()))
    return first.(ms), getindex.(ms, 2), 𝑝
end
function hierarchicalkendall(x::AbstractVector{<:Real}, y::AbstractDimArray, ::Val{:individual})
    y = permutedims(y, (Depth, SessionID, Structure))
    ms = asyncmap(eachslice(y, dims = Depth)) do yy
        mnms = map(eachslice(yy, dims = SessionID)) do yyy
            μ = corkendall(x, yyy)
            s = begin
                idxs = randperm(length(yyy))
                corkendall(x, yyy[idxs])
            end
            return μ, s
        end
        μ = first.(mnms)
        s = last.(mnms)
        σ = (percentile(μ, 25), percentile(μ, 75))
        𝑝 = SignedRankTest(convert(Vector{Float64}, μ), convert(Vector{Float64}, s)) |> pvalue
        μ = median(μ)
        return μ, σ, 𝑝
    end
    𝑝 = last.(ms)
    𝑝 = set(𝑝, adjust(collect(𝑝), BenjaminiHochberg()))
    return first.(ms), getindex.(ms, 2), 𝑝
end
function mediankendallpvalue(x::AbstractVector, Y::AbstractMatrix; N = 10000)
    τ = map(eachslice(collect(Y), dims = 2)) do y
        notnan = .!isnan.(y)
        corkendall(x[notnan], y[notnan])
    end
    τsur = asyncmap(1:N) do _
        idxs = randperm(length(x))
        map(eachslice(collect(Y), dims = 2)) do y
            notnan = .!isnan.(y)
            corkendall(x[idxs][notnan], y[notnan])
        end
    end
    τsur = vcat(τsur...)
    tst = MannWhitneyUTest(τ[:], τsur[:])
    𝑝 = tst |> pvalue
    mtau, ci = bootstrapmedian(collect(τ .+ randn(length(τ)) .* eps()))
    return (; τ = mtau, ci, 𝑝, U = tst.U, n = length(x), N)
end

val_to_string(v) = v isa Regex ? v.pattern : string(v)
const allowedtypes = (Real, String, Regex, Symbol, TimeType, Vector, Tuple)

function savepath(D::Union{Dict, NamedTuple}, ext::String = "", args...)
    filename = savename(D, ext; connector, val_to_string, allowedtypes)
    return joinpath(args..., filename)
end
function savepath(prefix::String, D::Union{Dict, NamedTuple}, ext::String = "", args...)
    filename = savename(prefix, D, ext; connector, val_to_string, allowedtypes)
    return joinpath(args..., filename)
end
savepath(prefix::String, aargs...) = (args...) -> savepath(prefix, args..., aargs...)

"""
Check the quality of a calculations directory e.g. `data/madev`. Copied from SpatiotemporalMotifs
(part of dropping the SM dep).
"""
function calcquality(
        dirname;
        suffix = "jld2",
        connector = connector,
        require = true,
        requirekeys = nothing # completeness keys for non-madev outputs (e.g. surrogates)
    )
    if isempty(readdir(dirname))
        return []
    end
    if !(require isa Bool)
        _require = require
        require = true
    else
        _require = []
    end

    @info "Checking quality of calculations in `$dirname`"
    files = readdir(dirname)
    threadlog = Threads.Atomic{Int}(0)
    threadmax = length(files)
    lk = Threads.ReentrantLock()
    ps = []
    @withprogress name = "Checking quality" begin
        Threads.@threads for f in files
            f = joinpath(dirname, f)
            _, parameters, _suffix = parse_savename(f; connector)
            if _suffix == suffix
                try
                    if require
                        canload = jldopen(f, "r"; iotype = IOStream) do fl
                            # fl["performance_metrics"] # Can load
                            complete = isnothing(requirekeys) ? has_calc_keys(fl) :
                                has_calc_keys(fl, requirekeys)
                            return complete && has_keys(fl, string.(_require))
                        end
                    else
                        canload = true
                    end
                    if !canload
                        continue
                    end
                catch e
                    @warn e
                    continue
                end
                lock(lk) do
                    push!(ps, parameters) # Only add to list of good files if file exists and has no error
                    if require
                        if Threads.threadid() ∈ 1:2
                            Threads.atomic_add!(threadlog, Threads.nthreads() ÷ 2)
                            @logprogress threadlog[] / threadmax
                        end
                    end
                end
            end
        end
    end
    isempty(ps) && return [] # no complete files yet: same as an empty directory
    ks = keys.(ps) |> collect
    vs = values.(ps) .|> collect
    vs = stack(vs)
    dims = unique(stack(ks))
    uvs = unique.(eachrow(vs))
    si = findfirst(dims .== ["stimulus"])
    if !isnothing(si)
        vs = replace(vs, "Natural_Images" => r"Natural_Images")
        uvs[si] = replace(uvs[si], "Natural_Images" => r"Natural_Images")
    end

    function map2dims(d)
        if d == "sessionid"
            return SessionID
        elseif d == "trial"
            return Trial
        elseif d == "structure"
            return Structure
        else
            return Dim{Symbol(d)}
        end
    end
    ddims = Tuple([map2dims(d)(s) for (s, d) in zip(uvs, dims)])

    Q = falses(length.(uvs)...)
    Q = ToolsArray(Q, ddims)
    for v in eachcol(vs)
        ds = [map2dims(d)(At(s)) for (s, d) in zip(v, dims)]
        Q[ds...] = true
    end
    if any(isa.(DimensionalData.dims(Q), (Structure,))) &&
            all(lookup(Q, Structure) .∈ [structures])
        s = structures .∈ [lookup(Q, Structure)]
        s = structures[s]
        Q = Q[Structure = At(s)] # * Sort to global structures order
    end
    @info "Mean quality: $(mean(Q))"
    return Q
end

function has_calc_keys(
        D,
        required_keys = [
            "S",
            "mad",
            "coeff",
            "coeffs",
            "knees",
            "chi",
            "unitdepths",
            # "plotfiles",
        ]
    )
    return (haskey(D, "error") && contains(D["error"], "Region error")) ||
        (
        has_keys(D, string.(required_keys)) #&& all(isfile.(projectdir.(D["plotfiles"])))
    )
end

function plotspectrum!(
        ax, s::AbstractToolsArray;
        textposition = (14, exp10(-2.9)),
        annotations = [:peaks, :mapple],
        color = cucumber, label = nothing,
        mappleargs = (;), domedian = false
    )
    if domedian
        μ = median(s, dims = (SessionID, :layer))
        μ = dropdims(μ, dims = (SessionID, :layer)) |> ustripall
        σl = map(x -> quantile(x[:], 0.25), eachslice(s, dims = 𝑓)) |> ustripall
        σh = map(x -> quantile(x[:], 0.75), eachslice(s, dims = 𝑓)) |> ustripall

        p = lines!(ax, freqs(μ), collect(μ); color = (color, 0.8), label)
        band!(
            ax, freqs(μ), σl, σh; color = (color, 0.32),
            label
        )
    else
        μ = mean(s, dims = (SessionID, :layer))
        μ = dropdims(μ, dims = (SessionID, :layer)) |> ustripall
        σ = std(s, dims = (SessionID, :layer)) ./ 2
        σ = dropdims(σ, dims = (SessionID, :layer)) |> ustripall
        p = lines!(ax, freqs(μ), collect(μ); color = (color, 0.8), label)
        band!(
            ax, freqs(μ), collect.([max.(μ - σ, eps()), μ + σ])...;
            color = (color, 0.32),
            label
        )
    end

    # * Find peaks
    if :peaks in annotations
        pks, proms = findpeaks(μ, 2; N = 2)
        scatter!(
            ax, collect(freqs(pks)), collect(pks .* 1.25), color = :black,
            markersize = 10, marker = :dtriangle
        )
        text!(
            ax, collect(freqs(pks)), collect(pks);
            text = string.(round.(collect(freqs(pks)), sigdigits = 2)) .* [" Hz"],
            align = (:center, :bottom), color = :black, rotation = 0,
            fontsize = 16,
            offset = (0, 5)
        )
    end

    # * MAPPLE fit
    ff, ps = WRExperiment.mapple_fit(μ)
    lines!(
        ax, freqs(μ), ff.(freqs(μ)); color = (color, 0.5), linestyle = :dash,
        linewidth = 5, mappleargs...
    )
    if :mapple in annotations
        text!(
            ax, textposition...; text = L"𝛂 = $(round(ps[:χ], sigdigits=3))",
            fontsize = 16, align = (:right, :center)
        )
    end
    return p, ps[:χ]
end
