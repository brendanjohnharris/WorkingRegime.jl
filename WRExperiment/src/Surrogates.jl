# Surrogate controls for the quoted statistics. The default null is FT (phase randomisation): a
# linear Gaussian process with the data's power spectrum, so significance means "beyond a linear
# Gaussian process". IAAFT (Schreiber & Schmitz 1996) additionally preserves the amplitude
# distribution and is kept for the stricter question of whether an effect is dynamical.

import TimeseriesSurrogates: IAAFT, RandomFourier, Surrogate, surrogenerator
import MoreMaps: Chart, Threaded, LogLogger

"""
    surrogate_null(stat, x, method = RandomFourier(); n = 100, seed = 0, chart = ...)

Evaluate `stat` on `x` and on `n` surrogates of it: returns `(; s0, s)`, the data statistic and the
null statistics. Columns of a matrix are surrogated independently and `stat` receives the whole
draw, so pooled and channel-aggregated statistics see per-channel surrogates. `stat` must replicate
the full quoted pipeline (including any aggregation) and should be deterministic; draw `k` uses
`Xoshiro(seed + k)`, so results are reproducible and threading-invariant.
"""
function surrogate_null(
        stat, x::AbstractVecOrMat{<:Real}, method::Surrogate = RandomFourier();
        n = 100, seed = 0, chart = Chart(LogLogger(), Threaded())
    )
    s = map(chart, 1:n) do k
        stat(surrogatedraw(x, method, Xoshiro(seed + k)))
    end
    return (; s0 = stat(x), s)
end

surrogatedraw(x::AbstractVector, method, rng) = surrogenerator(collect(x), method, rng)()
function surrogatedraw(X::AbstractMatrix, method, rng)
    return mapreduce(c -> surrogenerator(collect(c), method, rng)(), hcat, eachcol(X))
end

"""
    lfp_surrogate_stats(Y, dt; band = MAD_BAND)

The two quoted statistics of one (time × channel) LFP block, per channel, computed together so one
set of surrogate draws serves both:

  - `kurt`: excess kurtosis of the single-sample increments. Zero for any Gaussian process, hence
    exactly zero for the FT null by construction, so the data value IS the effect size. Preferred
    over a stable-law index, which is bounded at α = 2 and therefore compressed and one-sided for
    near-Gaussian increments (these have Hill exponents of 6--10, i.e. no power-law tail at all).
  - `a`: the diffusion (MAD) exponent. A scaling exponent, so the increment distribution's SHAPE
    enters its prefactor rather than its slope. Against the FT null it has a closed form,
    `a − a_null ≈ ζ(1) − ζ(2)/2`, the first-order intermittency coefficient, zero for any
    monofractal process.

Fit failures degrade to NaN. Both are per channel so layers can be selected at collect time.
"""
function lfp_surrogate_stats(Y::AbstractMatrix, dt; band = MAD_BAND)
    taus = madev_taus(dt)
    lags = round.(Int, taus ./ dt)
    curves = mapreduce(c -> madev(c, lags), hcat, eachcol(Y))
    trydiff(c) = try
        diffusion_fit(Timeseries(c, taus); band)
    catch e
        @warn "Failed diffusion fit" e maxlog = 5
        NaN
    end
    return (;
        kurt = map(c -> kurtosis(diff(c)), eachcol(Y)),   # StatsBase: excess kurtosis
        a = map(trydiff, eachcol(curves)),
    )
end

"""
    send_surrogates(sessionid, stimulus, structure; n = 20, method = :ft, outpath)

Surrogate nulls for the quoted LFP statistics of one condition, loading the LFP exactly as
[`send_madev`](@ref) does. `method` defaults to `:ft` (`RandomFourier`: spectrum preserved, marginal
Gaussian), the null for "heavier-tailed / more anomalous than a linear Gaussian process"; `:iaaft`
additionally preserves the amplitude distribution and answers the stricter question of whether the
effect is dynamical rather than inherited from the marginal. Each method writes to its own `outpath`
(`rootdatadir("surrogates_ft")` and `rootdatadir("surrogates")`). Saves the data statistics (`s0`), the `n`
null statistics (`s`, one [`lfp_surrogate_stats`](@ref) tuple per draw), and per-channel layer labels
for the collect step. Both statistics are one-sided LARGER than the null (`tail = :right`).
"""
function send_surrogates(
        sessionid, stimulus, structure;
        n = 20, method = :ft,
        outpath = rootdatadir(method === :iaaft ? "surrogates" : "surrogates_$(method)")
    )
    surr = method === :iaaft ? IAAFT() :
        method === :ft ? RandomFourier() :
        throw(ArgumentError("method must be :iaaft or :ft"))
    params = (; sessionid, epoch = :longest, band = (1.0e-3, 1.0e-2), pass = (1, 625)) # send_madev's LFP
    _params = (; params..., stimulus, structure)
    if stimulus == r"Natural_Images"
        _params = (; _params..., epoch = (:longest, :active))
    end
    outfile = savepath(
        Dict("sessionid" => sessionid, "stimulus" => stimulus, "structure" => structure),
        "jld2", outpath
    )
    @info "Surrogates for $(stimulus) LFP in $(structure), session $(sessionid)"
    session = AN.Session(sessionid)
    try
        probestructures = unique(vcat(values(AN.getprobestructures(session))...))
        if structure ∉ probestructures
            str = "Region error: structure $(structure) not found in $(sessionid)"
            @warn str
            tagsave(outfile, Dict("error" => str))
            return outfile
        end
        LFP = AN.formatlfp(session; tol = 3, _params...)
        lfp = ustripall(LFP)
        dt = TimeseriesTools.samplingperiod(lfp)
        X = Float64.(parent(lfp))
        r = surrogate_null(Y -> lfp_surrogate_stats(Y, dt), X, surr; n)
        channels = collect(lookup(LFP, Chan))
        # `layers`/`layernums` are in COLUMN order, unlike `_layerinfo`, which sorts by depth; the
        # collect and plotting steps index the statistics by column, and the plots project has no
        # WRExperiment dependency with which to reparse the labels.
        layers = string.(last(AN.getchannellayers(session, channels)))
        outD = Dict(
            "s0" => r.s0, "s" => r.s, "n" => n, "dt" => dt,
            "channels" => channels,
            "layers" => layers,
            "layernums" => parselayernum.(layers),
            "depths" => AN.getchanneldepths(session, LFP; method = :probe),
            "streamlinedepths" => AN.getchanneldepths(session, LFP; method = :streamlines),
        )
        tagsave(outfile, outD)
        @info "Data saved to `$outfile`"
    catch e
        @warn e
        tagsave(outfile, Dict("error" => sprint(showerror, e)))
    end
    GC.safepoint()
    GC.gc()
    return outfile
end
