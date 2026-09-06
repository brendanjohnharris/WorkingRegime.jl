#! /bin/bash
#=
exec julia +1.12 "${BASH_SOURCE[0]}" "$@"
=#
# Pooled distribution of L2/3 LFP increments, cached for the Figure 1 panel. Each channel's
# increments are standardised by their own standard deviation before pooling, so channels and
# sessions with different amplitudes contribute on equal terms and the pooled curve is directly
# comparable to a standard Gaussian.
#
# One FT surrogate per channel is histogrammed alongside, giving the null EMPIRICALLY as well as
# analytically: phase randomisation leaves Gaussian increments, so the surrogate curve should lie on
# the Gaussian, and the panel then shows exactly the comparison the main text quotes.
#
# Serial: Allen data access goes through PythonCall, which is not thread-safe. ~2.5 min per session,
# so the full 69-session run takes roughly three hours; set N_SESSIONS to an integer for a fast draft.
#
# Aggregation follows `collect_surrogates.jl` so that the panel annotation and the main text quote the
# same statistic: per session, the median over that session's L2/3 channels, then the median across
# sessions (`kurtosis_session`). The flat per-channel vector (`kurtosis`) is kept for the shape
# diagnostics, but pooling channels weights each session by its channel count and reads higher.
#
# DUPLICATE: the `increment_histograms` block of `WRExperiment/scripts/collect_calculations.jl` writes
# this same cache through `produce_or_load`, which fires only when the file is absent. Keep the two in
# sync (better: consolidate them) or a deleted cache silently reverts the sample and the aggregation.
using DrWatson
@quickactivate "WRExperiment"
using WRExperiment
using TimeseriesTools
using Statistics, StatsBase, Random
using FileIO, JLD2
import AllenNeuropixelsBase as AN
import TimeseriesSurrogates: RandomFourier, surrogenerator

const N_SESSIONS = nothing                         # nothing = every QC-passing session; an integer takes a fast draft subset
const STRUCTURE = "VISp"
const STIMULUS = "spontaneous"
const EDGES = range(-15, 15, length = 601)         # standard deviations; 0.05 SD bins
const OUTFILE = DrWatson.datadir("increment_histograms.jld2")

sessions = load(DrWatson.datadir("session_table.jld2"), "session_table").ecephys_session_id
isnothing(N_SESSIONS) || (sessions = sessions[1:min(N_SESSIONS, length(sessions))])

counts, counts_surr = zeros(Int, length(EDGES) - 1), zeros(Int, length(EDGES) - 1)
kurt, kurt_surr = Float64[], Float64[]                 # per channel, pooled across sessions
kurt_sess, kurt_sess_surr = Float64[], Float64[]       # per session: median over that session's L2/3 channels
nchan = 0

for (i, sessionid) in enumerate(sessions)
    @info "[$i/$(length(sessions))] session $sessionid"
    try
        session = AN.Session(sessionid)
        LFP = AN.formatlfp(
            session; tol = 3, sessionid, epoch = :longest, band = (1.0e-3, 1.0e-2),
            pass = (1, 625), stimulus = STIMULUS, structure = STRUCTURE
        )
        X = Float64.(parent(ustripall(LFP)))
        lnum = parselayernum.(string.(last(AN.getchannellayers(session, collect(lookup(LFP, AN.Chan))))))
        k_this, ks_this = Float64[], Float64[]     # this session's per-channel values
        for j in findall(lnum .== 2)               # L2/3
            d = diff(@view X[:, j])
            s = diff(surrogenerator(collect(@view X[:, j]), RandomFourier(), Xoshiro(j))())
            counts .+= fit(Histogram, d ./ std(d), EDGES).weights
            counts_surr .+= fit(Histogram, s ./ std(s), EDGES).weights
            push!(k_this, kurtosis(d))             # excess kurtosis, per channel, as the sweep computes it
            push!(ks_this, kurtosis(s))
            global nchan += 1
        end
        append!(kurt, k_this)
        append!(kurt_surr, ks_this)
        # Session-level value: median over this session's L2/3 channels, matching `collect_surrogates.jl`.
        kk, kks = filter(!isnan, k_this), filter(!isnan, ks_this)
        isempty(kk) || push!(kurt_sess, median(kk))
        isempty(kks) || push!(kurt_sess_surr, median(kks))
    catch e
        @warn "Skipping $sessionid" e
    end
end

density(c) = c ./ (sum(c) * step(EDGES))
wsave(
    OUTFILE, Dict(
        "edges" => collect(EDGES), "centres" => collect(EDGES)[1:(end - 1)] .+ step(EDGES) / 2,
        "density" => density(counts), "density_surrogate" => density(counts_surr),
        "counts" => counts, "counts_surrogate" => counts_surr,
        "kurtosis" => kurt, "kurtosis_surrogate" => kurt_surr,
        "kurtosis_session" => kurt_sess, "kurtosis_surrogate_session" => kurt_sess_surr,
        "nchannels" => nchan, "sessions" => sessions,
        "structure" => STRUCTURE, "stimulus" => STIMULUS
    )
)
@info "Saved $OUTFILE" nchan length(kurt_sess) median(kurt) median(kurt_surr) median(kurt_sess) median(kurt_sess_surr)
