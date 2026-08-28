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
# so the default subset takes about half an hour; it is a shape panel, and the quoted statistics come
# from the full 69-session sweep in `run_surrogates.jl` rather than from here.
using DrWatson
@quickactivate "WRExperiment"
using WRExperiment
using TimeseriesTools
using Statistics, StatsBase, Random
using FileIO, JLD2
import AllenNeuropixelsBase as AN
import TimeseriesSurrogates: RandomFourier, surrogenerator

const N_SESSIONS = 12                              # raise for a smoother tail; the cache re-runs
const STRUCTURE = "VISp"
const STIMULUS = "spontaneous"
const EDGES = range(-15, 15, length = 601)         # standard deviations; 0.05 SD bins
const OUTFILE = DrWatson.datadir("increment_histograms.jld2")

sessions = load(DrWatson.datadir("session_table.jld2"), "session_table").ecephys_session_id
sessions = sessions[1:min(N_SESSIONS, length(sessions))]

counts, counts_surr = zeros(Int, length(EDGES) - 1), zeros(Int, length(EDGES) - 1)
kurt, kurt_surr = Float64[], Float64[]
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
        for j in findall(lnum .== 2)               # L2/3
            d = diff(@view X[:, j])
            s = diff(surrogenerator(collect(@view X[:, j]), RandomFourier(), Xoshiro(j))())
            counts .+= fit(Histogram, d ./ std(d), EDGES).weights
            counts_surr .+= fit(Histogram, s ./ std(s), EDGES).weights
            push!(kurt, kurtosis(d))               # excess kurtosis, per channel, as the sweep computes it
            push!(kurt_surr, kurtosis(s))
            global nchan += 1
        end
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
        "nchannels" => nchan, "sessions" => sessions,
        "structure" => STRUCTURE, "stimulus" => STIMULUS
    )
)
@info "Saved $OUTFILE" nchan median(kurt) median(kurt_surr)
