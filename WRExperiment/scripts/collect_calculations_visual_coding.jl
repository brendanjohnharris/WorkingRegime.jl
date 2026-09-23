#! /bin/bash
#=
exec julia +1.12 -t auto "${BASH_SOURCE[0]}" "$@"
=#
# Everything Figure S1 draws, assembled from `run_calculations_visual_coding.jl`'s per-block output
# into one file that `scripts/FigS1_visual_coding.jl` reads. Same division of labour as Visual
# Behaviour's `collect_calculations.jl`: the slow work (LFP passes, MAPPLE fits) happens here and is
# cached by `produce_or_load`, and the plot script does nothing but arithmetic on the result.
#
# Two deliberate departures from the Visual Behaviour collector:
#
#   1. No interpolation onto `commondepths`. Visual Coding stores every fourth channel, so an area
#      yields ~21 channels against ~90, and the `maximum(streamlinedepths) > 0.9` coverage filter
#      that Visual Behaviour applies would drop most blocks. Channels are kept as they are and
#      selected by layer, which is all the panels need.
#   2. Ragged area coverage is expected rather than an error. No functional-connectivity session
#      records all six visual areas, so the hierarchy correlation is taken per session across
#      whatever areas that session has and summarised afterwards. That correlation is NOT computed
#      here: the `(session x area)` exponent matrices are stored as they are, and the figure runs
#      `WRExperiment.sessionkendall` on them, the same estimator Figure 4c uses.
#
# The variability exponent comes from WRExperiment's own `variability_exponent` (src/Variability.jl),
# not a local MAPPLE call, so this cohort is fit by the same estimator the rest of the package uses.
using DrWatson
@quickactivate "WRExperiment"
using WRExperiment
using TimeseriesTools   # re-exports TimeseriesBase, which src/Variability.jl needs
using Optim
using ForwardDiff          # with Optim, activates TimeseriesTools' OptimExt (MAPPLE `fit!`)
using DimensionalData
using Statistics
using StatsBase
using Random
using MoreMaps
using FileIO
using JLD2
import AllenNeuropixelsBase as AN
import AllenNeuropixelsBase: Depth
import StatsBase: corkendall
import TimeseriesSurrogates: RandomFourier, surrogenerator
import TimeseriesTools: FFTW
import LinearAlgebra
FFTW.set_num_threads(1)   # see fftw-threads-segfault
# `variabilitymatrix` threads over the fits, so each fit's linear algebra must stay single-threaded:
# otherwise 20 Julia threads each spawn BLAS threads and oversubscribe the machine, and BLAS's
# threaded reductions make the optimiser's path irreproducible.
LinearAlgebra.BLAS.set_num_threads(1)

const COHORT = isempty(ARGS) ? :functional_connectivity :
    (
        lowercase(ARGS[1]) in ("bo", "brain_observatory") ? :brain_observatory :
        :functional_connectivity
    )
const TAG = COHORT === :brain_observatory ? "bo" : "fc"
const CALCDIR = visual_coding_calcdir(COHORT)
const STIM = VISUAL_CODING_STIMULUS

# Layer integers as `parselayernum` assigns them: 1 = L1, 2 = L2/3, 3 = L4, 4 = L5, 5 = L6.
const LAYERNUMS = [1, 2, 3, 4, 5]
const LAYERNAMES = ["1", "2/3", "4", "5", "6"]
const TAULAYERS = [2, 3, 4, 5]      # L1 carries too few channels per block for a hierarchy read
const MIN_CHANNELS = 2              # per block per layer, for a layer median
const MIN_FANO_CHANNELS = 5         # the Fano curve is noisier; needs more units behind it
const EDGES = range(-15, 15, length = 601)     # increment histogram, in SDs; 0.05 SD bins

hvec = [hierarchy_scores[s] for s in structures]

# ---------------------------------------------------------------------------- per-block curves
"""
Load every completed block as `(sessionid, structure) => (; mad, S, coeffs, chi, layers, unitdepths)`,
with `layers` aligned to the data columns by [`channellayers`](@ref).
"""
function loadblocks(path)
    Q = calcquality(path)
    D = Dict{Tuple{Int, String}, Any}()
    for structure in structures, sessionid in visual_coding_sessions(COHORT)
        ok = try
            Q[stimulus = At(STIM), Structure = At(structure), SessionID = At(sessionid)]
        catch
            false
        end
        ok == 0 && continue
        f = savepath(
            Dict(
                "sessionid" => sessionid, "stimulus" => STIM,
                "structure" => structure
            ), "jld2", path
        )
        isfile(f) || continue
        d = jldopen(f, "r") do fl
            haskey(fl, "error") && return nothing
            # Units come off here, not downstream: the spectrum carries V²·s, and the figure
            # script runs in the root project, which does not depend on Unitful.
            mad = ustripall(fl["mad"])
            (;
                mad, S = ustripall(fl["S"]), coeffs = fl["coeffs"], chi = fl["chi"],
                layers = channellayers(mad), unitdepths = fl["unitdepths"],
            )
        end
        isnothing(d) || (D[(sessionid, structure)] = d)
    end
    return D
end

"""
Median curve over one layer's channels within each block, then the median across blocks. Restricted
to one `structure` when given; pooled over all of them otherwise.
"""
function layercurve(blocks, field, layernum; structure = nothing)
    sel_blocks = isnothing(structure) ? collect(values(blocks)) :
        [v for (k, v) in blocks if last(k) == structure]
    cs = map(sel_blocks) do d
        sel = findall(==(layernum), d.layers)
        length(sel) < MIN_CHANNELS && return nothing
        vec(median(Float64.(parent(getfield(d, field))[:, sel]); dims = 2))
    end
    cs = filter(!isnothing, cs)
    isempty(cs) && return nothing
    # `reduce(hcat, [v])` on a single vector stays a vector, which would silently skip the median
    M = length(cs) == 1 ? reshape(cs[1], :, 1) : reduce(hcat, cs)
    return vec(median(M; dims = 2))
end

"Median Fano curve over one layer's units within each block, then the median across blocks."
function fanolayercurve(blocks, layernum, ftaus; structure = nothing)
    sel_blocks = isnothing(structure) ? collect(values(blocks)) :
        [v for (k, v) in blocks if last(k) == structure]
    cs = map(sel_blocks) do d
        M = unitfano(d, layernum, ftaus)
        isnothing(M) && return nothing
        [(w = filter(!isnan, view(M, k, :)); isempty(w) ? NaN : median(w)) for k in axes(M, 1)]
    end
    cs = filter(!isnothing, cs)
    isempty(cs) && return nothing
    M = length(cs) == 1 ? reshape(cs[1], :, 1) : reduce(hcat, cs)
    return [(w = filter(!isnan, view(M, k, :)); isempty(w) ? NaN : median(w)) for k in axes(M, 1)]
end

"""
Fano curves (bin widths × units) for the units of one block assigned to `layernum`, or `nothing`
when too few carry a curve. Unit layers come from the block's own channel layer map, read at each
unit's probe depth --- the same construction `collect_calculations.jl` uses for Visual Behaviour.
"""
function unitfano(d, layernum, ftaus)
    ud = d.unitdepths
    isempty(ud) && return nothing
    depths = collect(lookup(d.mad, Depth))
    lay = d.layers
    sel = Int[]
    for (u, pd) in enumerate(ud.probedepth)
        j = argmin(abs.(depths .- pd))
        lay[j] == layernum && ud[u, :].fano_factor isa AbstractVector && push!(sel, u)
    end
    length(sel) < MIN_FANO_CHANNELS && return nothing
    return reduce(hcat, [Float64.(collect(ud[u, :].fano_factor)) for u in sel])
end

# ---------------------------------------------------------------------------- exponents
"""
Layer-median exponent per (session, structure), as a matrix with NaN where a block is absent or has
too few channels in that layer. `field` is `:a` (diffusion, `coeffs`) or `:b` (spectral, `-chi`).
"""
function exponentmatrix(blocks, sessions, field, layernum)
    Y = fill(NaN, length(sessions), length(structures))
    for (i, s) in enumerate(sessions), (j, st) in enumerate(structures)
        haskey(blocks, (s, st)) || continue
        d = blocks[(s, st)]
        sel = findall(==(layernum), d.layers)
        length(sel) < MIN_CHANNELS && continue
        v = field === :a ? Float64.(parent(d.coeffs)[sel]) : .-Float64.(parent(d.chi)[sel])
        v = filter(isfinite, v)
        isempty(v) || (Y[i, j] = median(v))
    end
    return Y
end

"""
Variability exponent per (session, structure) at one layer: the package's [`variability_exponent`](@ref)
fit to that block's unit-median Fano curve. Fit the aggregated curve, never per unit --- per-unit
curves have no SNR for a free-knot fit (see the estimator's docstring).
"""
function variabilitymatrix(blocks, sessions, layernum, ftaus)
    # Threaded: this is the only expensive step here (a multistart MAPPLE fit per session and area,
    # against plain medians everywhere else), and it dominates the whole script. `Iterators.product`
    # flattens the two axes into one map, so the result comes back shaped (session, structure).
    # `variability_exponent` reseeds the task-local RNG on entry, so its multistart draws are the
    # same whichever task runs the fit and the matrix matches the serial one exactly.
    Y = map(
        Chart(LogLogger(), Threaded()),
        collect(Iterators.product(eachindex(sessions), eachindex(structures)))
    ) do (i, j)
        s, st = sessions[i], structures[j]
        haskey(blocks, (s, st)) || return NaN
        M = unitfano(blocks[(s, st)], layernum, ftaus)
        isnothing(M) && return NaN
        med = [(w = filter(!isnan, view(M, k, :)); isempty(w) ? NaN : median(w)) for k in axes(M, 1)]
        count(!isnan, med) < length(med) ÷ 2 && return NaN
        return variability_exponent(Timeseries(med, collect(ftaus))).β
    end
    return Float64.(Y)
end

# ---------------------------------------------------------------------------- assemble
curves, _ = produce_or_load(
    Dict(), DrWatson.datadir();
    filename = savepath("visual_coding_$(TAG)")
) do _
    blocks = loadblocks(CALCDIR)
    isempty(blocks) && error("no completed blocks in $(CALCDIR); run run_calculations_visual_coding.jl")
    sessions = sort(unique(first.(keys(blocks))))
    @info "collected" cohort = COHORT blocks = length(blocks) sessions = length(sessions)

    ref = first(values(blocks))
    taus = collect(lookup(ref.mad, 1))
    fr = collect(lookup(ref.S, 1))
    ftaus = exp10.(range(log10(1), log10(10000), length = 200))   # ms; matches send_madev

    mad_layer = Dict(l => layercurve(blocks, :mad, l) for l in LAYERNUMS)
    psd_layer = Dict(l => layercurve(blocks, :S, l) for l in LAYERNUMS)
    fano_layer = Dict(l => fanolayercurve(blocks, l, ftaus) for l in LAYERNUMS)
    # VISp alone, for the figure's curve panels: pooling areas mixes their exponents, so a pooled
    # curve cannot be labelled with any one area's exponent.
    mad_visp = Dict(l => layercurve(blocks, :mad, l; structure = "VISp") for l in LAYERNUMS)
    psd_visp = Dict(l => layercurve(blocks, :S, l; structure = "VISp") for l in LAYERNUMS)
    fano_visp = Dict(l => fanolayercurve(blocks, l, ftaus; structure = "VISp") for l in LAYERNUMS)

    @info "fitting exponents by layer"
    A = Dict(l => exponentmatrix(blocks, sessions, :a, l) for l in LAYERNUMS)
    B = Dict(l => exponentmatrix(blocks, sessions, :b, l) for l in LAYERNUMS)
    C = Dict(l => variabilitymatrix(blocks, sessions, l, ftaus) for l in TAULAYERS)

    return Dict(
        "cohort" => string(COHORT), "sessions" => sessions, "structures" => structures,
        "layernums" => LAYERNUMS, "layernames" => LAYERNAMES, "taulayers" => TAULAYERS,
        "taus" => taus, "freqs" => fr, "fano_taus" => ftaus,
        "mad_layer" => mad_layer, "psd_layer" => psd_layer, "fano_layer" => fano_layer,
        "mad_visp" => mad_visp, "psd_visp" => psd_visp, "fano_visp" => fano_visp,
        "a" => A, "b" => B, "c" => C,
        "hierarchy" => hvec, "nblocks" => length(blocks)
    )
end

# ---------------------------------------------------------------------------- increment histogram
# A second LFP pass, cached separately so it is not repeated when the curves above are recomputed.
# Mirrors the Visual Behaviour version exactly: each channel's increments
# are standardised by their own SD before pooling, and one Fourier-transform surrogate per channel
# gives the Gaussian null empirically alongside the analytic one.
increments, _ = produce_or_load(
    Dict(), DrWatson.datadir();
    filename = savepath("visual_coding_increments_$(TAG)")
) do _
    counts, counts_surr = zeros(Int, length(EDGES) - 1), zeros(Int, length(EDGES) - 1)
    kurt, kurt_surr, kurt_sess, kurt_sess_surr = Float64[], Float64[], Float64[], Float64[]
    nchan = 0
    for (i, sessionid) in enumerate(visual_coding_sessions(COHORT))
        @info "[$i] increments, session $sessionid"
        try
            session = AN.Session(sessionid)
            LFP = formatlfp(
                session; epoch = :longest, band = (1.0e-3, 1.0e-2),
                pass = (1, 625), stimulus = STIM, structure = "VISp"
            )   # tolerance ladder
            X = Float64.(parent(ustripall(LFP)))
            lnum = parselayernum.(
                String.(
                    last(
                        AN.getchannellayers(
                            session,
                            collect(lookup(LFP, AN.Chan))
                        )
                    )
                )
            )
            length(lnum) == size(X, 2) || (@warn "channel/layer length mismatch, skipping"; continue)
            k_this, ks_this = Float64[], Float64[]
            for j in findall(==(2), lnum)          # L2/3
                d = diff(@view X[:, j])
                s = diff(surrogenerator(collect(@view X[:, j]), RandomFourier(), Xoshiro(j))())
                counts .+= fit(Histogram, d ./ std(d), EDGES).weights
                counts_surr .+= fit(Histogram, s ./ std(s), EDGES).weights
                push!(k_this, kurtosis(d))
                push!(ks_this, kurtosis(s))
                nchan += 1
            end
            append!(kurt, k_this); append!(kurt_surr, ks_this)
            kk, kks = filter(!isnan, k_this), filter(!isnan, ks_this)
            isempty(kk) || push!(kurt_sess, median(kk))
            isempty(kks) || push!(kurt_sess_surr, median(kks))
        catch e
            @warn "Skipping $sessionid" e
        end
    end
    density(c) = c ./ (sum(c) * step(EDGES))
    return Dict(
        "centres" => collect(EDGES)[1:(end - 1)] .+ step(EDGES) / 2,
        "density" => density(counts), "density_surrogate" => density(counts_surr),
        "counts" => counts, "counts_surrogate" => counts_surr,
        "kurtosis" => kurt, "kurtosis_surrogate" => kurt_surr,
        "kurtosis_session" => kurt_sess, "kurtosis_surrogate_session" => kurt_sess_surr,
        "nchannels" => nchan, "structure" => "VISp", "stimulus" => STIM
    )
end

@info "done" curves = DrWatson.datadir("visual_coding_$(TAG).jld2") increments = DrWatson.datadir("visual_coding_increments_$(TAG).jld2") nblocks = curves["nblocks"] nchannels = increments["nchannels"]
