#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.13 --handle-signals=yes -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Theta-peak variation --- how the SLOW LFP peak moves with the adaptation conductance Δg_K.
#
# Companion to peak_variation.jl. There the gamma peak is set by the PING loop and steered by δ; here
# the slow peak is set by the E cells' spike-triggered K⁺ adaptation --- each spike does
# `gK ← gK + Δg_K` with `gK` decaying over `τ_K = 40 ms` and reversing at `V_K = -85 mV`
# (`Dewdrop.FNSNeuron`), i.e. a rate-dependent hyperpolarising feedback on the excitatory pool. The
# bottom of the sweep, Δg_K = 0, removes the adaptation entirely and so is a negative control.
#
# Two things differ from the gamma measurement, both because the frequency is ~10x lower:
#   - the Welch window must be ~10x longer, so the run is longer to keep enough windows to average
#     (1 Hz resolution, fine for gamma, returns a single-bin FWHM here --- an artefact, not a width);
#   - the LFP patch has to be re-chosen, and SMALLER wins. The slow rhythm is not globally coherent:
#     it is a patchy, locally organised up/down modulation, so a large patch averages over
#     INDEPENDENT local oscillators and cancels it (prominence falls from ~3 at 8x8 to ~0.2 over the
#     whole sheet, while the 4/8/16 rungs agree with each other). Consistent with connection kernels
#     of sigma = 0.06-0.14 mm on a 0.5 mm sheet, i.e. a coherence length well under the sheet.
# This script records the WHOLE patch-size ladder in one pass, so that choice is settled on the
# sweep's own data --- see the `_patches` supplementary figure --- rather than asserted.

using DrWatson
DrWatson.@quickactivate :WRCircuit
using JLD2
using Optim
using ForwardDiff                       # with Optim, activates the MAPPLE fit! refinement (OptimExt)
using MoreMaps
using CairoMakie
using DelimitedFiles
using Dewdrop: Aggregate, Trace
WRCircuit.@preamble
set_theme!(fathom())

# ──────────────────────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────────────────────

const PARAM = :Delta_g_K
const VALUES = round.(range(0, 0.005, length = 32); sigdigits = 4)   # 0 = no adaptation; working point Δg_K₀ = 0.002
const SEED = 1

const TMAX = 60u"s"
const TRANSIENT = 10u"s"   # discarded; simulations always begin at 0. Longer than the gamma run's ---
const DT = 0.1             # the adaptation transient itself is slow
const DT_S = (DT / 1000)u"s"   # the recorded LFP's sample interval, in SECONDS, so `spectrum` returns Hz

# The patch-size ladder, in E grid cells per side (71 = the entire sheet). Every size is recorded in
# the same run; `PATCH` is the one the main figure fits. `NSIDE²` patches per size (one for the whole
# sheet) give a spread across statistically equivalent patches.
const PATCH_SIZES = (2, 4, 8, 16, 32, 71)
const NSIDE = 2

# ---- set from the patch-size exploration; see the header and the `_patches` figure ----
const PATCH = 8            # side of the patch the main figure fits; 4/8/16 agree, 32 and up wash out
const FBAND_HZ = (0.5, 30.0)   # brackets the 3-6 Hz peak; stops short of the gamma peak at ~50 Hz
const FMIN = 0.25u"Hz"     # Welch window = fs/FMIN. The gamma run's 1 Hz is too coarse here: it
#                            returns FWHM ≈ 0.9 Hz, one frequency bin, rather than a width
# --------------------------------------------------------------------------------------

const FBAND = FBAND_HZ[1]u"Hz" .. FBAND_HZ[2]u"Hz"
const FSHOW = 0.25u"Hz" .. 200u"Hz"   # displayed range of the spectrum panels (wider than the fit band)
const SAVE_FMAX = 500u"Hz"            # spectra above this are neither fit nor drawn, so never saved
const NCOMPONENTS = 1

const NAME = "theta_peak_variation"
const PATH = plotsdir(NAME)

# ──────────────────────────────────────────────────────────────────────────────
# Simulate: one batch, every patch size at once, reduced to Welch spectra
# ──────────────────────────────────────────────────────────────────────────────

const d0 = defaults(models.Spatial)
const ne = round(Int, sqrt(Float64(d0.rho)) * Float64(d0.dx))   # E sheet side (build_spatial); NE = ne²

"""
    patch_indices(i0, j0, side)

Linear E indices of the `side`×`side` block of the sheet whose corner cell is `(i0, j0)`.
`Dewdrop.grid_positions` lays x out fastest, so neuron `(j-1)·ne + i` sits at grid cell `(i, j)` ---
the same column-major convention demo_run.jl uses when it reshapes `V` into `(ne, ne)`. `mod1` wraps,
since the sheet is periodic. E is the first population built, so these are global neuron indices too.
"""
patch_indices(i0, j0, side) =
    vec([mod1(i0 + di, ne) + (mod1(j0 + dj, ne) - 1) * ne for di in 0:(side - 1), dj in 0:(side - 1)])

"Patch corner cells for one size: an `NSIDE`×`NSIDE` spread over the sheet, or just the origin when
the patch already covers it."
function patch_origins(side)
    side >= ne && return [(1, 1)]
    s = round.(Int, range(1, ne - side + 1, length = NSIDE))
    return vec([(i, j) for i in s, j in s])
end

const origins = NamedTuple(Symbol(:s, s) => patch_origins(s) for s in PATCH_SIZES)
monitor_key(side, k) = Symbol(:lfp, side, :_, k)

const config = (;
    param = string(PARAM), lo = first(VALUES), hi = last(VALUES), n = length(VALUES),
    seed = SEED, tmax = ustrip(u"s", TMAX), sizes = join(PATCH_SIZES, "-"), nside = NSIDE,
)

"""
    simulate_lfp(config) -> Dict

Run the batch and reduce it to what the fits need: `"spectra"`, a `Dict` mapping each patch size to a
`(𝑓 × member × patch)` array of LFP power spectra. The traces themselves are dropped rather than saved
--- they are several GB and only ever consumed by `spectrum`.
"""
function simulate_lfp(config)
    record = NamedTuple(
        monitor_key(s, k) => Aggregate(Trace(:V; of = patch_indices(o..., s)), :mean)
            for s in PATCH_SIZES for (k, o) in enumerate(origins[Symbol(:s, s)])
    )
    B = length(VALUES)
    swept(p) = PARAM === p ? collect(Float64, VALUES) : fill(Float64(d0[p]), B)

    @info "Simulating $B members over $(length(record)) patch monitors ($PARAM ∈ [$(first(VALUES)), $(last(VALUES))])"
    bs = simulate_batch(
        Spatial(; key = SEED), TMAX, swept(:delta), swept(:Delta_g_K);
        dt = DT, record, progress = true, scatter = :compacted
    )

    # (B, nsteps) per monitor → drop the transient → a (𝑡 × member) timeseries → Welch power spectrum.
    tr = round(Int, ustrip(u"ms", TRANSIENT) / DT)
    members = Dim{:member}(collect(Float64, VALUES))
    taxis(n) = 𝑡(range(0.0u"s"; step = DT_S, length = n))
    # Kept only up to SAVE_FMAX: everything above is display padding, and dropping it takes the saved
    # file from ~100 MB to ~10 MB.
    function spec_of(key)
        y = permutedims(getproperty(bs.record, key).data[:, (tr + 1):end])
        x = Timeseries(Float64.(y), taxis(size(y, 1)), members)
        return spectrum(x .- mean(x, dims = 𝑡), FMIN)[𝑓 = 0u"Hz" .. SAVE_FMAX]
    end
    function spectra_of(side)
        ks = eachindex(origins[Symbol(:s, side)])
        return stack(ToolsArray([spec_of(monitor_key(side, k)) for k in ks], Dim{:patch}(ks)))
    end
    return Dict(
        "spectra" => Dict(s => spectra_of(s) for s in PATCH_SIZES),
        "values" => collect(VALUES), "config" => config,
    )
end

data, datapath = produce_or_load(
    simulate_lfp, config, datadir(NAME);
    filename = savename(config; connector), tag = true
)
@info "LFP spectra at $datapath"

# ──────────────────────────────────────────────────────────────────────────────
# Fit: one single-peak MAPPLE per (member, patch), at every patch size
# ──────────────────────────────────────────────────────────────────────────────

"""
    peakfit(s) -> NamedTuple

Single-peak MAPPLE fit to one LFP power spectrum over `FBAND`, log-sampled first so every decade
weighs equally. Returns the peak's centre frequency (Hz), FWHM (Hz), height (linear power above the
aperiodic background) and `prominence` --- that height divided by the fitted background AT the peak,
so it is dimensionless and comparable across patch sizes, whose absolute power differs. All-`NaN` if
the fit finds no peak.
"""
function peakfit(s)
    p = logsample(ustripall(s[𝑓 = FBAND]))
    m = fit(MAPPLE, p; components = NCOMPONENTS, peaks = 1)
    fit!(m, p)
    fc, fwhm, h = peakparams_hz(m)
    isempty(fc) && return (; f = NaN, fwhm = NaN, height = NaN, prominence = NaN, β = NaN)
    A = only(h)
    bg = only(predict(m, [only(fc)])) - A      # the aperiodic model at the peak, peak removed
    return (;
        f = only(fc), fwhm = only(fwhm), height = A,
        prominence = bg > 0 ? A / bg : NaN, β = last(betas(m)),
    )
end

const NOFIT = (; f = NaN, fwhm = NaN, height = NaN, prominence = NaN, β = NaN)

spectra = data["spectra"]
xvals = data["values"]      # `values` is a Base export, so the swept axis lives under a local name

"Fit every (member, patch) spectrum at one patch size. A member with no detectable peak (Δg_K near 0)
comes back as `NOFIT` and drops out of the summaries."
function fit_size(side)
    return map(Chart(LogLogger(), Threaded()), eachslice(spectra[side]; dims = (:member, :patch))) do sp
        try
            peakfit(sp)
        catch err
            @warn "Failed peak fit" patch_side = side err
            NOFIT
        end
    end
end

fits = Dict(s => fit_size(s) for s in PATCH_SIZES)

# A member only counts as having a slow rhythm if EVERY patch's fit found a peak and the median peak
# clears `MIN_PROMINENCE` --- it must add at least as much power as the background beneath it. Without
# the gate the low-Δg_K members report a peak on the strength of one patch in four, with the centre
# frequency jumping around (1.1, 2.0, 4.3 Hz over three adjacent members) and prominence below 1: a
# Gaussian laid over a curved background, not a resonance. The gate is what makes Δg_K → 0 read as the
# negative control it is.
const MIN_PROMINENCE = 1.0

"""
    across_patches(fx, k; gate = true)

Median and inter-quartile range of field `k` across patches, one entry per member; `NaN` where the
member fails the detection gate above. Pass `gate = false` for the raw value regardless of quality ---
used only by the patch-size figure, whose whole point is to show fit quality collapsing.
"""
function across_patches(fx, k; gate = true)
    stat = map(eachslice(fx; dims = :member)) do row
        v = [getproperty(x, k) for x in row]
        prom = [x.prominence for x in row]
        if gate
            detected = all(!isnan, v) && all(!isnan, prom) && median(prom) >= MIN_PROMINENCE
            detected || return (NaN, NaN, NaN)
        else
            v = filter(!isnan, v)
            isempty(v) && return (NaN, NaN, NaN)
        end
        return (median(v), quantile(v, 0.25), quantile(v, 0.75))
    end
    return collect(first.(stat)), collect(getindex.(stat, 2)), collect(last.(stat))
end

const PANELS = (
    (:height, "Peak height (a.u.)", true),
    (:f, "Peak frequency (Hz)", false),
    (:fwhm, "Peak width, FWHM (Hz)", false),
)
summaries = NamedTuple(k => across_patches(fits[PATCH], k) for (k, _, _) in PANELS)

let ok = findall(isfinite, first(summaries[:f]))
    @info "$(length(ok))/$(length(xvals)) members pass the detection gate " *
        "(all $(NSIDE^2) patches peaked, prominence ≥ $MIN_PROMINENCE): $PARAM ∈ [$(xvals[first(ok)]), $(xvals[last(ok)])]"
    for (k, lab, _) in PANELS
        med = first(summaries[k])[ok]
        @info "$lab: $(round(first(med); sigdigits = 4)) → $(round(last(med); sigdigits = 4)) across those members"
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

const paramlabel = "Δg_K  (mS/cm²)"
const pcolors = cgrad(sunrise, length(xvals); categorical = true)
# Patch size is categorical and only six-valued, so the theme's own colour cycle reads better than a
# gradient (whose light end vanishes against the page).

"Patch-averaged spectrum of member `i` at patch size `s`, restricted to `band`."
patch_mean(s, i; band = FSHOW) =
    ustripall(dropdims(mean(spectra[s][member = i][𝑓 = band], dims = :patch), dims = :patch))

begin # * Main figure --- the peak against Δg_K, at the chosen patch size
    f = FourPanel()
    gs = subdivide(f, 2, 2)

    ax_s = Axis(
        gs[1][1, 1]; xlabel = "Frequency (Hz)", ylabel = "Power (a.u.)",
        xscale = log10, yscale = log10, title = "Patch LFP spectrum ($(PATCH)×$(PATCH))"
    )
    vspan!(ax_s, FBAND_HZ...; color = (:gray, 0.12), strokewidth = 0)
    for i in eachindex(xvals)
        y = patch_mean(PATCH, i)
        lines!(ax_s, collect(lookup(y, 𝑓)), collect(y); color = pcolors[i], linewidth = 1.5)
    end
    Colorbar(gs[1][1, 2]; colormap = pcolors, limits = extrema(xvals), label = paramlabel, width = 12)

    axs = map(enumerate(PANELS)) do (i, (k, lab, logy))
        med, lo, hi = summaries[k]
        ok = findall(isfinite, med)     # members whose fit found no peak drop out of the line
        ax = Axis(
            gs[i + 1][1, 1]; xlabel = paramlabel, ylabel = lab,
            yscale = logy ? log10 : identity
        )
        band!(ax, xvals[ok], lo[ok], hi[ok]; color = (mesopelagic, 0.25))
        lines!(ax, xvals[ok], med[ok]; color = mesopelagic, linewidth = 2.5)
        scatter!(ax, xvals[ok], med[ok]; color = mesopelagic, markersize = 7)
        vlines!(ax, [Float64(d0[PARAM])]; color = :gray, linestyle = :dash, linewidth = 1)
        Box(gs[i + 1][1, 2]; visible = false, width = 12)   # match panel a's colorbar column width
        ax
    end

    addlabels!(f)
    display(f)
end

# The patch-size test. The slow rhythm is only locally coherent, so a patch wider than the coherence
# length averages over independent oscillators and cancels it; the peak's prominence over its own
# background therefore FALLS as the patch grows past a few cells. Panel a shows the spectra at the
# working point, panel b the prominence against patch size for every Δg_K --- the evidence behind
# `PATCH`.
const i0 = argmin(abs.(xvals .- Float64(d0[PARAM])))   # member nearest the working point
const prominences = [first(across_patches(fits[s], :prominence; gate = false))[i]
    for i in eachindex(xvals), s in PATCH_SIZES]
begin # * Supplementary figure --- the patch-size ladder
    fp = TwoPanel()
    gp = subdivide(fp, 1, 2)

    ax1 = Axis(
        gp[1][1, 1]; xlabel = "Frequency (Hz)", ylabel = "Power (a.u.)",
        xscale = log10, yscale = log10,
        title = "Δg_K = $(round(xvals[i0]; sigdigits = 3))"
    )
    vspan!(ax1, FBAND_HZ...; color = (:gray, 0.12), strokewidth = 0)
    for s in PATCH_SIZES
        y = patch_mean(s, i0)
        lines!(ax1, collect(lookup(y, 𝑓)), collect(y); linewidth = 2, label = "$(s)×$(s)")
    end
    axislegend(ax1; position = :lb, framevisible = false, labelsize = 10)

    ax2 = Axis(
        gp[2][1, 1]; xlabel = "Patch side (cells)", ylabel = "Peak prominence",
        xscale = log2, yscale = log10, title = "Peak / background"
    )
    for i in eachindex(xvals)
        pr = prominences[i, :]
        ok = findall(isfinite, pr)
        length(ok) < 2 && continue
        lines!(ax2, collect(PATCH_SIZES)[ok], pr[ok]; color = pcolors[i], linewidth = 1.5)
    end
    hlines!(ax2, [MIN_PROMINENCE]; color = :gray, linewidth = 1)   # the detection floor
    vlines!(ax2, [PATCH]; color = :gray, linestyle = :dash, linewidth = 1)
    Colorbar(gp[2][1, 2]; colormap = pcolors, limits = extrema(xvals), label = paramlabel, width = 12)

    addlabels!(fp)
    display(fp)
end

begin # * Save figures
    wsave(PATH * ".pdf", f)
    wsave(PATH * ".png", f)
    wsave(PATH * "_patches.pdf", fp)
    wsave(PATH * "_patches.png", fp)
    @info "Saved $PATH"
end

begin # * Save source data
    mkpath(PATH)
    s̄ = reduce(hcat, [collect(patch_mean(PATCH, i; band = FBAND)) for i in eachindex(xvals)])
    fax = collect(lookup(patch_mean(PATCH, 1; band = FBAND), 𝑓))
    writedlm(
        joinpath(PATH, "panelA.tsv"),
        vcat(hcat("f_Hz", permutedims(["$PARAM=$v" for v in xvals])), hcat(fax, s̄)), '\t'
    )
    for (i, (k, _, _)) in enumerate(PANELS)
        med, lo, hi = summaries[k]
        writedlm(
            joinpath(PATH, "panel$('A' + i).tsv"),
            vcat([string(PARAM) "median" "q25" "q75"], hcat(xvals, med, lo, hi)), '\t'
        )
    end
    # The patch-size test: prominence for every (patch size, Δg_K).
    writedlm(
        joinpath(PATH, "patches_prominence.tsv"),
        vcat(hcat(string(PARAM), permutedims(["side=$s" for s in PATCH_SIZES])), hcat(xvals, prominences)), '\t'
    )
    @info "Saved source data to $PATH"
end
