#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.13 --handle-signals=yes -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
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

# The swept parameter. `simulate_batch` co-executes members over ONE connectome, so only parameters
# that leave the connectome untouched can be swept here: :delta, :Delta_g_K, :tau_r_e, :tau_d_e.
const PARAM = :delta
const VALUES = round.(range(2.5, 4.05, step = 0.05); sigdigits = 4)   # 32 members; working point δ₀ = 4
const SEED = 1

const TMAX = 35u"s"
const TRANSIENT = 5u"s"    # discarded; simulations always begin at 0
const DT = 0.1             # ms
const DT_S = (DT / 1000)u"s"   # the recorded LFP's sample interval, in SECONDS, so `spectrum` returns Hz

# LFP patch geometry, matching demo_run.jl: a `PATCH`×`PATCH` block of the E sheet, averaged. `NSIDE²`
# such patches are sampled on an evenly spaced grid, giving a spread over statistically equivalent
# patches to put an interval on each fitted peak parameter.
const PATCH = 10
const NSIDE = 3

# Band the peak is fit over, and the model fit to it. The peak sits at 50-85 Hz, so the band must
# bracket it with clean background on both sides. We use 10-200 Hz with ONE power-law component
# rather than demo_run.jl's 10-1000 Hz: past ~200 Hz the LFP spectrum steepens and then meets the
# Welch noise floor, which a single component cannot follow, and the Gaussian widens to absorb the
# misfit (FWHM 52 → 70 Hz for the same peak). Two components over the wide band is worse still ---
# the knee and the peak trade off and the fit runs away (β > 0, heights ~10³ too large). Below 10 Hz
# a separate low-frequency component dominates and is not what we are measuring here.
const FBAND_HZ = (10.0, 200.0)
const FBAND = FBAND_HZ[1]u"Hz" .. FBAND_HZ[2]u"Hz"
const FSHOW = 2u"Hz" .. 1000u"Hz"   # displayed range of panel a (wider than the fit band)
const NCOMPONENTS = 1
const FMIN = 1.0u"Hz"      # Welch window length = fs/FMIN

const NAME = "peak_variation"
const PATH = plotsdir(NAME)

# ──────────────────────────────────────────────────────────────────────────────
# Simulate: one batch, NPATCH patch LFPs per member, reduced to Welch spectra
# ──────────────────────────────────────────────────────────────────────────────

const d0 = defaults(models.Spatial)
const ne = round(Int, sqrt(Float64(d0.rho)) * Float64(d0.dx))   # E sheet side (build_spatial); NE = ne²

"""
    patch_indices(i0, j0; side = PATCH)

Linear E indices of the `side`×`side` block of the sheet whose corner cell is `(i0, j0)`.
`Dewdrop.grid_positions` lays x out fastest, so neuron `(j-1)·ne + i` sits at grid cell `(i, j)` ---
the same column-major convention demo_run.jl uses when it reshapes `V` into `(ne, ne)`. `mod1` wraps,
since the sheet is periodic. E is the first population built, so these are global neuron indices too.
"""
patch_indices(i0, j0; side = PATCH) =
    vec([mod1(i0 + di, ne) + (mod1(j0 + dj, ne) - 1) * ne for di in 0:(side - 1), dj in 0:(side - 1)])

const patch_origins = let s = round.(Int, range(1, ne - PATCH + 1, length = NSIDE))
    vec([(i, j) for i in s, j in s])
end
const NPATCH = length(patch_origins)

const config = (;
    param = string(PARAM), lo = first(VALUES), hi = last(VALUES), n = length(VALUES),
    seed = SEED, tmax = ustrip(u"s", TMAX), patch = PATCH, npatch = NPATCH,
)

"""
    simulate_lfp(config) -> Dict

Run the batch and reduce it to what the fits need: a `(𝑓 × member × patch)` array of LFP power spectra
plus a short raw LFP excerpt. The traces themselves are dropped here rather than saved --- they are
~350 MB and only ever consumed by `spectrum`.
"""
function simulate_lfp(config)
    record = NamedTuple(
        Symbol(:lfp, k) => Aggregate(Trace(:V; of = patch_indices(o...)), :mean)
            for (k, o) in enumerate(patch_origins)
    )
    # Members that do not sweep a parameter hold it at its working-regime default.
    B = length(VALUES)
    swept(p) = PARAM === p ? collect(Float64, VALUES) : fill(Float64(d0[p]), B)
    member = PARAM in (:tau_r_e, :tau_d_e) ? (; tau_r_e = swept(:tau_r_e), tau_d_e = swept(:tau_d_e)) : (;)

    @info "Simulating $B members over $NPATCH patches ($PARAM ∈ [$(first(VALUES)), $(last(VALUES))])"
    bs = simulate_batch(
        Spatial(; key = SEED), TMAX, swept(:delta), swept(:Delta_g_K);
        dt = DT, record, member..., progress = true, scatter = :compacted
    )

    # (B, nsteps) per patch → drop the transient → a (𝑡 × member) timeseries → Welch power spectrum.
    tr = round(Int, ustrip(u"ms", TRANSIENT) / DT)
    members = Dim{:member}(collect(Float64, VALUES))
    taxis(n) = 𝑡(range(0.0u"s"; step = DT_S, length = n))
    spectra = map(1:NPATCH) do k
        y = permutedims(getproperty(bs.record, Symbol(:lfp, k)).data[:, (tr + 1):end])   # (𝑡 × member)
        x = Timeseries(Float64.(y), taxis(size(y, 1)), members)
        return spectrum(x .- mean(x, dims = 𝑡), FMIN)
    end
    excerpt = let n = round(Int, 1u"s" / DT_S)   # last 1 s of patch 1, for eyeballing
        y = permutedims(bs.record.lfp1.data[:, (end - n + 1):end])
        Timeseries(Float64.(y), taxis(n), members)
    end
    return Dict(
        "spectra" => ToolsArray(spectra, Dim{:patch}(1:NPATCH)) |> stack,
        "excerpt" => excerpt, "values" => collect(VALUES), "config" => config,
    )
end

data, datapath = produce_or_load(
    simulate_lfp, config, datadir(NAME);
    filename = savename(config; connector), tag = true
)
@info "LFP spectra at $datapath"

# ──────────────────────────────────────────────────────────────────────────────
# Fit: one 1-component, 1-peak MAPPLE per (member, patch)
# ──────────────────────────────────────────────────────────────────────────────

"""
    peakfit(s) -> NamedTuple

Single-peak MAPPLE fit to one LFP power spectrum over `FBAND`, log-sampled first so every decade
weighs equally. Returns the peak's centre frequency (Hz), FWHM (Hz) and height (linear power above
the aperiodic background), the aperiodic exponent, and the fitted model. All-`NaN` if the fit finds
no peak.
"""
function peakfit(s)
    p = logsample(ustripall(s[𝑓 = FBAND]))
    m = fit(MAPPLE, p; components = NCOMPONENTS, peaks = 1)
    fit!(m, p)
    fc, fwhm, h = peakparams_hz(m)
    isempty(fc) && return (; f = NaN, fwhm = NaN, height = NaN, β = NaN, m)
    return (; f = only(fc), fwhm = only(fwhm), height = only(h), β = last(betas(m)), m)
end

spectra = data["spectra"]
xvals = data["values"]      # `values` is a Base export, so the swept axis lives under a local name

fits = map(Chart(LogLogger(), Threaded()), eachslice(spectra; dims = (:member, :patch))) do s
    try
        peakfit(s)
    catch err
        @warn "Failed peak fit" err
        (; f = NaN, fwhm = NaN, height = NaN, β = NaN, m = nothing)
    end
end

"Median and inter-quartile range of field `k` across patches, one entry per member. NaNs are dropped."
function across_patches(fits, k)
    stat = map(eachslice(fits; dims = :member)) do row
        v = filter(!isnan, [getproperty(f, k) for f in row])
        isempty(v) ? (NaN, NaN, NaN) : (median(v), quantile(v, 0.25), quantile(v, 0.75))
    end
    return collect(first.(stat)), collect(getindex.(stat, 2)), collect(last.(stat))
end

const PANELS = (
    (:height, "Peak height (a.u.)", true),
    (:f, "Peak frequency (Hz)", false),
    (:fwhm, "Peak width, FWHM (Hz)", false),
)
summaries = NamedTuple(k => across_patches(fits, k) for (k, _, _) in PANELS)

for (k, lab, _) in PANELS
    med = first(summaries[k])
    @info "$lab: $(round(first(med); sigdigits = 4)) → $(round(last(med); sigdigits = 4)) " *
        "over $PARAM ∈ [$(first(xvals)), $(last(xvals))]"
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

const paramlabel = Dict(
    :delta => "δ  (I:E ratio)", :Delta_g_K => "Δg_K  (mS/cm²)",
    :tau_r_e => "τ_r_e (ms)", :tau_d_e => "τ_d_e (ms)",
)[PARAM]
const pcolors = cgrad(sunrise, length(xvals); categorical = true)

begin # * Render
    f = FourPanel()
    gs = subdivide(f, 2, 2)

    begin # * a --- the LFP spectra themselves, one line per member. Drawn over a wider range than
        # FBAND, with the fit band shaded, so the band choice can be judged against the curves.
        ax = Axis(
            gs[1][1, 1]; xlabel = "Frequency (Hz)", ylabel = "Power (a.u.)",
            xscale = log10, yscale = log10, title = "Patch LFP spectrum"
        )
        vspan!(ax, FBAND_HZ...; color = (:gray, 0.12), strokewidth = 0)
        for i in eachindex(xvals)   # patch-averaged, so the panel shows what the fits see on average
            s = ustripall(dropdims(mean(spectra[member = i][𝑓 = FSHOW], dims = :patch), dims = :patch))
            lines!(ax, collect(lookup(s, 𝑓)), collect(s); color = pcolors[i], linewidth = 1.5)
        end
        Colorbar(
            gs[1][1, 2]; colormap = pcolors, limits = extrema(xvals),
            label = paramlabel, width = 12
        )
    end

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

begin # * Save figure
    wsave(PATH * ".pdf", f)
    wsave(PATH * ".png", f)
    @info "Saved $PATH"
end

begin # * Save source data
    mkpath(PATH)
    s̄ = ustripall(dropdims(mean(spectra[𝑓 = FBAND], dims = :patch), dims = :patch))   # (𝑓 × member)
    writedlm(
        joinpath(PATH, "panelA.tsv"),
        vcat(
            hcat("f_Hz", permutedims(["$PARAM=$v" for v in xvals])),
            hcat(collect(lookup(s̄, 𝑓)), collect(s̄))
        ), '\t'
    )
    for (i, (k, lab, _)) in enumerate(PANELS)
        med, lo, hi = summaries[k]
        writedlm(
            joinpath(PATH, "panel$('A' + i).tsv"),
            vcat([string(PARAM) "median" "q25" "q75"], hcat(xvals, med, lo, hi)), '\t'
        )
    end
    @info "Saved source data to $PATH"
end
