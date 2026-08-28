#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WorkingRegime"
using JLD2
using CairoMakie
using Fathom
using DataFrames
using TimeseriesTools
using TimeseriesBase
using Statistics
using Random
set_theme!(Fathom.fathom())


function bootstrapmedian(x; N = 10_000, α = 0.05)
    x = collect(skipmissing(x))
    x = filter(!isnan, x)
    rng = Random.MersenneTwister(42)
    n = length(x)
    meds = [median(x[rand(rng, 1:n, n)]) for _ in 1:N]
    m = median(x)
    lo, hi = quantile(meds, (α / 2, 1 - α / 2))
    return m, (lo, hi)
end

stimuli_str = ["r\"Natural_Images\"", "spontaneous", "flash_250ms"]

_select(x, selectors::Pair...) = getindex(x; (Symbol(n) => At(v) for (n, v) in selectors)...)

# Load experiment data
inpath = projectdir("WRExperiment", "data", "WRExperiment.jld2")
plot_data = jldopen(
    f -> Dict(k => f[k] for k in keys(f)), inpath;
    typemap = toolsarray_typemap
)

# Load circuit data
circuit_path = projectdir("WRCircuit", "data", "circuit_curves.jld2")
circuit = loadtoolsarray(circuit_path, "circuit_curves")

outdir = plotsdir("combined_curves")
experiment_color = :cornflowerblue
circuit_color = :crimson

for stim_str in stimuli_str
    # begin
    stim_str = "spontaneous"
    @info "Plotting combined curves for $stim_str"

    mad = plot_data["mad_curves"][stim_str]
    psd = plot_data["spectral_curves"][stim_str]["VISp"]
    fano = plot_data["fano_curves"][stim_str]

    f = TwoPanel()

    # Panel 1: MAD curve (normalized)
    begin
        # Normalize experiment and circuit to same range
        normalize(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))

        ax = Axis(
            f[1, 1]; ylabel = "MAD (arb. units)",
            xlabel = "Time lag (s)",
            title = "Mean absolute deviation",
            xscale = log10, yscale = log10
        )

        mad_norm = normalize(log10.(mad.mu))
        mad_σl_norm = normalize(log10.(mad.σl))
        mad_σh_norm = normalize(log10.(mad.σh))
        band!(
            ax, mad.t_all, exp10.(mad_σl_norm), exp10.(mad_σh_norm),
            color = experiment_color, alpha = 0.5
        )
        lines!(ax, mad.t_all, exp10.(mad_norm), color = experiment_color, label = "Experiment\n(LFP)")

        mad_fit_vals = exp10.(mad.meanintercept .+ mad.meanslope .* log10.(mad.fit_t))

        idxs = mad.fit_t .< 0.005
        lines!(
            ax, mad.fit_t[idxs] .* 2, exp10.(normalize(log10.(mad_fit_vals)))[idxs],
            linestyle = :dash, color = experiment_color
        )

        circuit_mad_norm = normalize(log10.(circuit.mad.mu))
        lines!(
            ax, circuit.mad.t, exp10.(circuit_mad_norm);
            color = circuit_color, label = "Circuit\n(input)"
        )

        idxs = circuit.mad.fit_t .< 0.005
        lines!(
            ax, circuit.mad.fit_t[idxs] ./ 2, exp10.(normalize(log10.(circuit.mad.fit_vals)))[idxs];
            color = circuit_color, linestyle = :dash
        )

        text!(
            ax, 1.0e-2, 10^0.7;
            text = "a = $(round(mad.meanslope, sigdigits = 2))",
            color = experiment_color, align = (:left, :top)
        )
        text!(
            ax, 10^(-3.35), 10^1;
            text = "a = $(round(circuit.mad.exponent, sigdigits = 2))",
            color = circuit_color, align = (:left, :top)
        )

        axislegend(ax; position = :rb, fontsize = 12)

        ax.limits = ((10^(-3.4), 10^0.1), nothing)
    end

    # Panel 2: PSD curve (normalized)
    begin
        ax = Axis(
            f[1, 2];
            xlabel = "Frequency (Hz)",
            ylabel = "PSD (arb. units)",
            title = "Power spectral density",
            xscale = log10, yscale = log10,
            xticks = [3, 10, 30, 100]
        )

        psd_norm = normalize(log10.(psd.μ))
        psd_σl_norm = normalize(log10.(psd.σl))
        psd_σh_norm = normalize(log10.(psd.σh))
        lines!(ax, psd.f, exp10.(psd_norm); color = (experiment_color, 0.8))
        band!(
            ax, psd.f, exp10.(psd_σl_norm), exp10.(psd_σh_norm);
            color = (experiment_color, 0.32)
        )

        # # Peaks (normalized)
        # psd_peak_norm = normalize(log10.(psd.peak_vals))
        # scatter!(ax, psd.peak_freqs, exp10.(psd_peak_norm) .* 1.25, color=:black,
        #     markersize=10, marker=:dtriangle)
        # text!(ax, psd.peak_freqs, exp10.(psd_peak_norm);
        #     text=string.(round.(psd.peak_freqs, sigdigits=2)) .* [" Hz"],
        #     align=(:center, :bottom), color=:black, rotation=0,
        #     fontsize=16, offset=(0, 5))

        # The APERIODIC (1/f) component of the MAPPLE fit, not the full fit. Both fits here use ONE
        # power-law component, so the background is exactly a straight line in log-log whose slope is
        # the spectral exponent --- drawing it directly shows the quantity the panel quotes, without
        # the Gaussian bumps the exponent is defined to exclude. Only the slope carries information:
        # the offset comes from the same min-max normalisation as every other curve in this panel,
        # and drawing it this way makes the line and the annotated `b` the same number by
        # construction (the full fit was a fit to the median curve, while `b` is the median of the
        # per-session fits, so the two used to disagree slightly).
        lines!(
            ax, psd.f, 1.25 .* exp10.(normalize(log10.(psd.f .^ psd.spectral_exponent_median)));
            color = experiment_color,
            linestyle = :dash
        )

        # Circuit overlay (normalized to same range)
        circuit_psd_norm = normalize(log10.(circuit.psd.mu))
        lines!(
            ax, circuit.psd.f, exp10.(circuit_psd_norm) .* 1.35;
            color = circuit_color
        )
        lines!(
            ax, circuit.psd.fit_f,
            1.25 .* exp10.(normalize(log10.(circuit.psd.fit_f .^ circuit.psd.exponent)));
            color = circuit_color, linestyle = :dash
        )

        text!(
            ax, 7, 10^0.4;
            text = "b = $(round(psd.spectral_exponent_median; sigdigits = 3))",
            color = experiment_color, align = (:left, :top)
        )
        text!(
            ax, 20, 10;
            text = "b = $(round(circuit.psd.exponent; sigdigits = 3))",
            color = circuit_color, align = (:left, :bottom)
        )

        ax.limits = ((2, 500), nothing)
    end

    # Panel 3: Fano factor curve (unnormalized, single axis)
    begin
        ax = Axis(
            f[1, 3];
            xlabel = "Time lag (s)",
            ylabel = "Fano factor",
            title = "Fano factor",
            xscale = log10, yscale = log10
        )

        band!(ax, 0.001 .* fano.t_all, fano.sl, fano.su, color = experiment_color, alpha = 0.3)
        lines!(ax, 0.001 .* fano.t_all, fano.mu, color = experiment_color)

        lines!(
            ax, 0.001 .* exp10.(fano.fit_t_range) ./ 2,
            exp10.(fano.mintercept .+ fano.mslope .* fano.fit_t_range);
            color = experiment_color, linestyle = :dash, linewidth = 3
        )

        lines!(
            ax, 0.001 .* circuit.fano.t, circuit.fano.mu;
            color = circuit_color
        )
        idxs = 10 .< circuit.fano.t .< 100
        lines!(
            ax, 0.001 .* circuit.fano.t[idxs] ./ 2, circuit.fano.mu[idxs];
            color = circuit_color, linestyle = :dash
        )


        text!(
            ax, 0.001 .* 40, 10^0.32;
            text = "c = $(round(fano.mslope, digits = 2))",
            color = experiment_color, align = (:left, :center)
        )
        text!(
            ax, 0.001 .* 1.2, 1.4;
            text = "c = $(round(circuit.fano.exponent, digits = 2))",
            color = circuit_color, align = (:left, :center)
        )
    end

    # addlabels!(f)

    display(f)
    outfile = joinpath(outdir, "combined_curves_$(stim_str).svg")
    wsave(outfile, f)
    wsave(Base.replace(outfile, ".svg" => ".png"), f)
    @info "Saved $outfile"

    # Summary statistics (matching per-metric reference scripts)
    begin
        # MAD: bootstrap CI over per-session slopes (requires mad.slope field)
        if hasproperty(mad, :slope)
            mad_m, (mad_sl, mad_su) = bootstrapmedian(collect(mad.slope))
            mad_line = "$stim_str mad median: $mad_m, CI: ($mad_sl, $mad_su)"
        else
            mad_line = "$stim_str mad median: $(mad.meanslope), CI: (NA, NA)"
        end
        @info mad_line

        # Spectral: bootstrap CI over sessions at VISp L2/3
        spec = plot_data["spectral_exponents"][stim_str]
        mb = _select(spec, :Structure => "VISp", :layer => 2)
        sp_m, (sp_sl, sp_su) = bootstrapmedian(collect(mb))
        spec_line = "$stim_str spectral median: $sp_m, CI: ($sp_sl, $sp_su)"
        @info spec_line

        # Fano: bootstrap CI over sessions at VISp L2/3 slopes
        fslopes = plot_data["fano_slopes"][stim_str]
        fs = _select(fslopes, :Structure => "VISp", :layer => 2)
        fa_m, (fa_sl, fa_su) = bootstrapmedian(collect(fs))
        fano_line = "$stim_str fano median: $fa_m, CI: ($fa_sl, $fa_su)"
        @info fano_line

        # Circuit: bootstrap CI from per-neuron exponents when available.
        function _circuit_line(label, sub)
            if hasproperty(sub, :exponents)
                m, (lo, hi) = bootstrapmedian(collect(sub.exponents))
                return "circuit $label median: $m, CI: ($lo, $hi)"
            else
                return "circuit $label exponent: $(sub.exponent)"
            end
        end
        circ_mad_line = _circuit_line("mad", circuit.mad)
        circ_psd_line = _circuit_line("spectral", circuit.psd)
        circ_fano_line = _circuit_line("fano", circuit.fano)
        @info circ_mad_line
        @info circ_psd_line
        @info circ_fano_line

        txtfile = joinpath(outdir, "combined_curves_$(stim_str).txt")
        open(txtfile, "w") do io
            println(io, mad_line)
            println(io, spec_line)
            println(io, fano_line)
            println(io, circ_mad_line)
            println(io, circ_psd_line)
            println(io, circ_fano_line)
        end
        @info "Saved $txtfile"
    end
end
