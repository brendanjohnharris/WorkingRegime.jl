#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
using StableDistributions
WRTheory.@preamble()
set_theme!(foresight(:physics))

begin
    tmax = 25000.0
    tmin = 5000.0
    shared_params = (;
                     α = 1.5,
                     β = 0.85,
                     γ = 0.03,
                     η = 0.01,
                     domain = -10 .. 10,
                     boundaries = PeriodicBox(-5 .. 5),
                     u0 = [0.0, 0.0],
                     dt = 0.1,
                     tspan = tmax,
                     approx_n_modes = 1000,
                     τ = 1000.0,
                     seed = 44,
                     λ = 1e-4)

    noise = @time FractionalNeuralSampling.Samplers.gen_lfsm_fns(shared_params.α,
                                                                 shared_params.β;
                                                                 tspan = shared_params.tspan,
                                                                 dt = shared_params.dt,
                                                                 seed = shared_params.seed,
                                                                 nhist = round(Int,
                                                                               shared_params.τ /
                                                                               shared_params.dt)) # Same noise for all panels
end

begin # * Set up figure
    f = FourPanel()
    gs = subdivide(f, 2, 2)
end
begin # * Define an example sampler
    𝜋 = test_density(:flat)
    ps = (; shared_params..., 𝜋)

    S = bFNS(; ps..., noise)

    sol = @time solve(S) |> Timeseries |> eachcol |> first
    sol = rectify(sol, dims = 𝑡; tol = 1)
    sol = sol[𝑡 = tmin .. tmax] # Remove transient
    ssol = set(sol, 𝑡 => times(sol) ./ 1000) # To seconds
end

begin # * And do a similar simulation but with momentum
    𝜋 = test_density(:unimodal)
    gamma_ps = (; shared_params..., 𝜋)

    gS = bFNS(; gamma_ps..., noise)

    gsol = @time solve(gS) |> Timeseries |> eachcol |> first
    gsol = rectify(gsol, dims = 𝑡; tol = 1)
    gsol = gsol[𝑡 = tmin .. tmax] # Remove transient
    gsol = set(gsol, 𝑡 => times(gsol) ./ 1000) # To seconds
end

function effective_potential(S::AbstractSampler)
    # * Get the drift term
    ps, 𝜋 = S.p
    @unpack ∇𝒟𝜋, 𝜋s, λ = ps

    ∇V = x -> ∇𝒟𝜋(x) / (𝜋s(x) + λ)

    function V(xs::AbstractRange)
        dx = step(xs)
        xs = range(first(xs) - dx / 4, stop = last(xs) + dx / 4, step = dx)
        ∇Vs = .-∇V.(xs)
        V = cumsum(∇Vs) .* dx
        V .-= minimum(V)
        return V
    end
end
begin # * MAD
    τs = logrange(step(ssol), 1.0, length = 100) # Seconds
    mads = madev(ssol, τs) * 1.5
    gmads = madev(gsol, τs)

    ax = Axis(gs[1], xlabel = "Time lag (s)", ylabel = "MAD",
              xscale = log10, yscale = log10, title = "Mean absolute deviation")

    lines!(ax, τs, mads; label = "Unconfined")
    lines!(ax, τs, gmads; color = cucumber, label = "Unimodal")

    frange = 1e-4 .. 1e-2
    m = fit(MAPPLE, mads[𝑡 = frange]; peaks = 0, components = 1)
    fit!(m, mads[𝑡 = frange])
    # m.params.components
    β = S.p[1].β
    α = S.p[1].α
    label = "a = $(round(m.params.components[1].β, digits=2))"
    lines!(ax, τs[τs .∈ [frange]], predict(m, τs[τs .∈ [frange]]); label, color = :crimson,
           linestyle = :dash)
    text!(ax, [1], [0]; space = :relative,
          text = "α=$(shared_params.α) \nβ=$(shared_params.β) \nγ=$(shared_params.γ) \nη=$(shared_params.η) \n ",
          align = (:right, :bottom))
    # ! Add annotation and brownian spectrum
    l = axislegend(ax; position = :lt)
    reverselegend!(l)
    ax.limits = ((1e-4, 1.0), (0.02, 3))
end
begin # * Power spectral density
    psd = spectrum(ssol .- mean(ssol), 1.0; padding = 1000)
    psd = psd[𝑓 = eps() .. 1000] .* 1 # Offset

    gpsd = spectrum(gsol .- mean(gsol), 1.0; padding = 1000)
    gpsd = gpsd[𝑓 = eps() .. 1000]

    ax = Axis(gs[2]; xtickformat = x -> string.(round.(Int, x)),
              title = "Power spectral density")
    plotspectrum!(ax, psd; label = "Unconfined")
    plotspectrum!(ax, gpsd; color = cucumber, label = "Unimodal")
    logs = logsample(ustripall(psd)[𝑓 = 10 .. 1000])
    m = fit(MAPPLE, logs; peaks = 0, components = 1)
    fit!(m, logs)
    β = S.p[1].β
    α = S.p[1].α
    label = "b = $(round(m.params.components[end].β, digits=2))"
    lines!(ax, lookup(logs, 1), predict(m, lookup(logs, 1)); color = :red,
           linewidth = 2, label, linestyle = :dash)
    l = axislegend(ax; position = :lb)
    text!(ax, [1], [1]; space = :relative,
          text = "\nα=$(shared_params.α) \nβ=$(shared_params.β) \nγ=$(shared_params.γ) \nη=$(shared_params.η) ",
          align = (:right, :top))
    reverselegend!(l)
    # !! Add annotation and brownian spectrum
    ax.xlabel = "Frequency (Hz)"
    ax.limits = ((1, 1000), (2e-6, 1e-1))
end

# * Heatmaps
begin # * Load data
    η = S.p[1].η
    γ = S.p[1].γ
    file = datadir("bFNS_sweep", "flat_γ=$(γ)_η=$(η).jld2")
    data = load(file)
    diffusion_exponents = data["diffusion_exponent"][η = At(η), γ = At(γ)]
    spectral_exponents = data["spectral_exponent"][η = At(η), γ = At(γ)]
    accuracy = data["accuracy"][η = At(η), γ = At(γ)]
    ma = Dropdims(mean)(diffusion_exponents, dims = Obs)
    ms = Dropdims(mean)(spectral_exponents, dims = Obs)
    macc = Dropdims(mean)(accuracy, dims = Obs)
end

begin # * Plot MAD exponent heatmap
    ax = Axis(gs[3][1, 1];
              xlabel = "α", ylabel = "β", title = "Diffusion exponent",
              xgridvisible = false, ygridvisible = false,
              backgroundcolor = :gray88,
              limits = ((1.2, 2.0), (0.2, 1.0)))
    # p = heatmap!(ax, ma)
    p = contourf!(ax, ma; levels = range(0.25, 0.75, length = 10), colormap = darksunset,
                  extendhigh = :auto, extendlow = :auto)
    contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash)
    scatter!(ax, [S.p[1].α], [S.p[1].β]; color = cucumber, markersize = 10,
             strokecolor = :white, strokewidth = 1)

    # contour!(ax, ms; color = :white, levels = [-1.9], linestyle = :dash)
    Colorbar(gs[3][1, 2], p)
    Label(gs[3][1, 2, Top()], "a", font = :regular)
    # ! Add annotation and brownian spectrum
    f
end

begin # * Plot PSD exponent heatmap
    ax = Axis(gs[4][1, 1];
              xlabel = "α", ylabel = "β", title = "Spectral exponent", xgridvisible = false,
              ygridvisible = false, backgroundcolor = :gray88,
              limits = ((1.2, 2.0), (0.2, 1.0)))
    p = contourf!(ax, ms; levels = range(-2.0, -1.0, length = 10), extendhigh = :auto,
                  extendlow = :auto,
                  colormap = lightsunset)
    contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash)
    scatter!(ax, [S.p[1].α], [S.p[1].β]; color = cucumber, markersize = 10,
             strokecolor = :white, strokewidth = 1)

    # contour!(ax, ms; color = :white, levels = [-1.9], linestyle = :dash)
    Colorbar(gs[4][1, 2], p)
    Label(gs[4][1, 2, Top()], "b", font = :regular)
end

begin # * Save figure
    # rowsize!(f.layout, 0, Relative(0.1))
    # rowsize!(f.layout, 1, Relative(0.2))
    f |> display
    addlabels!(f)
    wsave(plotdir("model_summary", "model_summary.pdf"), f)
end

begin
    @info "Generating unimodal supplementary heatmaps"
    f2 = FourPanel()

    begin # * First plot the effective potential
        prange = (-1.1, 1.1)
        _xs = range(prange..., length = 1000)
        xs = _xs[1:10:end]

        V = potential(Density(gS))
        Vs = V.(xs)
        Vs = Vs .- minimum(Vs)

        Ṽ = effective_potential(gS)
        Ṽs = Ṽ(_xs)
        idxs = indexin(xs, _xs) |> Vector{Int}
        Ṽs = Ṽs[idxs]
        Ṽs .-= minimum(Ṽs)

        ax = Axis(f2[1, 1], xlabel = "x", ylabel = "V(x)",
                  limits = (extrema(xs), (-0.5, maximum(Vs))),
                  title = "Potential function")
        lines!(ax, xs, Vs; color = :cornflowerblue, label = "Potential")
        lines!(ax, xs, Ṽs; linestyle = :dash, color = :crimson,
               label = "Effective potential")
        axislegend(ax; position = :ct)
    end

    begin # * Plot distribution
        accuracy = samplingaccuracy(gsol, Density(gS);
                                    domain = only(FractionalNeuralSampling.domain(gamma_ps.boundaries)))

        ax = Axis(f2[1, 2], xlabel = "x", ylabel = "𝜋(x)",
                  limits = (extrema(xs), (0, 2.0)),
                  title = "Distribution", yticks = WilkinsonTicks(4; k_max = 5))
        bins = range(prange..., length = 25)
        ziggurat!(ax, gsol; normalization = :pdf, bins,
                  color = (cornflowerblue, 0.8),
                  label = "Empirical (Δ = $(round(accuracy, digits=2)))")
        lines!(ax, xs, Density(gS).(xs); color = :crimson, label = "Target")
        l = axislegend(ax; position = :lt, orientation = :horizontal)
        reverselegend!(l)
    end

    # * Heatmaps
    begin # * Load data
        η = gS.p[1].η
        γ = gS.p[1].γ
        file = datadir("bFNS_sweep", "unimodal_γ=$(γ)_η=$(η).jld2")
        data = load(file)
        diffusion_exponents = data["diffusion_exponent"][η = At(η), γ = At(γ)]
        spectral_exponents = data["spectral_exponent"][η = At(η), γ = At(γ)]
        accuracy = data["accuracy"][η = At(η), γ = At(γ)]
        ma = Dropdims(mean)(diffusion_exponents, dims = Obs)
        ms = Dropdims(mean)(spectral_exponents, dims = Obs)
        macc = Dropdims(mean)(accuracy, dims = Obs)
    end

    begin # * Plot MAD exponent heatmap
        ax = Axis(f2[2, :][2, 1];
                  xlabel = "α", ylabel = "β",
                  xgridvisible = false, ygridvisible = false,
                  backgroundcolor = :gray88,
                  limits = ((1.2, 2.0), (0.2, 1.0)))
        # p = heatmap!(ax, ma)
        p = contourf!(ax, ma; levels = range(0.25, 0.75, length = 10),
                      colormap = darksunset, extendhigh = :auto, extendlow = :auto)
        contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash)
        # contour!(ax, ms; color = :white, levels = [-1.9], linestyle = :dash)
        Colorbar(f2[2, :][1, 1], p; vertical = false, label = "Diffusion exponent")
        # Label(f2[2, :][1, 1][1, 2, Top()], "a", font = :regular)
        # ! Add annotation and brownian spectrum
    end

    begin # * Plot PSD exponent heatmap
        ax = Axis(f2[2, :][2, 2];
                  xlabel = "α", ylabel = "β",
                  xgridvisible = false,
                  ygridvisible = false, backgroundcolor = :gray88,
                  limits = ((1.2, 2.0), (0.2, 1.0)))
        p = contourf!(ax, ms; levels = range(-2, -1.0, length = 10), extendhigh = :auto,
                      extendlow = :auto,
                      colormap = lightsunset)
        contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash)
        # contour!(ax, ms; color = :white, levels = [-1.9], linestyle = :dash)
        Colorbar(f2[2, :][1, 2], p; vertical = false, label = "Spectral exponent")
        # Label(f2[2, :][1, 2][1, 2, Top()], "b", font = :regular)
    end

    begin # * Plot Aaccuracy heatmap
        ax = Axis(f2[2, :][2, 3];
                  xlabel = "α", ylabel = "β",
                  xgridvisible = false, ygridvisible = false,
                  backgroundcolor = :gray88,
                  limits = ((1.2, 2.0), (0.2, 1.0)))
        p = contourf!(ax, log10.(macc))
        # p = heatmap!(ax, log10.(macc))
        contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash)
        rs = x -> rich("10", superscript(string(round(x, digits = 2))))
        Colorbar(f2[2, :][1, 3], p; ticks = WilkinsonTicks(4),
                 tickformat = x -> rs.(x), vertical = false, label = "Sampling accuracy")
    end

    display(f2)
    wsave(plotdir("model_summary", "unimodal_summary.pdf"), f2)
end

begin # * Bimodal figure
    f3 = SixPanel()
    𝜋 = test_density(:bimodal)
    b_ps = (shared_params..., γ = 0.02,
            𝜋)

    bS = bFNS(; b_ps..., noise)
    bsol = @time solve(bS) |> Timeseries |> eachcol |> first
    bsol = rectify(bsol, dims = 𝑡; tol = 1)
    bsol = bsol[𝑡 = tmin .. tmax] # Remove transient
    bsol = set(bsol, 𝑡 => times(bsol) ./ 1000) # To seconds

    begin # * First plot the effective potential
        prange = (-1.1, 1.1)
        _xs = range(prange..., length = 1000)
        xs = _xs[1:10:end]

        V = potential(Density(bS))
        Vs = V.(xs)
        Vs = Vs .- minimum(Vs)
        # Vs = Vs ./ Vs[xs .== 0]

        Ṽ = effective_potential(bS)
        Ṽs = Ṽ(_xs)
        idxs = indexin(xs, _xs) |> Vector{Int}
        Ṽs = Ṽs[idxs]
        Ṽs .-= minimum(Ṽs)
        # Ṽs .= Ṽs ./ Ṽs[xs .== 0]

        ax = Axis(f3[1, 1], xlabel = "x", ylabel = "V(x)",
                  limits = (extrema(xs), (-0.5, maximum(Vs))), title = "Potential function")
        lines!(ax, xs, Vs; color = :cornflowerblue, label = "Potential")
        lines!(ax, xs, Ṽs; linestyle = :dash, color = :crimson,
               label = "Effective potential")
        axislegend(ax; position = :ct)
    end

    begin # * Plot distribution
        accuracy = samplingaccuracy(bsol, Density(bS);
                                    domain = only(FractionalNeuralSampling.domain(b_ps.boundaries)))

        ax = Axis(f3[1, 2], xlabel = "x", ylabel = "𝜋(x)", limits = (extrema(xs), (0, 2.0)),
                  title = "Distribution", yticks = WilkinsonTicks(4; k_max = 5))
        bins = range(prange..., length = 25)
        ziggurat!(ax, bsol; normalization = :pdf, bins,
                  color = :gray,
                  label = "Empirical (Δ = $(round(accuracy, sigdigits=1)))")
        lines!(ax, xs, Density(bS).(xs); color = :crimson, label = "Target")
        l = axislegend(ax; position = :lt, orientation = :horizontal)
        reverselegend!(l)
    end

    begin # * MAD
        τs = logrange(step(bsol), 1.0, length = 100) # Seconds
        mads = madev(bsol, τs)

        ax = Axis(f3[2, 1], xlabel = "Time lag (s)", ylabel = "MAD",
                  xscale = log10, yscale = log10, title = "Mean absolute deviation")

        lines!(ax, τs, mads; label = "Bimodal")
        frange = 1e-4 .. 1e-2
        m = fit(MAPPLE, mads[𝑡 = frange]; peaks = 0, components = 1)
        fit!(m, mads[𝑡 = frange])
        # m.params.components
        β = bS.p[1].β
        α = bS.p[1].α
        label = "a = $(round(m.params.components[1].β, digits=2))"
        lines!(ax, τs[τs .∈ [frange]], predict(m, τs[τs .∈ [frange]]); label,
               color = :crimson,
               linestyle = :dash)
        # ! Add annotation and brownian spectrum
        text!(ax, [1], [0]; space = :relative,
              text = "α=$(b_ps.α) \nβ=$(b_ps.β) \nγ=$(b_ps.γ) \nη=$(b_ps.η) \n ",
              align = (:right, :bottom))
        l = axislegend(ax; position = :lt)
        reverselegend!(l)
    end
    begin # * Power spectral density
        bpsd = spectrum(bsol .- mean(bsol), 3.0; padding = 1000)
        bpsd = bpsd[𝑓 = eps() .. 1000]

        ax = Axis(f3[2, 2]; xtickformat = x -> string.(round.(Int, x)),
                  title = "Power spectral density")
        plotspectrum!(ax, bpsd; label = "Bimodal")
        logs = logsample(ustripall(bpsd)[𝑓 = 10 .. 1000])
        m = fit(MAPPLE, logs; peaks = 0, components = 1)
        fit!(m, logs)
        β = bS.p[1].β
        α = bS.p[1].α
        label = "b = $(round(m.params.components[end].β, digits=2))"
        lines!(ax, lookup(logs, 1), predict(m, lookup(logs, 1)); color = :red,
               linewidth = 2, label, linestyle = :dash)
        l = axislegend(ax; position = :lb)
        reverselegend!(l)
        text!(ax, [1], [1]; space = :relative,
              text = "\nα=$(b_ps.α) \nβ=$(b_ps.β) \nγ=$(b_ps.γ) \nη=$(b_ps.η) ",
              align = (:right, :top))
        # !! Add annotation and brownian spectrum
        ax.xlabel = "Frequency (Hz)"
    end

    # * Heatmaps
    begin # * Load data
        η = bS.p[1].η
        γ = bS.p[1].γ
        file = datadir("bFNS_sweep", "bimodal_γ=$(γ)_η=$(η).jld2")
        data = load(file)
        diffusion_exponents = data["diffusion_exponent"][η = At(η), γ = At(γ)]
        spectral_exponents = data["spectral_exponent"][η = At(η), γ = At(γ)]
        accuracy = data["accuracy"][η = At(η), γ = At(γ)]
        ma = Dropdims(mean)(diffusion_exponents, dims = Obs)
        ms = Dropdims(mean)(spectral_exponents, dims = Obs)
        macc = Dropdims(mean)(accuracy, dims = Obs)
    end

    begin # * Plot MAD exponent heatmap
        ax = Axis(f3[3, 1:2][1, 1];
                  xlabel = "α", ylabel = "β",
                  xgridvisible = false, ygridvisible = false,
                  backgroundcolor = :gray88,
                  limits = ((1.2, 2.0), (0.2, 1.0)))
        # p = heatmap!(ax, ma)
        p = contourf!(ax, ma; levels = range(0.25, 0.75, length = 10),
                      colormap = darksunset, extendhigh = :auto, extendlow = :auto)
        contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash)
        # contour!(ax, ms; color = :white, levels = [-1.9], linestyle = :dash)
        scatter!(ax, [bS.p[1].α], [bS.p[1].β]; color = cucumber, markersize = 10,
                 strokecolor = :white, strokewidth = 1)
        Colorbar(f3[3, 1:2][0, 1], p; vertical = false, label = "Diffusion exponent")
        # Label(f2[2, :][1, 1][1, 2, Top()], "a", font = :regular)
        # ! Add annotation and brownian spectrum
    end

    begin # * Plot PSD exponent heatmap
        ax = Axis(f3[3, 1:2][1, 2];
                  xlabel = "α", ylabel = "β",
                  xgridvisible = false,
                  ygridvisible = false, backgroundcolor = :gray88,
                  limits = ((1.2, 2.0), (0.2, 1.0)))
        p = contourf!(ax, ms; levels = range(-2, -1.0, length = 10), extendhigh = :auto,
                      extendlow = :auto,
                      colormap = lightsunset)
        contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash)
        # contour!(ax, ms; color = :white, levels = [-1.9], linestyle = :dash)
        scatter!(ax, [bS.p[1].α], [bS.p[1].β]; color = cucumber, markersize = 10,
                 strokecolor = :white, strokewidth = 1)
        Colorbar(f3[3, 1:2][0, 2], p; vertical = false, label = "Spectral exponent")
        # Label(f2[2, :][1, 2][1, 2, Top()], "b", font = :regular)
    end

    begin # * Plot Aaccuracy heatmap
        ax = Axis(f3[3, 1:2][1, 3];
                  xlabel = "α", ylabel = "β",
                  xgridvisible = false, ygridvisible = false,
                  backgroundcolor = :gray88,
                  limits = ((1.2, 2.0), (0.2, 1.0)))
        p = contourf!(ax, log10.(macc))
        # p = heatmap!(ax, log10.(macc))
        contour!(ax, ma; color = :black, levels = [0.5], linestyle = :dash)
        scatter!(ax, [bS.p[1].α], [bS.p[1].β]; color = cucumber, markersize = 10,
                 strokecolor = :white, strokewidth = 1)
        rs = x -> rich("10", superscript(string(round(x, digits = 2))))
        Colorbar(f3[3, 1:2][0, 3], p;
                 tickformat = x -> rs.(x), vertical = false, label = "Sampling accuracy")
    end

    wsave(plotdir("model_summary", "bimodal_summary.pdf"), f3)
    display(f3)
end
