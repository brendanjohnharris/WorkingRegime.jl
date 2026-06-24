#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
WRTheory.@preamble()
set_theme!(foresight(:physics))

import FractionalNeuralSampling: Density

begin # * Create fractional gaussian noise
    using FFTW
    using Random
    # ! gpt
    function fractional_gaussian_noise(N::Int, H::Float64; rng = Random.GLOBAL_RNG, dt = 1)
        # Validate inputs
        @assert 0<H<1 "Hurst exponent H must be in (0, 1)"
        @assert N>0 "N must be positive"

        # Autocovariance function for fGn
        function gamma_fgn(k, H)
            return 0.5 * (abs(k - 1)^(2H) - 2 * abs(k)^(2H) + abs(k + 1)^(2H))
        end

        # Build the circulant covariance matrix first row
        # We need 2N values for the circulant embedding
        g = zeros(2N)
        g[1] = 1.0  # variance at lag 0

        for k in 1:N
            g[k + 1] = gamma_fgn(k, H)
        end

        # Fill the second half (circulant symmetry)
        for k in 1:(N - 1)
            g[2N - k + 1] = g[k + 1]
        end

        # Compute eigenvalues via FFT
        lambda = real(fft(g))

        # Check for non-negative eigenvalues (should be satisfied for valid H)
        if any(lambda .< -1e-10)
            @warn "Negative eigenvalues detected. Results may be inaccurate."
        end
        lambda = max.(lambda, 0)  # Ensure non-negative

        # Generate white noise in frequency domain
        V1 = randn(rng, N + 1)
        V2 = randn(rng, N - 1)

        # Construct the frequency domain representation
        W = zeros(ComplexF64, 2N)
        W[1] = sqrt(lambda[1] / (2N)) * V1[1]
        W[N + 1] = sqrt(lambda[N + 1] / (2N)) * V1[N + 1]

        for k in 2:N
            W[k] = sqrt(lambda[k] / (4N)) * (V1[k] + im * V2[k - 1])
            W[2N - k + 2] = conj(W[k])
        end

        # Inverse FFT to get the fGn
        Z = real(ifft(W))

        # Return middle N samples
        idxs = (length(Z) ÷ 4 + 1):(length(Z) * 3 ÷ 4)
        Z = Z[idxs]
        Z = (Z .- mean(Z)) ./ std(Z)
        Z = Z .* dt^(H)  # Scale by dt^H to account for time step
        return Z
    end

    # Example usage
    N = 1000000
    H = 0.55  # Persistent noise (long-range positive correlation)

    fgn = fractional_gaussian_noise(N, H)
    s = spectrum(Timeseries(fgn, range(0, step = 0.001, length = N)), 5.0)
    plotspectrum(s)
end

begin # * Generate a sample path and test the distribution is ok
    β = 0.8
    H = 1 - β / 2
    tspan = 1000.00
    dt = 0.01
    η = 1.0
    N = Int(tspan / dt) + 1
    u0 = [0.0]

    ts = range(0, step = dt, length = N)
    noise = cumsum(fractional_gaussian_noise(N, H; dt))
    # noise = cumsum(randn(N)) .* sqrt(dt)
    noise = NoiseGrid(ts, noise)

    𝜋 = MixtureModel([Normal(-1, 0.5), Normal(1, 0.5)]) |> Density #
    S = OLE(; η, u0, 𝜋, tspan, noise, alg = CaputoEM(β, 1000))
    _sol = solve(S; dt)
    sol = _sol |> Timeseries |> eachcol |> first
    sol = rectify(sol, dims = 𝑡, tol = 1)
    #

    f = Figure()
    ax = Axis(f[1, 1])
    xs = -3:0.01:3
    ys = 𝜋.(xs)
    hist!(ax, sol; bins = 50, normalization = :pdf, color = (:crimson, 0.5))
    lines!(ax, xs, ys)

    display(f)
end

begin # * plot spectral exponent
    s = spectrum(sol, 0.1)
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Frequency", ylabel = "Power", xscale = log10,
              yscale = log10)
    s = logsample(s[10:end])[1:(end - 10)] # Remove edge effects
    m = fit(MAPPLE, s; peaks = 0, components = 1)
    fit!(m, s)
    beta = m.params.components.β |> first

    lines!(ax, s; label = "β = $(round(beta, digits = 2))")
    lines!(ax, freqs(s), predict(m, freqs(s)); linestyle = :dash, color = :gray)
    axislegend(ax)
    display(f)
end

# begin
#     etas = 0.2:0.4:2.0
#     dt = 0.01
#     β = 0.6
#     τs = round.(Int, logrange(1, 1000, 10)) .÷ dt .|> Int
#     𝜋 = MixtureModel([Normal(-2, 0.5), Normal(2, 0.5)]) |> Density
#     u0 = [0.0]
#     tspan = 5000.00

#     xs = map(Chart(ProgressLogger(), Threaded()), Dim{:η}(etas)) do η
#         S = OLE(; η, u0, 𝜋, tspan)
#         sol = solve(S, CaputoEM(β, 1000); dt) |> Timeseries |> eachcol |> first
#         return rectify(sol, dims = 𝑡; tol = 1)
#     end

#     accuracy = map(Chart(Threaded(), ProgressLogger()), xs) do x
#         y = samplingaccuracy(x, 𝜋, τs; p = 1000)  #/ sqrt(samplingpower(x, dt))
#         ToolsArray(y, 𝑡(τs))
#     end |> stack
#     accuracy = map(mean, accuracy)

#     _τs = τs * dt # For efficiency
#     efficiency = map(Chart(Threaded(), ProgressLogger()), xs) do x
#         y = samplingefficiency(x, 𝜋, _τs; downsample = 5, p = 1000)
#         ToolsArray(y, 𝑡(_τs))
#     end |> stack
#     efficiency = map(mean, efficiency)
# end

# begin
#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Time lag", ylabel = "Accuracy", xscale = log10)
#     p = traces!(ax, accuracy, linewidth = 2)
#     hlines!(ax, [1.0]; color = :gray, linestyle = :dash)
#     Colorbar(f[1, 2], p; label = "η")
#     display(f)
# end
# begin
#     ts = 1:100
#     vd = map(Chart(Threaded(), ProgressLogger()), xs) do x
#         y = map(ts) do t
#             samplingpower(x[1:t:end])
#         end
#         ToolsArray(y, 𝑡(ts))
#     end |> stack

#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Time step", ylabel = "Sampling power", xscale = log10)
#     p = traces!(ax, vd, linewidth = 2)
#     Colorbar(f[1, 2], p; label = "η")
#     display(f)
# end

# begin
#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Time lag", ylabel = "Efficiency", xscale = log10)
#     p = traces!(ax, efficiency, linewidth = 2)
#     hlines!(ax, [1.0]; color = :gray, linestyle = :dash)
#     Colorbar(f[1, 2], p; label = "η")
#     display(f)
# end
