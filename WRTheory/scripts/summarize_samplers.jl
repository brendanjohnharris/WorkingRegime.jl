#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
@quickactivate "WRTheory"
using WRTheory
WRTheory.@preamble()

begin # * Pick a Sampler
    noisestrength = (:η => 1.0,)
    tspan = 1000.0
    dt = 0.01
    S = OLE(; tspan, dt, alg = EM(), noisestrength...)
end

begin # summarize_sampler
    out = Dict()
    p = 100 # Space between windows
    etas = [0.1, 0.5, 1.0, 2.0]
    τs = round.(Int, logrange(1, 100, 20)) .÷ S.kwargs[:dt] .|> Int

    begin # * Accuracy on single well
        @info "Computing single well accuracy"
        𝜋 = Normal(0, 1) |> Density
        xs = map(Dim{:η}(etas)) do η
            sol = solve(S(; η, 𝜋)) |> Timeseries |> eachcol |> first
            sol = rectify(sol, dims = 𝑡; tol = 1)
            sol = set(map(Float32, sol), 𝑡 => map(Float32, times(sol)))
        end

        accuracy = map(Chart(Threaded(), ProgressLogger()), xs) do x
            y = samplingaccuracy(x, 𝜋, τs; p)  # / sqrt(samplingpower(x, dt))
            ToolsArray(y, 𝑡(τs))
        end |> stack
        out[:onewell_acc] = accuracy
    end

    begin # * Accuracy on a double well
        @info "Computing double well accuracy"
        𝜋 = MixtureModel([Normal(-2, 1), Normal(2, 1)]) |> Density
        xs = map(Dim{:η}(etas)) do η
            sol = solve(S(; η, 𝜋)) |> Timeseries |> eachcol |> first
            sol = rectify(sol, dims = 𝑡; tol = 1)
            sol = set(map(Float32, sol), 𝑡 => map(Float32, times(sol)))
        end

        accuracy = map(Chart(Threaded(), ProgressLogger()), xs) do x
            y = samplingaccuracy(x, 𝜋, τs; p)  # / sqrt(samplingpower(x, dt))
            ToolsArray(y, 𝑡(τs))
        end |> stack
        out[:twowell_acc] = accuracy
    end
end

begin # * Plot
    f = TwoPanel()
    ax = Axis(f[1, 1], xlabel = "τ", ylabel = "Sampling accuracy", xscale = log10,
              title = "Single Well")
    traces!(ax, median.(out[:onewell_acc]))

    ax = Axis(f[1, 2], xlabel = "τ", ylabel = "Sampling accuracy", xscale = log10,
              title = "Double Well")
    traces!(ax, median.(out[:twowell_acc]))

    display(f)
end
