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

begin # * Parameters
    params = (;
              γ = 0.1,
              η = 0.5,
              #   𝜋 = MixtureModel([Normal(-2, 0.5), Normal(2, 0.5)]) |> Density,
              𝜋 = Density(Normal(0.0, 1.0)),
              domain = -10 .. 10,
              boundaries = PeriodicBox(-5 .. 5),
              u0 = [0.0, 0.0],
              dt = 0.1,
              tspan = 10000.00,)
end
if false # * Little test
    α = 2.0
    β = 0.7
    S = bFNS(; α, β, params..., tspan = 5000.0)
    sol = solve(S) |> Timeseries |> eachcol |> first
    sol = rectify(sol, dims = 𝑡; tol = 1)

    hist(sol; normalization = :pdf, bins = -3:0.1:3)
    lines!(-3:0.1:3, Density(S).(-3:0.1:3), color = :red, linewidth = 2)
    current_figure() |> display

    ws = samplingaccuracy(sol, Density(S), 100:100:5000;
                          domain = params[:boundaries])

    # * Fit exponential
    # Loss function
    x_data = times(ws)
    y_data = mean.(ws)
    function loss(p)
        A, b, tau = p
        pred = A * exp.(-x_data ./ tau) .+ b
        return sum((y_data .- pred) .^ 2)
    end

    result = optimize(loss, [1.0, 0.01, 1000])
    A_fit, b_fit, tau_fit = result.minimizer

    lines(mean.(ws))#; axis = (; xscale = log10, yscale = log10))
    lines!(x_data, A_fit * exp.(-x_data / tau_fit) .+ b_fit, color = :red, linewidth = 2)
    current_figure() |> display
end

begin # * Now sweep across alpha and beta, plotting accuracy at longest time lag
    αs = range(1.2, 2.0, length = 10) |> Dim{:α}
    βs = range(0.2, 1.0, length = 10) |> Dim{:β}
    C = Chart(Threaded(), ProgressLogger(), Iterators.product)
    pac = map(C, αs, βs) do α, β
        H = (1 - β) / 2 + 1 / α
        if !(0 < H < 1)
            return NaN
        end

        S = bFNS(; α, β, params...)
        sol = solve(S) |> Timeseries |> eachcol |> first
        sol = rectify(sol, dims = 𝑡; tol = 1)

        N = round(Int, 1000 / params[:dt])
        ws = samplingaccuracy(sol, Density(S), [N]; domain = params[:boundaries])
        ps1 = samplingpower(sol, params[:dt]; p = 1)
        ps2 = samplingpower(sol, params[:dt]; p = 2)
        # τs = 1:1000
        # τs = τs .÷ params[:dt]
        # τs = round.(Int, τs)
        # ws = samplingaccuracy(sol, Density(S), τs; domain = params[:boundaries])

        # # * Fit exponential
        # x_data = times(ws)
        # y_data = median.(ws)
        # function loss(p)
        #     A, b, tau = p
        #     pred = A * exp.(-x_data ./ tau) .+ b
        #     return sum((y_data .- pred) .^ 2)
        # end

        # result = optimize(loss, [1.0, 0.01, 1000])
        # A_fit, b_fit, tau_fit = result.minimizer

        return only(ws), (ps1, ps2)
    end
    accuracy = first.(pac)
    power = last.(pac)
    power1 = first.(power)
    power2 = last.(power)
end
begin
    # heatmap(log10.(accuracy .|> mean))
    tagsave(datadir("sampling_accuracy.jld2"),
            (@strdict params accuracy power1 power2))
end
if false
    file = datadir("sampling_accuracy.jld2")
    accuracy = load(file, "accuracy")
    power1 = load(file, "power1")
    power2 = load(file, "power2")
    heatmap(log10.(median.(accuracy) ./ power1))
end
