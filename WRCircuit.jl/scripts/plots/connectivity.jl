#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using DrWatson
DrWatson.@quickactivate
using WRCircuit
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Parameters
    N = 100 # ? Number of neurons
    γ = 4 # ? Ratio of excitatory to inhibitory neurons
    g = 4.0 # ? Ratio of excitatory to inhibitory synaptic strengths
    Ne = γ * N / (γ + 1) # ? Number of excitatory neurons
    Ni = N / (γ + 1) # ? Number of inhibitory neurons
    α = 2.0 # ? Tail index of
end

using Random
using SparseArrays
using Distributions
using StatsBase
using Statistics

"""
    prune_and_hebb!(A, edge_inds, p, num_prunes, E)

1) Randomly picks `num_prunes` edges to prune, sums the pruned weights.
2) Draws how many of these pruned weights get "Hebbian" increments (binomial),
   chosen via weighted sampling `A`.
3) The remainder are re-added at random.
4) Increments `A` in-place.
"""
function prune_and_hebb!(A, edge_inds, p, num_prunes)
    inds_remove = sample(edge_inds, num_prunes, replace = false)
    s_temp_int = round(Int, sum(view(A, inds_remove)))
    A[inds_remove] .= 0
    s_Hebb = rand(Binomial(s_temp_int, p))
    # weights_vec = A[edge_inds] .+ eps() |> Weights
    weights = Weights(view(A, edge_inds))
    inds_inc_Hebb = sample(edge_inds, weights, s_Hebb)
    inds_inc_rand = sample(edge_inds, s_temp_int - s_Hebb, replace = true)
    A[inds_inc_Hebb] .+= 1
    A[inds_inc_rand] .+= 1
    return
end

"""
    compute_heterogeneity(conn_strengths)

Given a vector of positive connection strengths, returns a tuple:
- `svals`: sorted unique strengths
- `scounts`: corresponding histogram bin counts
- `het`: the "heterogeneity" measure (like the original MATLAB code)
"""
function compute_heterogeneity(conn_strengths)
    isempty(conn_strengths) && return ([], [], 0)

    unique_vals = unique(conn_strengths)
    sort!(unique_vals)
    bin_edges = vcat(unique_vals, [Inf])
    h = fit(Histogram, conn_strengths, bin_edges)

    svals = unique_vals
    scounts = collect(h.weights)
    sum_cts = sum(scounts)
    if sum_cts == 0
        return (svals, scounts, 0)
    end

    p_vec = scounts ./ sum_cts
    n = length(svals)
    # Matrix of pairwise differences: svals[i] - svals[j]
    D = reshape(svals, n, 1) .- reshape(svals, 1, n)
    # Outer product of probabilities: p[i]*p[j]
    P = reshape(p_vec, n, 1) .* reshape(p_vec, 1, n)

    avg_strength = mean(conn_strengths)
    het = 0.5 * sum(abs.(D) .* P) / avg_strength
    return (svals, scounts, het)
end

"""
    hebbian_heavy_tails(N; p=0.5, s=1.0)

Simulate a directed activity-independent model of Hebbian-like synaptic updates
on a network of `N` neurons. The optional parameters:
- `p=0.5`: probability of Hebbian growth per pruned synapse.
- `s=1.0`: average connection strength scale (controls how many edges are initially set).
- `maxiters=1000`: number of iterations to run the simulation.
"""
function hebbian_heavy_tails(N; p = 0.5, s = 1.0, maxiters = 1000)
    # 1) Basic parameters
    E = N * (N - 1)                # Number of possible directed edges
    num_syn = Int(round(s * E))
    # num_samples = 100
    # num_updatesPerSample = 100
    # num_prunesPerUpdate = Int(ceil(E / num_updatesPerSample))
    num_prunes = Int(ceil(E / 100))
    # num_updatesBurn = 50 * num_updatesPerSample

    # # Arrays for results
    # s_values = Vector{Vector}(undef, num_samples)
    # s_counts = Vector{Vector}(undef, num_samples)
    # density = zeros(num_samples)
    # heterogeneity = zeros(num_samples)

    A = zeros(Int16, N, N)
    edge_inds = [c for c in CartesianIndices(A) if c[1] != c[2]]
    inds_sample = sample(edge_inds, num_syn, replace = true)
    [A[i] += 1 for i in inds_sample]

    weights = Weights(view(A, edge_inds))

    for _ in 1:maxiters
        inds_remove = sample(edge_inds, num_prunes, replace = false)
        s_temp_int = round(Int, sum(view(A, inds_remove)))
        A[inds_remove] .= 0
        s_Hebb = rand(Binomial(s_temp_int, p))
        # weights_vec = A[edge_inds] .+ eps() |> Weights
        weights.sum = sum(weights.values) # Because we are using a view we need to update this manually
        inds_inc_Hebb = sample(edge_inds, weights, s_Hebb)
        inds_inc_rand = sample(edge_inds, s_temp_int - s_Hebb, replace = true)
        A[inds_inc_Hebb] .+= 1
        A[inds_inc_rand] .+= 1
    end

    # @info "Starting burn-in phase"
    # for _ in 1:num_updatesBurn
    #     prune_and_hebb!(A, edge_inds, p, num_prunesPerUpdate)
    # end

    # @info "Starting sampling phase"
    # for i in 1:num_samples
    #     # (a) Perform updates
    #     for _ in 1:num_updatesPerSample
    #         prune_and_hebb!(A, edge_inds, p, num_prunesPerUpdate)
    #     end
    #     # conn_strengths = A[A .> 0]
    #     # density[i] = length(conn_strengths) / E
    #     # (vals_i, counts_i, het_i) = compute_heterogeneity(conn_strengths)
    #     # s_values[i] = vals_i
    #     # s_counts[i] = counts_i
    #     # heterogeneity[i] = het_i
    # end
    return A # ,
    #    (s_values = s_values,
    #     s_counts = s_counts,
    #     density = density,
    #     heterogeneity = heterogeneity)
end

begin
    A = hebbian_heavy_tails(N; p = 1, s = 5.0, maxiters = 5)
end
begin # * Plot weight distribution
    f = Figure()
    ax = Axis(f[1, 1]; yscale = log10,
              xtickformat = x -> ["10^$x" for x in x])
    hist!(ax, log10.(A[A .> 0]); bins = 50)
    f
end
