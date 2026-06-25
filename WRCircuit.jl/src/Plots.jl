using CairoMakie
import ProgressLogging: @withprogress, @logprogress
import Statistics: quantile

function infer_geometry(A, dx)
    ns = lookup(A, Neuron)
    n = sqrt(length(ns)) |> Int
    ns = WRCircuit.python_reshape(ns, n, n)
    δx = dx / n
    x = range(δx / 2, dx, step = δx)
    idxs = ToolsArray(
        Matrix{eltype(ns)}(undef, n, n),
        (DimensionalData.X(x), DimensionalData.Y(x))
    )
    idxs .= ns # Assume Neurons are just flattened
    positions = Iterators.product(x, x)
    positions = ToolsArray(collect(positions)[:], dims(A, Neuron))
    return positions, idxs
end

function animate_rates(
        X, dx; filename,
        figure = (), axis = (), record = (; framerate = 12),
        colormap = cgrad([:transparent, :crimson]), # For rates
        colorrange = (0, quantile(vec(ustrip(collect(X))), 0.99)),
        highclip = :crimson
    )
    mkpath(dirname(filename))
    positions, idxs = infer_geometry(X, dx)
    color = Observable(zeros(length(positions)))

    f = Figure(; figure...)
    ax = Axis(f[1, 1]; axis...)
    p = scatter!(
        ax, collect(Point2f.(positions)); markersize = 5, color, colormap,
        colorrange, highclip
    )
    Colorbar(f[1, 2], p; label = "Firing rate (Hz)")
    ts = lookup(X, 𝑡)
    n = length(ts)

    return Makie.record(f, filename, ts; record...) do t
        r = X[𝑡 = At(t)] |> ustrip
        color[] = r
    end
end
