using Makie

function summarize_sampler!(f::Figure, S::AbstractSampler; noisestrength)
    return nothing
end

function summarize_sampler(S::AbstractSampler; kwargs...)
    f = SixPanel()
    summarize_sampler!(f, S; kwargs...)
    return f
end
