"""

The [Gumbel Softmax](https://arxiv.org/pdf/1611.01144.pdf) distribution is a reparametrization trick to enable
differentiating categorical variables.

It is controlled by two parameters:
    (1) π, a vector of log-probabilities ("logits")
    (2) τ, a temperature; as τ → 0, the GS distribution approaches categorical.
"""

function softmax(logits)
    y = exp.(logits)
    return y ./ sum(y)
end

struct GumbelSoftmax{T} <: ContinuousMultivariateDistribution
    logits::Vector{T}       # Probabilities
    temperature::T # Temperatures
    # Derived fields
    logPdfConst::T # log-pdf constant term

    function GumbelSoftmax{T}(logits::Vector{T}, temp_::S) where {T,S} 
        k = length(logits)
        kf = convert(T,k)
        temp = convert(T, temp_)
        new{T}(log.(softmax(logits)), temp, lgamma(kf) + (kf - one(T))*log(temp))
    end
end

GumbelSoftmax(p::Vector{T}, t) where {T} = GumbelSoftmax{T}(p, t)

length(g::GumbelSoftmax) = length(g.logits)

Base.show(io::IO, g::GumbelSoftmax) = show(io, g, (:probs, :temperature))

# Evaluation
function insupport(g::GumbelSoftmax, x::Vector{T}) where T
    return length(g) == length(x)
end

function _logpdf(g::GumbelSoftmax, y::Vector)
    k = length(g.logits) 

    ly = log.(y)
    T1 = -k*log(sum(exp.(g.logits-g.temperature*ly)))
    T2 = sum(g.logits)
    T3 = -(1 + g.temperature)*sum(ly)
    return g.logPdfConst + T1 + T2 + T3
end

# Sampling
base_dist = Gumbel(0, 1)
function _rand!(rng::AbstractRNG,
                g::GumbelSoftmax{T},
                x::Vector{T}) where T

    rand!(rng, x)
    sx::T = zero(T)
    for i in 1:length(x)
        x[i] = -log(-log(x[i]))
        x[i] = exp( (x[i] + g.logits[i])/g.temperature )
        sx += x[i]
    end
    for i in 1:length(x)
        x[i] = x[i] / sx
    end
    return x
end

