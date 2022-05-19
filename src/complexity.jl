#=
Created on 19/05/2022 11:30:56
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implements the GAIT metrics from Gallego et al. (2019)
=#

using StatsBase, LinearAlgebra, Kronecker

function normkern!(K)
    v = sqrt.(diag(K))
    K ./= v
    K ./= v'
    return K
end

abstract type AbstractCoupling end

struct CoupingArray{TP,TS} <: AbstractCoupling
    P::TP
    Psum::TS
    function CoupingArray(P::AbstractArray, norm=false)
        Psum = norm ? 1 : sum(P)
        return new{typeof(P),eltype(P)}(P, Psum)
    end
end


# Might need a coupling?, Abstractoupling datatype
# this links different objects to oneother. 
# use weights?

# K is PSD with values in [0, 1]!


# Shannon entropies
H(p::AbstractVector, K) = -dot(p, log.(K * p))
H(P::AbstractArray, Ks...) = -dot(vec(P), log.(⊗(Ks...) * vec(P)))
H(K::AbstractMatrix) = -mean(log, mean(K, dims=2))
H(Ks...) = -mean(log, mean(.*(Ks...), dims=2))

# Reyni entropy
entropy_renyi(p::AbstractVector, K; α) = log(dot(p, (K * p).^(α-1))) / (1-α)
entropy_renyi(P::AbstractVector, Ks...; α) = log(dot(vec(P), (⊗(Ks...) * vec(P)).^(α-1))) / (1-α)
entropy_renyi(K; α) = log(mean(x->x^(α-1), mean(K, dims=2))) / (1-α)
entropy_renyi(Ks...; α) = log(mean(x->x^(α-1), mean(.*(Ks...), dims=2))) / (1-α)


# maybe should be made as Hc((K, L), (θ, C))?
Hc(K, Λ) = H(K, Λ) - H(Λ)  # conditional entropy
Hc(K, Λ, Θ) = H(K, Λ, Θ) - H(Θ)  # to check
MI(K, Λ) = H(K) + H(Λ) - H(K, Λ)  # mutual information
MIc(K, Λ, Θ; α=α) = Hc(K, Θ; α) + Hc(Λ, Θ; α) - Hc(K, Λ, Θ; α)  # conditional mutual information



entropy(Ks...; α=α) = α == 1 ?
	-mean(log, mean(.*(Ks...), dims=2)) :
	log(mean(x->x^(α-1), mean(.*(Ks...), dims=2))) / (1-α)