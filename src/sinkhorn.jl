using LinearAlgebra
using ChainRulesCore

function lse(x)
    m = maximum(x)
    return m + log.(sum(exp, x .- m))
end

function lse(x, dims)
    m = maximum(x; dims)
    return m .+ log.(sum(exp, x .- m; dims))
end

entropy(p) = -sum(x->x*log(x), P)

function sinkhorn(C::AbstractMatrix, α::AbstractVector, β::AbstractVector;
    ε=1e-1, maxiter=100, tol=1e-3)
    n, m = size(C)
    f = zero(α)
    g = zero(β)
    fold = zero(α)
    for _ in 1:maxiter
        fold .= f
        for i in 1:n
            f[i] = 0
            for k in 1:m
                @inbounds f[i] += log(β[k]) + g[k] / ε - C[i,k] / ε |> exp
            end
            f[i] = - ε * log(f[i])
        end
        for j in 1:m
            g[j] = 0
            for k in 1:n
                @inbounds g[j] += log(α[k]) + f[k] / ε - C[k,j] / ε |> exp
            end
            g[j] = -ε * log(g[j])
        end
        #f .= -ε .* lse(log.(β)' .+ g' ./ ε .- C ./ ε, 2)[:]
        #g .= -ε .* lse(log.(α) .+ f ./ ε .- C ./ ε, 1)[:]
        #f .= -ε .* (lse(g' ./ ε .- C ./ ε, 2)[:] .- log.(α))
        #g .= -ε .* (lse(f ./ ε .- C ./ ε, 1)[:] .- log.(β))
        if maximum(abs, fold .- f) < tol
            break
        else
            fold .= f
        end
    end
    return α ⋅ f + β ⋅ g, f, g
end

function symmetric_sinkhorn(C::AbstractMatrix, α::AbstractVector;
    ε=1e-1, maxiter=100, tol=1e-3)
    n, m = size(C)
    @assert n==m "`C` has to be symmetric!"
    f = zero(α)
    fold = copy(f)
    for _ in 1:maxiter
        f .= -ε .* lse(log.(α) .+ f ./ ε .- C ./ ε, 1)[:]
        f .= (f .+ fold) ./ 2
        fold .= f
        if maximum(abs, fold .- f) < tol
            break
        else
            fold .= f
        end
    end
    return 2(α⋅f), f
end




optimal_transport(C, α, β; kwargs...) = sinkhorn(C, α, β; kwargs...)[1]

function ChainRulesCore.rrule(::typeof(optimal_transport), C, α, β; kwargs...)
    s, f, g = sinkhorn(C, α, β; kwargs...)
    function OT_pullback(Ȳ)
        C̄ = @thunk Diagonal(α) * exp.((f .+ g' .- C) / ε) * Diagonal(β) .* Ȳ
        ᾱ = @thunk Ȳ' * f
        β̄ = @thunk Ȳ * g
        return NoTangent(), C̄, ᾱ, β̄
    end
    return s, OT_pullback
end


∂OT(C, α, β, f, g, ε) = Diagonal(α) * exp.((f .+ g' .- C) / ε) * Diagonal(β), f, g

C = randn(200, 100) .|> abs2
α = ones(200) / 200
β = ones(100) 
β ./= sum(β)
ε = 1e-1

s, f, g = sinkhorn(C, α, β; ε, tol=1e-3)

P = Diagonal(α) * exp.((f .+ g' .- C) / ε) * Diagonal(β)
sum(P, dims=1)[:] ≈ β
sum(P, dims=2)[:] .- α .|> abs
heatmap(P)