using KernelFunctions, LinearAlgebra, StatsBase
using JuMP, COSMO
using Plots

include("dummydata.jl")

import Base.Iterators
N = 2 # 5^4 = 625 combinations
mu_1 = range(start=75, stop=125, length=N)
sigma_1 = range(start=10, stop=30, length=N)
mu_2 = range(start=175, stop=225, length=N)
sigma_2 = range(start=20, stop=50, length=N)
# let us keep p_1 and p_2 fixed for now
Z = stack(Iterators.product(mu_1, sigma_1, mu_2, sigma_2), dims=1)

res = map(x -> dummydata_2(mu_1=x[1], sigma_1=x[2], mu_2=x[3], sigma_2=x[4], p_1=0.5, p_2=0.5), eachrow(Z))
# all the same, so we can just take the first one
# 25 bins
bins = res[1][1] |> collect # collecting to ensure it is a vector
# N x 25 Matrix
A = stack(map(x -> x[2], res), dims=1) 



# standardize process settings
Z = standardize(ZScoreTransform, Z, dims=1)

# This hyperparameter should be estimated
σ_RBF_grid = 0.712762951907117
k = with_lengthscale(RBFKernel(), σ_RBF_grid)
# adding some small bias for numerical stability
K = kernelmatrix(k, log10.(bins)) + 0.1*I

# This hyperparameter should be estimated
σ_RBF_procvars = 3
k = with_lengthscale(RBFKernel(), σ_RBF_procvars)
C = kernelmatrix(k, RowVecs(Z)) + 0.05*I

# heatmap(K, color = :viridis)
# heatmap(C, color = :viridis)


# kernel over the output distributions
Q = A * K

λ = 1e-4
# building a model
H = C / (C + λ*I)
F = H * Q
F_loo = (I - Diagonal(H)) \ (F - Diagonal(H) *Q)


function compute_pre_image(F_loo::Matrix{Float64}, K::Matrix{Float64})::Matrix{Float64}
    n_distr, n_classes = size(psd)
    predicted_weights = Array{Float64}(undef, size(psd))
    for i in 1:n_distr
        # COSMO
        model = JuMP.Model(COSMO.Optimizer)
        # Ipopt
        # model = JuMP.Model(Ipopt.Optimizer)
        # no printing to stdout !
        set_silent(model)
        @variable(model, β[1:n_classes] >= 0.0)
        @constraint(model, sum(β) == 1.0)
        @objective(model, Min, sum(β' * K * β) - 2dot(β, F_loo[i, :]))
        optimize!(model)
        predicted_weights[i,:] = JuMP.value.(β)
        # @show termination_status(model)
        # @show primal_status(model) 
        # @show objective_value(model)
    end
    return predicted_weights
end


predicted_weights = compute_pre_image(F_loo, K)
heatmap(predicted_weights, color = :viridis)
# SSE
sum((predicted_weights - psd).^2)


# store figures
# for i in 1:size(A)[1]
#     plot(bins, A[i,:], label="measured", dpi=150)
#     plot!(bins, predicted_weights[i,:], label="predicted")
#     plot!(xscale=:log10, xlabel="particle size", ylabel="volume fraction", legend=:topleft)
#     fn = "prediction_$(exps[i]).png"
#     savefig(fn)
# end

