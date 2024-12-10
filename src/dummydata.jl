# some functions to get dummy histogram data

using Plots
using Distributions

function get_grid()
    binwidth = 10
    # ! Note to self: x might be confusing, choose other variable name
    xmin = 5
    xmax = 250
    # midpoints of the bins
    x = xmin:binwidth:xmax
    xl = xmin-binwidth/2:binwidth:xmax - binwidth/2
    xu = xmin+binwidth/2:binwidth:xmax + binwidth/2
    x, xl, xu
end

"""
Get some dummy data.
This is normalised histogram data of some truncated normal distribution.
x defines the midpoints of the bins (i.e. for a physical interpretation, think the equivalent particle diameter).
p defines the probability mass of each bin.
sum(p) should be approximately 1.
"""
function dummydata_1(;mu=150, sigma=20)
    x, xl, xu = get_grid()

    l = xl[1]
    u = xu[end]
    d = truncated(Normal(mu, sigma), l, u)

    p = cdf.(d, xu) - cdf.(d, xl)

    @assert sum(p) ≈ 1

    x, p
end

"""
Get some dummy data, part 2: the bimodal one.
This is normalised histogram data of mixture model of 2 truncated normal distributions.
x defines the midpoints of the bins (i.e. for a physical interpretation, think the equivalent particle diameter).
p defines the probability mass of each bin.
sum(p) should be approximately 1.
"""
function dummydata_2(;
    mu_1=100, sigma_1=50, mu_2=200, sigma_2=20, p_1=0.5, p_2=0.5
)
    x, xl, xu = get_grid()

    l = xl[1]
    u = xu[end]
    d = MixtureModel([
        truncated(Normal(mu_1, sigma_1), l, u),
        truncated(Normal(mu_2, sigma_2), l, u),
        ], [p_1, p_2])

    p = cdf.(d, xu) - cdf.(d, xl)

    @assert sum(p) ≈ 1

    x, p
end