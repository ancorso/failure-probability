using Statistics
using Distributions

# Function for summing beta
function sum_betas(weights, betas)
    if length(betas) == 1
        return betas[1]
    end
    # if sum(weights) != 1
    #     println("sum of weights: ", sum(weights))
    # end
    # @assert sum(weights) == 1
    S = sum(weights.^2 .* var.(betas))
    E = sum(weights .* mean.(betas))
    F = E/(1-E)
    f = F/(S*(1+F)^3) - F/(1+F)
    e = F*f
    if !(e > 0 && f > 0)
        println("S: ", S, " E: ", E, " F: ", F, " f: ", f, " e: ", e)
        println("Betas: ", betas)
    end
    Beta(e,f)
end

# Computes the failure probability from the state at the root of the tree
function failure_prob_beta(tree; s=1, prior = Beta(1,1))
    sa_children = tree.children[s]
    if isempty(sa_children)
        warning("There were no state-action nodes in the tree, returning 0")
        return prior
    end
    failure_prob_sa_children_beta(sa_children, tree, prior = prior)
end

# Returns the failure probability averaging across a vector of sa nodes
function failure_prob_sa_children_beta(sa_children, tree, sa = nothing; prior = Beta(1,1))
    # If it does have children, then we can compute the estimates of each of the children
    Nc = length(sa_children)
    distributions = [failure_prob_sanode_beta(c, tree, prior = prior) for c in sa_children]
    weights = ones(Nc)./Nc
    sum_betas(weights, distributions)
end


# Returns the failure probability of an sa node in the tree
function failure_prob_sanode_beta(sa, tree; prior = Beta(1,1))
    # First check if there have been any transitions from this sanode, if not then return its val
    if tree.n_a_children[sa] == 0
        @assert tree.q2[sa] == 0. || tree.q2[sa] == 1.0
        return Beta(tree.q2[sa] + prior.α, 1-tree.q2[sa] + prior.β)
    end

    # In state s, taking action a, gives us state sp
    sp = tree.transitions2[sa][1][1]

    # if sp does not have any children then it does not have an estimate for q, so take the estimate of the sa node
    sa_children = tree.children[sp]
    if isempty(sa_children)
        @assert tree.q2[sa] == 0. || tree.q2[sa] == 1.0
        return Beta(tree.q2[sa] + prior.α, 1-tree.q2[sa] + prior.β)
    end

    # Otherwise compute the average failure probability of the children
    failure_prob_sa_children_beta(sa_children, tree, sa, prior = prior)
end

