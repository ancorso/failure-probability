using Statistics


# Computes the failure probability from the state at the root of the tree
function failure_prob_iid(tree; s=1)
    sa_children = tree.children[s]
    samples = Float64[]
    weights = Float64[]
    if isempty(sa_children)
        warning("There were no state-action nodes in the tree, returning 0")
        return samples, weights
    end
    failure_prob_sa_children_iid(sa_children, tree, 1., 1., samples, weights)
    samples, weights
end

# Returns the failure probability averaging across a vector of sa nodes
function failure_prob_sa_children_iid(sa_children, tree, weight, is_weight, samples, weights, sa = nothing)
    # If it does have children, then we can compute the estimates of each of the children
    Nc = length(sa_children)
    if sa != nothing
        Nc += 1
        push!(samples, tree.fail_sample[sa]*is_weight)
        push!(weights, weight / Nc)
    end
    for c in sa_children
        failure_prob_sanode_iid(c, tree, weight / Nc, is_weight*tree.ρ[c], samples, weights)
    end
end


# Returns the failure probability of an sa node in the tree
function failure_prob_sanode_iid(sa, tree, weight, is_weight, samples, weights)
    # First check if there have been any transitions from this sanode, if not then return its val
    if tree.n_a_children[sa] == 0
        push!(samples, tree.q2[sa]*is_weight)
        push!(weights, weight)
        return
    end

    # In state s, taking action a, gives us state sp
    sp = tree.transitions2[sa][1][1]

    # if sp does not have any children then it does not have an estimate for q, so take the estimate of the sa node
    sa_children = tree.children[sp]
    if isempty(sa_children)
        push!(samples, tree.q2[sa]*is_weight)
        push!(weights, weight)
        return
    end

    # Otherwise compute the average failure probability of the children
    failure_prob_sa_children_iid(sa_children, tree, weight, is_weight, samples, weights, sa)
end

