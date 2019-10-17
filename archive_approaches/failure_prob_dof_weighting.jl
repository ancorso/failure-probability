using Statistics


# Computes the failure probability from the state at the root of the tree
function failure_prob(tree; s=1, dof_weighting = false, α = 0)
    sa_children = tree.children[s]
    if isempty(sa_children)
        warning("There were no state-action nodes in the tree, returning 0")
        return 0., 0.
    end
    failure_prob_sa_children(sa_children, tree, dof_weighting = dof_weighting, α = α)
end

# Returns the failure probability averaging across a vector of sa nodes
function failure_prob_sa_children(sa_children, tree, sa = nothing; dof_weighting = false, α = 0)
    # If it does have children, then we can compute the estimates of each of the children
    Nc = length(sa_children)
    E_fail, var_fail = Array{Float64}(undef, Nc), Array{Float64}(undef, Nc)
    E_weight_sum, var_weight_sum = 0, 0
    for ci in 1:Nc
        c = sa_children[ci]
        efc, vfc = failure_prob_sanode(c, tree)
        w = tree.ρ[c]

        # Weight the sample by some value between the number of independent and total samples
        if dof_weighting
            N_ind, N_dep = length(tree.children[tree.transitions2[c][1][1]]) + 1, tree.n[c]
            new_weight = N_ind + α*(N_dep - N_ind)
            E_weight_sum += new_weight
            var_weight_sum += new_weight*new_weight
            w *= new_weight
        end
        E_fail[ci] =  w*efc
        var_fail[ci] = w*w*vfc
    end
    if sa != nothing
        push!(E_fail, tree.fail_sample[sa])
        push!(var_fail, 0)
        if dof_weighting
            E_weight_sum += 1
            var_weight_sum += 1
        end
    end
    if dof_weighting
        E_fail ./ E_weight_sum
        var_fail ./ var_weight_sum
    end

    # Now compute the mean and variance of the estimate
    E = mean(E_fail)

    V = mean(var_fail)/Nc + (length(E_fail) > 1 ? var(E_fail) / Nc : 0.)

    E, V
end


# Returns the failure probability of an sa node in the tree
function failure_prob_sanode(sa, tree)
    # First check if there have been any transitions from this sanode, if not then return its val
    if tree.n_a_children[sa] == 0
        return tree.q2[sa], 0
    end

    # In state s, taking action a, gives us state sp
    sp = tree.transitions2[sa][1][1]

    # if sp does not have any children then it does not have an estimate for q, so take the estimate of the sa node
    sa_children = tree.children[sp]
    if isempty(sa_children)
        return tree.q2[sa], 0
    end

    # Otherwise compute the average failure probability of the children
    failure_prob_sa_children(sa_children, tree, sa)
end



