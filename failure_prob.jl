using Statistics

# Rollout function that tracks a "heuristic reward" for search and a "true reward"
# for computing probability estimates
function double_rollout(mdp, s, depth)
    tot_hr, tot_tr, mul = 0, 0, discount(mdp)
    actions = []
    while !isterminal(mdp, s)
        ract, ρ = random_action(mdp, s, nothing)
        push!(actions, ract)
        sp, hr, tr = generate_sr(nothing, mdp, s, ract, Random.GLOBAL_RNG)
        tot_hr += mul*hr
        tot_tr += mul*tr
        mul *= discount(mdp)
        s = sp
    end
    return tot_hr, tot_tr, actions
end


# Computes the failure probability from the state at the root of the tree
function failure_prob(tree; s=1)
    sa_children = tree.children[s]
    if isempty(sa_children)
        warning("There were no state-action nodes in the tree, returning 0")
        return 0., 0.
    end
    failure_prob_sa_children(sa_children, tree)
end

# Returns the failure probability averaging across a vector of sa nodes
function failure_prob_sa_children(sa_children, tree, sa = nothing)
    # If it does have children, then we can compute the estimates of each of the children
    Nc = length(sa_children)
    E_fail, var_fail = Array{Float64}(undef, Nc), Array{Float64}(undef, Nc)
    for ci in 1:Nc
        c = sa_children[ci]
        efc, vfc = failure_prob_sanode(c, tree)
        w = tree.ρ[c]
        E_fail[ci] =  w*efc
        var_fail[ci] = w*w*vfc
    end
    if sa != nothing
        push!(E_fail, tree.fail_sample[sa])
        push!(var_fail, 0)
    end
    E = mean(E_fail)

    if length(E_fail) == 1
        V = (mean(var_fail))/Nc
    else
        V = (var(E_fail) + mean(var_fail))/Nc
    end

    return E, V
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


# TODO: Can I get rid of this?
# Get the results of the simulation from the specified state
function failure_rollout(s_in, mdp, sims_per_rollout)
    counts = []
    for i = 1:sims_per_rollout
        s, weight = s_in, 1
        while !isterminal(mdp, s)
            sp, _ = generate_sr(mdp, s, random_action(mdp,s_in,nothing), Random.GLOBAL_RNG)
            weight *= rollout_weight(mdp, s, sp)
            s = sp
        end
        push!(counts, weight*in_E(s[end], mdp))
    end
    mean(counts), var(counts)
end

# TODO: Can I get rid of this?
get_actual_state(mdp, s) = s