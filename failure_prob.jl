using POMDPs
using MCTS

# Check if a state index is a child in the tree
is_leaf(s, tree) = isempty(tree.children[s])

# Count the number of states in the tree
function num_leaves(tree)
    count = 0
    for s in 1:length(tree.s_labels)
        count += is_leaf(s, tree)
    end
    count
end

# Get the indices of the children (states) of the provided state
function get_children(s, tree)
    cis = tree.children[s]
    child_states = Array{typeof(s), 1}(undef, 0)
    for c in cis
        s = tree.transitions[c][1][1]
        push!(child_states, s)
    end
    return child_states
end

# Get the results of the simulation from the specified state
function failure_rollout(s_in, mdp, sims_per_rollout)
    counts, nA = [], n_actions(mdp)
    for i = 1:sims_per_rollout
        s, weight = s_in, 1
        while !isterminal(mdp, s)
            sp, _ = generate_sr(mdp, s, rand(actions(mdp)), Random.GLOBAL_RNG)
            weight *= nA*transition_prob(s, sp, mdp)
            s = sp
        end
        push!(counts, weight*in_E(s, mdp))
    end
    mean(counts), var(counts)
end

# Compute the probability of failure from a particular statea
function failure_prob(s, tree, mdp, sims_per_rollout)
    if is_leaf(s, tree)
        E_fail, var_fail = failure_rollout(tree.s_labels[s], mdp, sims_per_rollout)
        return E_fail, var_fail
    else
        cs = get_children(s, tree)
        Nc = length(cs)
        E_fail, var_fail = 0, 0
        for c in cs
            gws, gwc = tree.s_labels[s], tree.s_labels[c]
            w = Nc*transition_prob(gws, gwc, mdp)
            E_fail_c, var_fail_c = failure_prob(c, tree, mdp, sims_per_rollout)
            E_fail += w*E_fail_c / Nc
            var_fail += w^2*var_fail_c / Nc^2
        end
        return E_fail, var_fail
    end
end

