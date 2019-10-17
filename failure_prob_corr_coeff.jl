using Statistics

function action_rollout(mdp, s0, actions)
    s = [s0]
    sum_rewards = 0
    mul = 1
    for a in actions
        s, hr, tr = generate_sr(nothing, mdp, s, a, Random.GLOBAL_RNG)
        sum_rewards += tr*mul
        mul *= discount(mdp)
        isterminal(mdp, s) && break
    end

    while !isterminal(mdp, s)
        randa, ρ = random_action(mdp, s, nothing)
        s, hr, tr = generate_sr(nothing, mdp, s, randa, Random.GLOBAL_RNG)
        sum_rewards += tr*mul
        mul *= discount(mdp)
    end

    s, sum_rewards
end

function get_correlation_coeffs(tree)
    correlation_coeffs = Dict()
    s0_ind = 1
    for aseq in tree.action_sequences
        fill_correlation_coeffs!(tree, s0_ind, aseq, correlation_coeffs)
    end
    correlation_coeffs
end

function fill_correlation_coeffs!(tree, s0_ind, aseq, coeffs_dict)
    state = tree.s_labels[s0_ind][end]

    # Base case for empty sequence
    isempty(aseq) && return

    # Pull out the state-action nodes that originate from the input state, and remove the state action node that is associated with the found failure.
    sa_children = tree.children[s0_ind] # Indicres of all sa node children
    exclude = 0
    try
        exclude = tree.a_lookup[(s0_ind, aseq[1])] # sanode associated with the action sequence of the failure
    catch e
        # If no sa_node is found, then the rest of the sequence was from one of the estimation rollouts and there will be no nodes in the tree associated with the actions
        return
    end
    sa_children = sa_children[sa_children .!= exclude] # remove the node so we don't compare it to itself
    actions = [tree.a_labels[sai] for sai in sa_children] # Get the actions of the other nodes

    # Loop through each and see if the failure rollout still causes a failure if you had chosen another node
    for i in 1:length(sa_children)
        new_a = actions[i]
        sa = sa_children[i]
        reward = action_rollout(mdp_true, state, [new_a, aseq[2:end]...])[2]
        corr_coef = Tuple(sort([exclude, sa]))

        counts = get(coeffs_dict, corr_coef, (0,0))

        # Update the counts
        coeffs_dict[corr_coef] = (counts[1] + reward, counts[2] + 1 - reward)
    end

    # Determine the next state and action sequence to try
    next_s = tree.transitions2[exclude][1][1]
    fill_correlation_coeffs!(tree, tree.transitions2[exclude][1][1], aseq[2:end], coeffs_dict)
end

# Computes the failure probability from the state at the root of the tree
function failure_prob_cc(tree; s=1)
    sa_children = tree.children[s]
    if isempty(sa_children)
        warning("There were no state-action nodes in the tree, returning 0")
        return 0., 0.
    end
    correlation_coeffs = get_correlation_coeffs(tree)
    stored_statistics = fill((NaN,NaN), length(tree.q))
    failure_prob_sa_children_cc(tree, sa_children, [], correlation_coeffs, stored_statistics)
end

function combine_samples(xi, wi, σ2i)
    V1 = sum(wi)
    xbar = sum(xi .* wi) / V1
    var =  sum(σ2i .* wi) / V1
    if  V1 > 1
         var +=  sum(wi .* ( xi .- xbar).^2) / (V1-1)
    end
    var /= V1
    xbar, var
end

sa_children(tree, sa) = isempty(tree.transitions2[sa]) ? [] : tree.children[tree.transitions2[sa][1][1]]

function failure_prob_sa_children_cc(tree, children, siblings, correlation_coeffs, stored_statistics, sa = 0)
    # Inialize the vectors to store samples
    Nc = length(children)
    xi, wi, σ2i = zeros(Nc), zeros(Nc), zeros(Nc)

    # Get the children of the sa node, loop through them and gather samples
    for ci in 1:Nc
        c = children[ci]
        c_siblings = children[children .!= c]
        efc, vfc = failure_prob_sanode_cc(tree, c, c_siblings, correlation_coeffs, stored_statistics)
        xi[ci] = efc
        σ2i[ci] = vfc
        wi[ci] =  1
    end

    # Loop through the siblings, and for each, grab the children and add the samples
    for sib in siblings
        # Get the correlation between nodes (default is 0)
        ρp, ρn = get(correlation_coeffs, Tuple(sort([sib, sa])), (0,1))
        ρ = ρp / (ρn + ρp)

        # If there is correlation then we can count up the samples with the appropriate weight
        if ρ > 0
            try
                sib_children = sa_children(tree, sib)
            catch
                println("sdfljsdf")
                println(tree.transitions2[sib])
                println(tree.transitions2[sib][1])
                println(tree.transitions2[sib][1][1])
                println(tree.children[tree.transitions2[sib][1][1]])

            end
            sib_children = sa_children(tree, sib)
            for c in sib_children
                c_siblings = sib_children[sib_children .!= c]
                efc, vfc = failure_prob_sanode_cc(tree, c, c_siblings, correlation_coeffs, stored_statistics)
                push!(xi, efc)
                push!(σ2i, vfc)
                push!(wi, ρ)
            end
        end
    end

    # Include the estimation sample
    if sa != 0
        push!(xi, tree.fail_sample[sa])
        push!(σ2i, 0)
        push!(wi, 1)
    end

    # Get the combined statistics
    e, v = combine_samples(xi, wi, σ2i)

    # Store them in the dictionary and then return
    if sa != 0
        stored_statistics[sa] = (e,v)
    end

    return e,v
end

function isleaf(sa, tree)
    # First check if there have been any transitions from this sanode, if not then return its val
    tree.n_a_children[sa] == 0 && return true

    # if sp does not have any children then it does not have an estimate for q, so take the estimate of the sa node
    isempty(sa_children(tree, sa)) && return true

    return false
end


# Returns the failure probability of an sa node in the tree
function failure_prob_sanode_cc(tree, sa, siblings, correlation_coeffs, stored_statistics)
    # Check if the sa node is already in the dictionary, if so return the value
    if !all(isnan.(stored_statistics[sa]))
        return stored_statistics[sa]
    end

    # Check if the sa node is a leaf, in which case return its q value
    if isleaf(sa, tree)
        return tree.q2[sa], 0
    end

    # Compute the average failure probability of the children
    failure_prob_sa_children_cc(tree, sa_children(tree, sa), siblings, correlation_coeffs, stored_statistics, sa)
end

