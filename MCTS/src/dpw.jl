POMDPs.solve(solver::DPWSolver, mdp::Union{POMDP,MDP}) = DPWPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::DPWPlanner)
    p.tree = nothing
end

function get_ucb(tree, snode, c, skip_terminal_branches)
    children = tree.children[snode]
    UCBs = -Inf.*ones(length(children))
    ltn = log(tree.total_n[snode])

    for ci in 1:length(children)
        child = children[ci]
        if skip_terminal_branches && child in tree.skip
            continue;
        end
        n = tree.n[child]
        q = tree.q[child]
        if (ltn <= 0 && n == 0) || c == 0.0
            UCBs[ci] = q
        else
            UCBs[ci] = q + c*sqrt(ltn/n)
        end
        @assert !isnan(UCBs[ci]) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        # @assert !isequal(UCBs[ci], -Inf)
    end
    UCBs, children
end

function get_best_ucb(tree, snode, c, skip_terminal_branches)
    ucbs, children = get_ucb(tree, snode, c, skip_terminal_branches)
    children[argmax(ucbs)]
end

function get_sample_ucb(tree, snode, c, rng, skip_terminal_branches)
    ucbs, children = get_ucb(tree, snode, c, skip_terminal_branches)
    ucbs = ucbs ./ 1e4

    sm = sum(exp10.(ucbs))

    if sm == Inf
        sai = findfirst(ucbs .== Inf)
    elseif sm == 0
        sai = rand(1:length(children))
    else
        sai = findfirst(rand(rng, Multinomial(1, exp10.(ucbs) ./ sm)) .== 1)
    end
    children[sai]
end

"""
Construct an MCTSDPW tree and choose the best action.
"""
POMDPs.action(p::DPWPlanner, s) = first(action_info(p, s))

"""
Construct an MCTSDPW tree and choose the best action. Also output some information.
"""
function POMDPModelTools.action_info(p::DPWPlanner, s; tree_in_info=false)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        S = statetype(p.mdp)
        A = actiontype(p.mdp)
        if p.solver.keep_tree
            if p.tree == nothing
                tree = DPWTree{S,A}(p.solver.n_iterations)
                p.tree = tree
            else
                tree = p.tree
            end
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = insert_state_node!(tree, s, true)
            end
        else
            tree = DPWTree{S,A}(p.solver.n_iterations)
            p.tree = tree
            snode = insert_state_node!(tree, s, p.solver.check_repeat_state)
        end

        nquery = 0
        start_us = CPUtime_us()
        for i = 1:p.solver.n_iterations
            nquery += 1
            simulate(p, snode, p.solver.depth) # (not 100% sure we need to make a copy of the state here)
            if CPUtime_us() - start_us >= p.solver.max_time * 1e6
                break
            end
        end
        info[:search_time_us] = CPUtime_us() - start_us
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        best_Q = -Inf
        sanode = 0
        for child in tree.children[snode]
            if tree.q[child] > best_Q
                best_Q = tree.q[child]
                sanode = child
            end
        end
        # XXX some publications say to choose action that has been visited the most
        a = tree.a_labels[sanode] # choose action with highest approximate value
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end


"""
Return the reward for one iteration of MCTSDPW.
"""
function simulate(dpw::DPWPlanner, snode::Int, d::Int, action_sequence = [])
    S = statetype(dpw.mdp)
    A = actiontype(dpw.mdp)
    sol = dpw.solver
    tree = dpw.tree
    s = tree.s_labels[snode]
    if d == 0 || isterminal(dpw.mdp, s)
        if dpw.solver.double_reward
            return 0.0, 0.0, true
        else
            return 0.0
        end
    end

    ucbs, children = get_ucb(tree, snode, sol.exploration_constant, sol.skip_terminal_branches)
    # action progressive widening
    if dpw.solver.enable_action_pw
        if all(ucbs .== -Inf) || length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a, ρ = next_action(dpw.next_action, dpw.mdp, s, DPWStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, dpw.mdp, s, a)
                insert_action_node!(tree, snode, a, ρ, n0,
                                    ρ*init_Q(sol.init_Q, dpw.mdp, s, a),
                                    sol.check_repeat_action
                                   )
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        println("Shouldn't be here!")
        error("Shouldn't be here!")
        for a in actions(dpw.mdp, s)
            n0 = init_N(sol.init_N, dpw.mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, dpw.mdp, s, a),
                                false)
            tree.total_n[snode] += n0
        end
    end


    if sol.sample_ucb
        sanode = get_sample_ucb(tree, snode, sol.exploration_constant, sol.rng, sol.skip_terminal_branches)
    else
        sanode = get_best_ucb(tree, snode, sol.exploration_constant, sol.skip_terminal_branches)
    end

    if sanode in tree.skip
        println("found a skipped node!")
    end

    a = tree.a_labels[sanode]
    push!(action_sequence, a)

    # state progressive widening
    new_node = false
    if tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state
        if dpw.solver.double_reward
            sp, r, r2 = generate_sr(nothing, dpw.mdp, s, a, dpw.rng)
        else
            sp, r = generate_sr(dpw.mdp, s, a, dpw.rng)
        end

        if sol.check_repeat_state && haskey(tree.s_lookup, sp)
            spnode = tree.s_lookup[sp]
        else
            spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end

        if dpw.solver.double_reward
            push!(tree.transitions2[sanode], (spnode, r, r2))
        else
            push!(tree.transitions[sanode], (spnode, r))
        end

        if !sol.check_repeat_state
            tree.n_a_children[sanode] += 1
        elseif !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        if dpw.solver.double_reward
            spnode, r, r2 = rand(dpw.rng, tree.transitions2[sanode])
        else
            spnode, r = rand(dpw.rng, tree.transitions[sanode])
        end
    end

    if new_node
        if dpw.solver.double_reward
            # ρ = tree.ρ[sanode]
            q, q2, rollout_actions = estimate_value(dpw.solved_estimate, dpw.mdp, sp, d-1)
            if !isnan(tree.fail_sample[sanode])
                error("already filled out this sanode")
            end
            tree.fail_sample[sanode] = q2
            if q2 > 0
                push!(action_sequence, rollout_actions...)
                push!(dpw.tree.action_sequences, action_sequence)
            end
            q = (r + discount(dpw.mdp)*q)
            q2 = (r2 + discount(dpw.mdp)*q2)
            on_terminal = false
        else
            q = r + discount(dpw.mdp)*estimate_value(dpw.solved_estimate, dpw.mdp, sp, d-1)
            # q *= tree.ρ[sanode]
        end
    else
        if dpw.solver.double_reward
            # ρ = tree.ρ[sanode]
            q, q2, on_terminal = simulate(dpw, spnode, d-1, action_sequence)
            if sol.skip_terminal_branches && on_terminal
                push!(dpw.tree.skip, sanode)
            end

            q = (r + discount(dpw.mdp)*q)
            q2 = (r2 + discount(dpw.mdp)*q2)
        else
            q = r + discount(dpw.mdp)*simulate(dpw, spnode, d-1, action_sequence)
            # q *= tree.ρ[sanode]
        end
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]
    if dpw.solver.double_reward
        tree.q2[sanode] += (q2 - tree.q2[sanode])/tree.n[sanode]
        return q, q2, on_terminal
    else
        return q
    end
end

