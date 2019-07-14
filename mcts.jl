using Statistics

Aspace = Int
Sspace = Int
mutable struct Node
    a::Aspace
    s::Sspace
    parent::Union{Node, Nothing}
    children::Dict{Int, Node}
    Q::Float64
    n::Int
end

Node(action::Aspace, state::Sspace, parent::Union{Node, Nothing}) = Node(action, state, parent, Dict{Int, Node}(), 0, 0)

Node(action::Aspace, state::Sspace) = Node(action, state, nothing)

is_root(node::Node) = node.parent == nothing

fully_expanded(node::Node, sim) = length(node.children) == length(actions_from(node.s, sim))

function add_child(parent, child)
    a = child.a
    @assert !haskey(parent.children, a)
    child.parent = parent
    parent.children[a] = child
    child
end

function pick_unvisited(node::Node, sim)
    fully_expanded(node, sim) && error("Node is already fully expanded")
    a = rand(setdiff(actions_from(node.s, sim), keys(node.children)))
    new_s = step(a, sim)
    add_child(node, Node(a, new_s))
end

function best_uct(node::Node)
    child_list = collect(values(node.children))
    c = argmax([child.Q + sqrt(200000) * sqrt(log(node.n) / child.n) for child in child_list])
    return child_list[c]
end

# function for node traversal
function traverse(node, sim)
    while fully_expanded(node, sim)
        terminal(node.s, sim) && return node
        node = best_uct(node)
    end
    n = pick_unvisited(node, sim)
    println("Leaf node: ", n.s)
    return n
end

# function for the result of the simulation
function rollout(node, sim, sims_per_rollout)
    Gs = []
    counts = []
    for i = 1:sims_per_rollout
        s = node.s
        G = 0
        multiplier = 1
        weight = 1
        while !terminal(s, sim)
            possible_actions = actions_from(s, sim)
            a = rand(possible_actions)
            sp = step(a, sim)
            weight *= length(possible_actions)*transition_prob(s, sp, sim)
            G += multiplier*reward(s,sp, sim)
            multiplier *= discount(sim)
            s = sp
        end
        push!(counts, weight*in_E(s, sim))
        push!(Gs, G) # the return
    end
    mean(Gs), mean(counts), var(counts)
end

# function for backpropagation
function backpropagate(node, G, sim)
    q = discount(sim)*G
    node.n += 1
    if !is_root(node)
        q += reward(node.parent.s, node.s, sim)
    end
    node.Q += (q - node.Q)/node.n
    if !is_root(node)
        backpropagate(node.parent, q, sim)
    end
end

# main function for the Monte Carlo Tree Search
function mcts(root, sim, max_sims)
    sims = 0
    while sims < max_sims
        leaf = traverse(root, sim)
        ret, _, _ = rollout(leaf, sim, 1)
        sims += 1
        backpropagate(leaf, simulation_result, sim)
    end
end

function failure_prob(node, sim, sims_per_rollout)
    if is_leaf(node)
         _, E_fail, var_fail = rollout(node, sim, sims_per_rollout)
         return E_fail, var_fail
    else
        Nc = length(node.children)
        E_fail = 0
        var_fail = 0
        for c in node.children
            E_fail_c, var_fail_c = failure_prob(c, sim, sims_per_rollout)
            E_fail += E_fail_c / N_c
            var_fail += var_fail_c / N_c^2
        end
        return E_fail, var_fail
    end
end

