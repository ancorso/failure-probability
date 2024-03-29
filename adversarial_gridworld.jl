using POMDPs
using Random
using Distributions

# Defines a gridworld MDP
mutable struct GridWorld
    w::Int64 # Gridworld width
    h::Int64 # Gridworld height
    reward::Function # The reward function
    P::Array{Array{Tuple{Int, Float64}}, 2} # transition probabilities
    γ::Float64 # Discount factor
    T::Int # Maximum number of steps
end

struct GridWorldState
    x::Int
    y::Int
    t::Int
end

# Converts gridworld actions to text
action_text = Dict(1 => "->", 2 => "^", 3 => "<-", 4 => "v")
action_dir = Dict(1 => (1,0), 2 => (0,1), 3 => (-1,0), 4 => (0,-1), 5 => (0,0))
policy_to_annotations(π) = [action_text[π[i]] for i in 1:length(π)]

# Gets the number of states in the gridworld
nS(g::GridWorld) = g.w*g.h

# Returns true if a state is terminal
function is_terminal(s, g)
    (s.t >= g.T) && return true
    for arr in g.P[to_index(s, g.w), :]
        length(arr) > 0 && return false
    end
    true
end

function gw_generate_sr(g::GridWorld, s::GridWorldState, a::Int, rng::Random.AbstractRNG = Random.GLOBAL_RNG)
    si = to_index(s, g.w)
    sps = g.P[si, a]
    Ps = [nstate[2] for nstate in sps]
    spi = findfirst(rand(rng, Multinomial(1, Ps)) .== 1)
    new_s = location(sps[spi][1], g.w, s.t+1)
    return new_s, g.reward(new_s)
end

# Returns the transition probability of a s,a,s` pair
function transition_prob(s, a, sp, g)
    si = to_index(s, g.w)
    spi = to_index(sp, g.w)
    possible_next_states = g.P[si,a]
    for ns in possible_next_states
        ns[1] == spi && return ns[2]
    end
    return 0
end

# rollout from a starting state
function policy_rollout(g::GridWorld, s0, π)
    s = s0
    tot_r = g.reward(s)
    gmult = 1
    while true
        s, r = gw_generate_sr(g, s, π[to_index(s, g.w)])
        gmult *= g.γ
        tot_r += gmult*r
        is_terminal(s,g) && return tot_r
    end
end

# Returns a list of probabilities of state transitions
function P_stencil(si, a, w, h, prob_in)
    prob = copy(prob_in)
    prob[a] += (1 - sum(prob))
    d = Dict{Int, Float64}()
    for i=1:length(prob)
        sp = to_index(move(location(si, w, 1), action_dir[i], w, h), w)
        if prob[i] > 0
            d[sp] = get(d, sp, 0) + prob[i]
        end
    end
    [(k, d[k]) for k in keys(d)]
end

function gen_P(w, h, terminals = [], prob = [0.1, 0.1, 0.1, 0.1, 0.1])
    nS = w*h
    P = Array{Array{Tuple{Int, Float64}}, 2}(undef, nS, 4)
    for s=1:nS, a=1:4
        if !(s in terminals)
            P[s, a] = P_stencil(s, a, w, h, prob)
        else
            P[s,a] = []
        end
    end
    P
end

# Gets the location of a state in a gridworld with the specified width
function location(si, width, t)
    c = Int(floor((si - 1) / width)) + 1
    r = mod1(si, width)
    GridWorldState(r, c, t)
end

# Gets the index of the state of a given location
to_index(s, width) = (s.y-1)*width + s.x

# Move to a new state in the specified direction
move(s, dir, w, h) = GridWorldState(clamp(s.x + dir[1], 1, w), clamp(s.y + dir[2], 1, h), s.t + 1)

# Creates a unit square at the specified location (for plotting)
square(x, y) = Shape(x .+ [-0.5, 0.5, 0.5, -0.5, -0.5], y .+ [-0.5, -0.5, 0.5, 0.5, -0.5])

# Plos a gridworld with the desired annotations
function display_gridworld(g::GridWorld, annotations::Array{String}; title = "")
    r = [g.reward(location(s,g.w,1)) for s in 1:nS(g)]
    minr, maxr = extrema(r)
    rs = (r .- minr) ./ (maxr - minr)
    p = plot(xaxis = false, yaxis = false, grid = false, linecolor=:black, title=title)
    for i=1:nS(g)
        s = location(i, g.w, 1)
        x,y = s.x, s.y
        c = cgrad(:RdYlGn)[rs[i]]
        plot!(square(x,y), fillcolor=c, label="", annotations = (x, y, annotations[i]))
    end
    p
end

function policy_evaluation(g::GridWorld, policy; tol=1e-3)
	vp = zeros(nS(g))
	v = vp .+ Inf
	while maximum(abs.(v .- vp)) > tol
		v .= vp
		for s in 1:nS(g)
			a = policy[s]
			next_states = g.P[s, a]
			s_f = location(s, g.w, 1)
			vp[s] = g.reward(s_f)
			for (sp, prob) in next_states
				vp[s] += g.γ*prob*v[sp]
            end
        end
    end
	vp
end

function value_iteration(g::GridWorld; tol=1e-3)
    Qp = zeros(nS(g), 4)
    Q = Qp .+ Inf
    while maximum(abs.(Q .- Qp)) > tol
    	Q .= Qp
    	for s in 1:nS(g), a in 1:4
    		next_states = g.P[s,a]
			s_f = location(s, g.w, 1)
    		Qp[s,a] = g.reward(s_f)
    		for (sp, prob) in next_states
    			Qp[s,a] += g.γ * prob * maximum(Q[sp,:])
            end
        end
    end
    policy = [argmax(Q[i,:]) for i in 1:size(Q,1)]

    return Qp, policy
end



mutable struct AdversarialGridWorld <: MDP{GridWorldState, Symbol}
    g::GridWorld
    π::Array
end

mutable struct SeedGridWorld <: MDP{Array{UInt}, UInt}
    g::GridWorld
    π::Array
    s0::GridWorldState
end

function id(a)
    (a == :right) && return 1
    (a == :up) && return 2
    (a == :left) && return 3
    (a == :down) && return 4
end

function POMDPs.generate_sr(mdp::AdversarialGridWorld, s::GridWorldState, a::Symbol, rng::Random.AbstractRNG)
    sp = move(s, action_dir[id(a)], mdp.g.w, mdp.g.h)
    r = ast_reward(s, sp, mdp)
    return sp, r
end

function get_actual_state(mdp::SeedGridWorld, s::Array{UInt})
    actual_state = mdp.s0
    for i in s
        actual_state, _ = gw_generate_sr(mdp.g, actual_state, mdp.π[to_index(actual_state, mdp.g.w)], MersenneTwister(i))
    end
    actual_state
end

function transition_probabilities(mdp, s)
    w = 1
    actual_state = mdp.s0
    for i in s
        next_state, _ = gw_generate_sr(mdp.g, actual_state, mdp.π[to_index(actual_state, mdp.g.w)], MersenneTwister(i))
        w *= transition_prob(actual_state, next_state, mdp)
        actual_state = next_state
    end
    w
end

function POMDPs.generate_sr(mdp::SeedGridWorld, s::Array{UInt}, a::UInt, rng::Random.AbstractRNG)
    actual_state = get_actual_state(mdp, s)
    next_state, _ = gw_generate_sr(mdp.g, actual_state, mdp.π[to_index(actual_state, mdp.g.w)], MersenneTwister(a))
    [s..., a], ast_reward(actual_state, next_state, mdp)
end

function POMDPs.generate_sr(mdp::SeedGridWorld, s::GridWorldState, a::UInt, rng::Random.AbstractRNG)
    sp, _ = gw_generate_sr(mdp.g, s, mdp.π[to_index(s, mdp.g.w)], MersenneTwister(a))
    return sp, ast_reward(s, sp, mdp)
end

POMDPs.actions(mdp::AdversarialGridWorld) = [:right, :up, :left, :down]

POMDPs.n_actions(mdp::AdversarialGridWorld) = 4

random_action(mdp::SeedGridWorld, s::Array{UInt}, snode) = rand(UInt)
random_action(mdp::SeedGridWorld, s::GridWorldState, snode) = rand(UInt)
random_action(mdp::AdversarialGridWorld, s::GridWorldState, snode) = rand(actions(mdp))

rollout_weight(mdp::SeedGridWorld, s, sp) = 1
rollout_weight(mdp::AdversarialGridWorld, s, sp) = n_actions(mdp)*transition_prob(s, sp, mdp)

tree_weight(mdp::SeedGridWorld, Nc, s, sp) = 1
tree_weight(mdp::AdversarialGridWorld, Nc, s, sp) = Nc*transition_prob(s, sp, mdp)


POMDPs.isterminal(mdp::AdversarialGridWorld, s::GridWorldState) = is_terminal(s, mdp.g)

POMDPs.isterminal(mdp::SeedGridWorld, s::Array{UInt}) = is_terminal(get_actual_state(mdp, s), mdp.g)
POMDPs.isterminal(mdp::SeedGridWorld, s::GridWorldState) = is_terminal(s, mdp.g)

POMDPs.discount(mdp::AdversarialGridWorld) = 1
POMDPs.discount(mdp::SeedGridWorld) = 1



transition_prob(s, sp, mdp) = transition_prob(s, mdp.π[to_index(s, mdp.g.w)], sp, mdp.g)

in_E(s, mdp) = mdp.g.reward(s) == -1

function shortest_distance_to_error(sp, mdp)
    error_states = findall([mdp.g.reward(location(s, mdp.g.w, 1)) for s in 1:nS(mdp.g)] .== -1)
    min_dist = Inf
    for e in error_states
        loc = location(e, mdp.g.w, 1)
        dist = sqrt((sp.x - loc.x)^2 + (sp.y - loc.y)^2)
        if dist < min_dist
            min_dist = dist
        end
    end
    return min_dist
end

function ast_reward(s, sp, sim)
    reward = 1. / (1+shortest_distance_to_error(sp, sim))#log(transition_prob(s, sp, sim))
    if is_terminal(sp, sim.g) && !in_E(sp, sim)
        reward += -10000
    end
    if is_terminal(sp, sim.g) && in_E(sp, sim)
        reward += 10000
    end
    reward
end

function create_sim(mdp_type; w=4, h=4, Nwins = 10, lose_states = [(2,1)], T = 30, p_val = 0.01, s0 = GridWorld(1,1,1))
    win_states = [(rand(1:w), rand(1:h)) for i in 1:Nwins]
    win_states = setdiff(win_states, [(s0.x, s0.y), lose_states...])

    # Define a reward function
    function reward_true(s)
        ((s.x, s.y) in win_states) && return 1
        ((s.x, s.y) in lose_states) && return -1
        return 0
    end

    function reward_fail(s)
        ((s.x, s.y) in lose_states) && return 1
        return 0
    end
    terminals = findall([reward_true(location(s, w, 1)) for s in 1:w*h] .!=  0)

    # Define a transition function
    P = gen_P(w, h, terminals, [p_val, p_val, p_val, p_val, 0])

    # Put it together in a gridworld
    g_true = GridWorld(w, h, reward_true, P, 1, T)
    g_fail = GridWorld(w, h, reward_fail, P, 1, T)

    # Obtain a policy to evaluate
    _, π0 = value_iteration(g_true)
    display_gridworld(g_true, policy_to_annotations(π0))

    # Get the exact probability of failure
    V = policy_evaluation(g_fail, π0, tol=1e-30)

    if mdp_type == AdversarialGridWorld
        mdp = AdversarialGridWorld(g_true, π0)
    elseif mdp_type == SeedGridWorld
        @assert s0 != nothing
        mdp = SeedGridWorld(g_true, π0, s0)
    end
    mdp, V, g_fail
end

