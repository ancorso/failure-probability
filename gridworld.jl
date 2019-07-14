using Distributions

# Defines a gridworld MDP
mutable struct GridWorld
    w::Int64 # Gridworld width
    h::Int64 # Gridworld height
    r::Array{Float64} # The reward vector
    P::Array{Array{Tuple{Int, Float64}}, 2} # transition probabilities
    γ::Float64 # Discount factor
end

# Converts gridworld actions to text
action_text = Dict(1 => "->", 2 => "^", 3 => "<-", 4 => "v")
action_dir = Dict(1 => (1,0), 2 => (0,1), 3 => (-1,0), 4 => (0,-1), 5 => (0,0))
policy_to_annotations(π) = [action_text[π[i]] for i in 1:length(π)]

# Gets the list of states in the gridworld
nS(g::GridWorld) = g.w*g.h

# Returns true if a state is terminal
function is_terminal(s, g)
    for arr in g.P[s, :]
        length(arr) > 0 && return false
    end
    true
end

# Returns the transition probability of a s,a,s` pair
function transition_prob(s, a, sp, g)
    possible_next_states = g.P[s,a]
    for ns in possible_next_states
        ns[1] == sp && return ns[2]
    end
    return 0
end

# rollout from a starting state
function rollout(g::GridWorld, s0, π)
    s = s0
    tot_r = 0
    gmult = 1
    while true
        tot_r = gmult*g.r[s]
        gmult *= g.γ
        sps = g.P[s, π[s]]
        isempty(sps) && return tot_r
        Ps = [nstate[2] for nstate in sps]
        spi = findfirst(rand(Multinomial(1, Ps)) .== 1)
        s = sps[spi][1]
    end
end

# Returns a list of probabilities of state transitions
function P_stencil(s, a, w, h, prob_in)
    prob = copy(prob_in)
    prob[a] += (1 - sum(prob))
    d = Dict{Int, Float64}()
    for i=1:length(prob)
        sp = move(s, action_dir[i], w, h)
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
function location(s, width)
    c = Int(floor((s - 1) / width)) + 1
    r = mod1(s, width)
    r, c
end

# Gets the index of the state of a given location
state(pos, width) = (pos[2]-1)*width + pos[1]

# Move to a new state in the specified direction
function move(s, dir, w, h)
    new_loc = location(s, w) .+ dir
    new_loc = max.(min.(new_loc, (w,h)), (1,1))
    state(new_loc, w)
end

# Creates a unit square at the specified location (for plotting)
square(x, y) = Shape(x .+ [-0.5, 0.5, 0.5, -0.5, -0.5], y .+ [-0.5, -0.5, 0.5, 0.5, -0.5])

# Plos a gridworld with the desired annotations
function display_gridworld(g::GridWorld, annotations::Array{String}; title = "")
    minr, maxr = extrema(g.r)
    rs = (r .- minr) ./ (maxr - minr)
    p = plot(xaxis = false, yaxis = false, grid = false, linecolor=:black, title=title)
    for i=1:nS(g)
        x,y = location(i, g.w)
        c = cgrad(:RdYlGn)[rs[i]]
        plot!(square(x,y), fillcolor=c, label="", annotations = (x, y, annotations[i]))
    end
    p
end

