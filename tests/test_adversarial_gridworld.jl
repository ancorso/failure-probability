using Test
using Random
using POMDPs
using Plots
using MCTS
include("../adversarial_gridworld.jl")
include("../failure_prob.jl")

################# Testing regular gridworld ##########################
# Test location(s, width)
@test location(8, 4, 1) == GridWorldState(4,2,1)

# Test state(pos, width)
@test to_index(GridWorldState(4,2,1), 4) == 8

# Test move(s, dir, w, h)
@test move(GridWorldState(4,2,1), (1,0), 4, 4) == GridWorldState(4,2,2)
@test move(GridWorldState(4,2,1), (0,1), 4, 4) == GridWorldState(4,3,2)
@test move(GridWorldState(1,1,1), (0,-1), 4, 4) == GridWorldState(1,1,2)
@test move(GridWorldState(1,1,1), (-1,0), 4, 4) == GridWorldState(1,1,2)
@test move(GridWorldState(2,1,1), (0,-1), 2, 2) == GridWorldState(2,1,2)

# Set the dimensions of our gridworld
w,h = 4,4

# Define a reward function
function reward(s)
    if (s.x, s.y) in [(3,3)]
        return 1
    elseif (s.x, s.y) in [(2,1)]
        return -1
    else
        return 0
    end
end

terminals = findall([reward(location(s, w, 1)) for s in 1:w*h] .!=  0)

# Define a transition function
P = gen_P(w, h, terminals, [0.01, 0.01, 0.01, 0.01, 0])

# Put it together in a gridworld
g = GridWorld(w, h, reward, P, 0.8, 10)

# Check transition probability
@test transition_prob(GridWorldState(1,1,1), 1, GridWorldState(2,1,1), g) == 0.97
@test transition_prob(GridWorldState(1,1,1), 1, GridWorldState(2,2,1), g) == 0
@test transition_prob(GridWorldState(1,1,1), 1, GridWorldState(1,2,1), g) == 0.01

# Check is_terminal function
@test is_terminal(GridWorldState(3, 3, 1), g)
@test is_terminal(GridWorldState(2, 1, 1), g)
@test is_terminal(GridWorldState(1, 1, 10), g)
@test !is_terminal(GridWorldState(2,2, 3), g)

# Demonstrate the functionality of value iteration for determining an optimal policy
Q, π = value_iteration(g)
# Q[terminals,:] .= r[terminals]

# Demonstrate value iteration
Veval = policy_evaluation(g, π)
# Veval[terminals] .= r[terminals]

# Turn results into annotations
Vstr = string.(round.(dropdims(maximum(Q, dims=2), dims=2), digits=2))
πstr = policy_to_annotations(π)
Veval_str = string.(round.(Veval, digits = 2))

# Plot the results and save a figure
p_pol = display_gridworld(g, πstr, title = "Optimal Policy")
p_V = display_gridworld(g, Vstr, title = "Optimal Value Function")
p_poleval = display_gridworld(g, Veval_str, title = "Policy Eval on Optimal Policy")

plot(p_pol, p_V, p_poleval, size = (1000,1000))
savefig("results.pdf")

################# Testing seed adversarial gridworld ########################
mdp, V, _ = create_sim(SeedGridWorld, s0 = GridWorldState(1,1,1), p_val = 0.25)
@test discount(mdp) == 1
s, r = generate_sr(mdp, Int[], 12, Random.GLOBAL_RNG)
@test get_actual_state(mdp, [12]) == GridWorldState(2,1,2)
@test s == [12]
@test isterminal(mdp,s)
@test r == 1e4 + 1

################# Testing adversarial gridworld ########################
mdp, V, _ = create_sim(AdversarialGridWorld)
@test discount(mdp) == 1
s, r = generate_sr(mdp, GridWorldState(1,1,29), :up, Random.GLOBAL_RNG)
@test s == GridWorldState(1,2,30)
@test isterminal(mdp,s)
@test r == log(0.97) - 1000*sqrt(2) - 10000

s, r = generate_sr(mdp, GridWorldState(1,1,29), :right, Random.GLOBAL_RNG)
@test s == GridWorldState(2,1,30)
@test isterminal(mdp, s)
@test r == log(0.01)


w, h, N = 10, 10, 15
wins = [(rand(1:w), rand(1:h)) for i in 1:N]
mdp, V, _ = create_sim(AdversarialGridWorld, win_states = wins, w=w, h=h, T=30)

display_gridworld(mdp.g, string.([(location(i, mdp.g.w, 1).x, location(i, mdp.g.w, 1).y) for i in 1:mdp.g.w*mdp.g.h]))


dpw_solver = DPWSolver(n_iterations=5000, depth=30, exploration_constant=1e4, k_state = .9, alpha_state = 0., check_repeat_state = false)

dpw_planner = solve(dpw_solver, mdp)


s0 = GridWorldState(6,5,1)
action(dpw_planner, s0)
dpw_planner.tree

@test !is_leaf(1, dpw_planner.tree)
num_leaves(dpw_planner.tree)
@test length(get_children(1, dpw_planner.tree)) == 4

s = 1
while !is_leaf(s, dpw_planner.tree)
    global s = get_children(s, dpw_planner.tree)[1]
end

@test is_leaf(s, dpw_planner.tree)
@test isempty(get_children(s, dpw_planner.tree))

@test failure_rollout(GridWorldState(2,1,1), mdp, 10) == (1., 0.)
failure_rollout(GridWorldState(3,3,1), mdp, 1000)

E, σ2 = failure_prob(1, dpw_planner.tree, mdp, 10)
σ = sqrt(σ2)
V[to_index(s0, mdp.g.w)]


# Display max_q predicted for each state
max_q = -Inf*ones(nS(mdp.g))a


for s in dpw_planner.tree.s_labels
    si = to_index(s, mdp.g.w)
    sl = dpw_planner.tree.s_lookup[s]
    children = dpw_planner.tree.children[sl]
    qs = dpw_planner.tree.q[children]
    if !isempty(qs)
        max_q[si] = max(max_q[si], maximum(qs))
    end
end
display_gridworld(mdp.g, string.(round.(max_q, digits=2)))
dpw_planner.tree



