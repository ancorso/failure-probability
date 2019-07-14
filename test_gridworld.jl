using Test
using Plots
pyplot()
include("gridworld.jl")
include("value_iteration.jl")

# Test location(s, width)
@test location(8, 4) == (4,2)

# Test state(pos, width)
@test state((4,2), 4) == 8

# Test move(s, dir, w, h)
@test move(8, (1,0), 4, 4) == 8
@test move(8, (0,1), 4, 4) == 12
@test move(1, (0,-1), 4, 4) == 1
@test move(1, (-1,0), 4, 4) == 1
@test move(2, (0,-1), 2, 2) == 2

# Set the dimensions of our gridworld
w,h = 4,4

# Define a reward function
r = zeros(w*h)
r[state((3,3), w)] = 1
r[state((2,1), w)] = -1
terminals = findall(r.!= 0)

# Define a transition function
P = gen_P(w, h, terminals, [0.01, 0.01, 0.01, 0.01, 0])

# Put it together in a gridworld
g = GridWorld(w,h,r,P,0.8)

# Check transition probability
@test transition_prob(state((1,1), w), 1, state((2,1), w), g) == 0.97
@test transition_prob(state((1,1), w), 1, state((2,2), w), g) == 0
@test transition_prob(state((1,1), w), 1, state((1,2), w), g) == 0.01

# Check is_terminal function
@test is_terminal(state((3,3), w), g)
@test is_terminal(state((2,1), w), g)
@test !is_terminal(state((2,2), w), g)

# Demonstrate the functionality of value iteration for determining an optimal policy
Q, π = value_iteration(g)
Q[terminals,:] .= r[terminals]

# Demonstrate value iteration
Veval = policy_evaluation(g, π)
Veval[terminals] .= r[terminals]

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



