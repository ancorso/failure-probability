using POMDPs
using Plots
using MCTS
include("adversarial_simulator.jl")

mdp, V = create_sim()
solver = MCTSSolver(n_iterations=10000, depth=100, exploration_constant=500000.0)

planner = solve(solver, mdp)
a = action(planner, 1)
planner.tree
solver

