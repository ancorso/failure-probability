using Plots
include("../adversarial_continuum_world.jl")
include("../failure_prob.jl")

# Starting state for the continuum gridworld
s0 = Vec2(7, 5)

# Construct the mdp and the result with value iteration
mdp, V, w_fail = create_sim(σ2 = 0.07, is_σ2 = 0.07, solver_m = 1)

# Sample Solving
# Construct the solver with different options
dpw_solver = DPWSolver(n_iterations=10000,
                       depth=100,
                       exploration_constant=1.,
                       k_state = .9,
                       alpha_state = 0.,
                       check_repeat_state = false,
                       estimate_value=myrollout,
                       next_action = random_action,
                       sample_ucb = true,
                       k_action = 1.,
                       alpha_action = 0.4
                       )
# Solve the mdp
dpw_planner = solve(dpw_solver, mdp)
action(dpw_planner, s0)

# Get the state locations of each state in the tree and plot
x = [s[1] for s in dpw_planner.tree.s_labels]
y = [s[2] for s in dpw_planner.tree.s_labels]
plot(CWorldVis(mdp.w, f=sp->reward(mdp.w, Vec2(0, 0),Vec2(0, 0), sp)))
scatter!(x,y)




failure_prob(1,dpw_planner.tree,mdp,3)
# evaluate(V, s0)

savefig("states_in_tree_2ndmode.pdf")

# Actual value for s0 = Vec2(7, 5), σ2 = 0.07, is_σ2 = 0.07
# is: 0.000937