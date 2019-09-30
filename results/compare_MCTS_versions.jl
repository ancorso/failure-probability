include("../adversarial_continuum_world.jl")
include("plotting.jl")
include("trials.jl")

# Construct the mdp and the result with value iteration
mdp, V, w_fail = create_adversarial_cworld(σ2 = 0.04, is_σ2 = 0.04, solver_m = 20, max_itrs = 20)

# Starting state for the continuum gridworld
s0 = Vec2(5, 5)

# Get the reference probability of failure
V0 = evaluate(V, s0)

mc_rollout() = policy_rollout(w_fail, s0, mdp.policy)

Nstep = 500
Nmax = 30000

val, rollout_states = policy_rollout(w_fail, s0, mdp.policy, return_states = true)

plot_path!(plot_cworld(mdp),rollout_states)
w_fail


t, planner = run_mcts_trial(get_planner(mdp, Nstep, skip_terminal_branches = true), Nmax)
# run_rollout_trial(mc_rollout, Nmax, Nstep)

plot_trial(t, V0,mdp, planner.tree, s0)

savefig("trial_2.pdf")



