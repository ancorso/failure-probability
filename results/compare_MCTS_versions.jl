include("../adversarial_continuum_world.jl")
include("plotting.jl")
include("trials.jl")

# Construct the mdp and the result with value iteration
mdp, V, w_fail = create_adversarial_cworld(σ2 = 0.03, is_σ2 = 0.03, solver_m = 100, max_itrs = 100)

# Starting state for the continuum gridworld
s0 = Vec2(7, 5)

# Get the reference probability of failure
V0 = evaluate(V, s0)

mc_rollout() = policy_rollout(w_fail, s0, mdp.policy)

Nstep = 500
Nmax = 30000

t, planner = run_mcts_trial(get_planner(mdp, Nstep), Nmax)
plot_trial(t, V0,mdp, planner.tree,s0)

savefig("trial_5.pdf")

