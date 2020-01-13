include("../adversarial_continuum_world.jl")
include("plotting.jl")
include("trials.jl")
include("../failure_prob.jl")
include("../failure_prob_corr_coeff.jl")
# include("../failure_prob_iid_samps.jl")

# Starting state for the continuum gridworld
s0 = Vec2(5, 5)

# Construct the mdp and the result with value iteration
σ2 = 0.07
is_σ2 = 0.1
mdp_is, mdp_true, V, w_fail = create_adversarial_cworld(σ2 = σ2, is_σ2 = is_σ2, solver_m = 100, max_itrs = 40, s0 = s0)

# Get the reference probability of failure
V0 = evaluate(V, s0)

mc_rollout() = policy_rollout(w_fail, s0, mdp_true.policy)
is_rollout() = policy_rollout(w_fail, s0, mdp_true.policy, is_distribution = mdp_is.action_dist)

Nstep = 500
Nmax = 50000

t_mc, samples_mc, weights_mc = run_rollout_trial(mc_rollout, Nmax, Nstep)

# t_is, samples_is, weights_is = run_rollout_trial(is_rollout, Nmax, Nstep)

t_mcts, planner_true = run_mcts_trial(get_planner(mdp_true, Nstep, skip_terminal_branches = true), Nmax, failure_prob_fns = [failure_prob, failure_prob_cc])

# a,b = fit_beta(failure_prob_iid(planner_true.tree)...)

# t_mcts_is, planner_is = run_mcts_trial(get_planner(mdp_is, Nstep, skip_terminal_branches = true), Nmax)

plot_metric_comparisons([t_mcts[1], t_mcts[2], t_mc], ["MCTS Orig", "MCTS CC", "MC"], V0)

savefig(string("comparison_sig_", σ2, ".pdf"))



