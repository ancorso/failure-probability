using Plots
using DataStructures
using Serialization
include("../adversarial_continuum_world.jl")
include("../failure_prob.jl")
include("common.jl")

ec = 1e3 # Exploration constant for MCTS
samples_per_leaf = 3 # Samples per leaf for computing failure prob MCTs
trials_per_size = 5
s0 = Vec2(7, 5)
T = 30

mdp, V, w_fail = create_sim(σ2 = 0.07, is_σ2 = 0.07, solver_m = 1000)

# Display the gridworld so we know what we are working with
plot(CWorldVis(mdp.w, f=sp->reward(mdp.w, Vec2(0, 0),Vec2(0, 0), sp)))
savefig("cworld_rewards.pdf")

# Sample Solving
dpw_solver = DPWSolver(n_iterations=10000, depth=100, exploration_constant=1., k_state = .9, alpha_state = 0., check_repeat_state = false, estimate_value=myrollout, next_action = random_action, sample_ucb = false, k_action = 1., alpha_action = 0.4)
dpw_planner = solve(dpw_solver, mdp)
action(dpw_planner, s0)
num_leaves(dpw_planner.tree)
dpw_planner.tree
maximum(dpw_planner.tree.q)
dpw_planner.tree.q[dpw_planner.tree.children[1]]
dpw_planner.tree.a_labels[dpw_planner.tree.children[1][8]]

x = [s[1] for s in dpw_planner.tree.s_labels]
y = [s[2] for s in dpw_planner.tree.s_labels]

dpw_planner.tree.s_labels
plot(CWorldVis(mdp.w, f=sp->reward(mdp.w, Vec2(0, 0),Vec2(0, 0), sp)))
scatter!(x,y)

s = [s0]
si = 1
argmax(dpw_planner.tree.q[dpw_planner.tree.children[si]])
while true
    if isempty(dpw_planner.tree.children[si])
        break
    end
    ai_local = argmax(dpw_planner.tree.q[dpw_planner.tree.children[si]])
    ai_global = dpw_planner.tree.children[si][ai_local]
    if isempty(dpw_planner.tree.transitions[ai_global])
        break
    end
    global si = dpw_planner.tree.transitions[ai_global][1][1]
    push!(s, dpw_planner.tree.s_labels[si])
end
s
plot(CWorldVis(mdp.w, f=sp->reward(mdp.w, Vec2(0, 0),Vec2(0, 0), sp)))
plot!([si[1] for si in s], [si[2] for si in s])


failure_prob(1,dpw_planner.tree,mdp,3)
evaluate(V, s0)

savefig("plot_points.pdf")

@assert evaluate(V, s0) > 0


tree_sizes = Int.(floor.(10 .^range(1, stop=3, length=5)))
data = OrderedDict{String, OrderedDict{Int, Array{Trial}}}() # Maps method to a dictionary that maps tree size to several versions of results

maxN, minN = 0, Inf
for ts in tree_sizes
    println("Tree size: ", ts)
    Ns = 0
    for i in 1:trials_per_size
        # MCTS - With regular UCB
        dpw_solver1 = DPWSolver(n_iterations=ts, depth=T, exploration_constant=1., k_state = .9, alpha_state = 0., check_repeat_state = false, estimate_value=myrollout, next_action = random_action, sample_ucb = false, k_action = 1., alpha_action = 0.4)
        dpw_planner1 = solve(dpw_solver1, mdp)
        action(dpw_planner1, s0)
        E, σ2 = failure_prob(1, dpw_planner1.tree, mdp, samples_per_leaf)

        # Get the number of rollouts for regular version
        if i == 1
            Ns = samples_per_leaf*num_leaves(dpw_planner1.tree) + ts
            global maxN = max(maxN, Ns)
            global minN = min(minN, Ns)
        end
        println("    Trial: ", i, " Ns: ", Ns)

        # Save the data
        add_data!(data, Trial("MCTS-Reg", ts, Ns, E, σ2))

        # ## MCTS - With sampling based UCB
        dpw_solver2 = DPWSolver(n_iterations=ts, depth=T, exploration_constant=1., k_state = .9, alpha_state = 0., check_repeat_state = false, estimate_value=myrollout, next_action = random_action, sample_ucb = true, k_action = 1., alpha_action = 0.4)
        dpw_planner2 = solve(dpw_solver2, mdp)
        action(dpw_planner2, s0)
        E, σ2 = failure_prob(1, dpw_planner2.tree, mdp, samples_per_leaf)
        add_data!(data, Trial("MCTS-Samp", ts, Ns, E, σ2))

        ## MC rollouts
        mc = [policy_rollout(w_fail, s0, mdp.policy) for i=1:Ns]
        add_data!(data, Trial("MC", ts, Ns, mean(mc), var(mc)/Ns))

        # ## IS rollouts
        # uis_mean, uis_var = failure_rollout(s0, mdp, Ns)
        # add_data!(data, Trial("UIS", ts, Ns, uis_mean, uis_var/Ns))
    end
end

##### Plot the results

## Show means with sample variance
p1 = plot(minN:maxN, fill(evaluate(V, s0), Int(maxN - minN + 1)), title="Probability Estimates - Numerical Std", xlabel="Rollout Budget", ylabel="Probability of Failure", label="Exact", xscale=:log10, yscale = :log10, legend=:bottomright)

# p2 = plot(title="Standard Deviation Estimates", xlabel="Rollout Budget", ylabel="Average error in Variance", legend=:bottomright)

for method in keys(data)
    Ns = []
    means = []
    var_errors = []
    sample_std = []
    for n in keys(data[method])
        trials = data[method][n]
        push!(Ns, n)
        fps = []

        for t in trials
            push!(fps, t.failure_prob)
        end
        push!(means, mean(fps))
        actual_std = std(fps)
        push!(sample_std, actual_std)

        vars = []
        for t in trials
            push!(vars, sqrt(t.var_failure_prob))
        end
        #     println(vars)
        # end
        push!(var_errors, mean(vars))
    end
    means[findall(means .== 0)] .= NaN
    # var_errors[findall(var_errors .== 0)] .= NaN
    # plot!(p1, Ns, means, label=method, ribbon = (0, 2.576.*var_errors))
    plot!(p1, Ns, means, label=method, ribbon = (0, 2.576.*sample_std))
    # plot!(p2, Ns, var_errors, label=method, xscale=:log10)
end

savefig(p1, "cworld_results_mean_sample_std.pdf")
# savefig(p2, "gridworld_results_var.pdf")

serialize("data.jls", data)

