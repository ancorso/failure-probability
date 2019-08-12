using Plots
using DataStructures
using Serialization
include("../adversarial_gridworld.jl")
include("../failure_prob.jl")
include("common.jl")

ec = 1e3 # Exploration constant for MCTS
samples_per_leaf = 5 # Samples per leaf for computing failure prob MCTs
trials_per_size = 10
s0 = GridWorldState(6,6,1)

mdp, V, g_fail = create_sim(AdversarialGridWorld, w=10, h=10, T=30, s0=s0, Nwins = 35)

# Display the gridworld so we know what we are working with
display_gridworld(mdp.g, string.([(location(i, mdp.g.w, 1).x, location(i, mdp.g.w, 1).y) for i in 1:mdp.g.w*mdp.g.h]))
savefig("adversarial_gridworld_results_world.pdf")

# Sample Solving
dpw_solver = DPWSolver(n_iterations=10000, depth=mdp.g.T, exploration_constant=1e3, k_state = .9, alpha_state = 0., check_repeat_state = false, estimate_value=myrollout, next_action = random_action, sample_ucb = true)
dpw_planner = solve(dpw_solver, mdp)
action(dpw_planner, s0)
num_leaves(dpw_planner.tree)
maximum(dpw_planner.tree.q)
failure_prob(1,dpw_planner.tree,mdp,10)
V[to_index(s0, mdp.g.w)]


tree_sizes = Int.(floor.(10 .^ range(1, stop=5, length=10)))
data = OrderedDict{String, OrderedDict{Int, Array{Trial}}}() # Maps method to a dictionary that maps tree size to several versions of results

data

maxN, minN = 0, Inf
for ts in tree_sizes
    println("Tree size: ", ts)
    Ns = 0
    for i in 1:trials_per_size
        # MCTS - With regular UCB
        dpw_solver1 = DPWSolver(n_iterations=ts, depth=mdp.g.T, exploration_constant=ec, k_state = .9, alpha_state = 0., check_repeat_state = false, sample_ucb = false)
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
        dpw_solver2 = DPWSolver(n_iterations=ts, depth=mdp.g.T, exploration_constant=ec, k_state = .9, alpha_state = 0., check_repeat_state = false, sample_ucb = true)
        dpw_planner2 = solve(dpw_solver2, mdp)
        action(dpw_planner2, s0)
        E, σ2 = failure_prob(1, dpw_planner2.tree, mdp, samples_per_leaf)
        add_data!(data, Trial("MCTS-Samp", ts, Ns, E, σ2))

        ## MC rollouts
        mc = [policy_rollout(g_fail, s0, mdp.π) for i=1:Ns]
        add_data!(data, Trial("MC", ts, Ns, mean(mc), var(mc)/Ns))

        ## IS rollouts
        uis_mean, uis_var = failure_rollout(s0, mdp, Ns)
        add_data!(data, Trial("UIS", ts, Ns, uis_mean, uis_var/Ns))
    end
end

##### Plot the results
cd("Workspace/failure-probability/")
data = deserialize("gridworld_data.jls")
minN = 1e3
maxN = 2e5

## Show means with sample variance
p1 = plot(minN:maxN, fill(V[to_index(s0, mdp.g.w)], Int(maxN - minN + 1)), title="Probability Estimates - Numerical Std", xlabel="Rollout Budget", ylabel="Probability of Failure", label="Exact", xscale=:log10, yscale = :log10, legend=:bottomright, xlims = (minN, maxN))

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

savefig(p1, "gridworld_results_mean_numerical_std.pdf")
# savefig(p2, "gridworld_results_var.pdf")


serialize("data.jls", data)

