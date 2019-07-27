using Plots
using DataStructures
include("../adversarial_gridworld.jl")
include("../failure_prob.jl")

mutable struct Trial
    method
    tree_size
    n_iterations
    failure_prob
    var_failure_prob
end

function add_data!(data, trial)
    if !haskey(data, trial.method)
        data[trial.method] = OrderedDict{Int, Array{Trial}}()
    end
    dict = data[trial.method]
    if !haskey(dict, trial.n_iterations)
        dict[trial.n_iterations] = Trial[]
    end
    arr = dict[trial.n_iterations]
    push!(arr, trial)
end

ec = 1e3 # Exploration constant for MCTS
samples_per_leaf = 10 # Samples per leaf for computing failure prob MCTs
trials_per_size = 10

s0 = GridWorldState(6,6,1)
w, h, N = 10, 10, 25
wins = [(rand(1:w), rand(1:h)) for i in 1:N]
wins = setdiff(wins, [(s0.x, s0.y)])
mdp, V, g_fail = create_sim(win_states = wins, w=w, h=h, T=30)

dpw_solver = DPWSolver(n_iterations=10000, depth=100, exploration_constant=1., k_state = .9, alpha_state = 0., check_repeat_state = false)
dpw_planner = solve(dpw_solver, mdp)
action(dpw_planner, s0)
dpw_planner.tree.q[dpw_planner.tree.children[1]]

# Display the gridworld so we know what we are working with
display_gridworld(mdp.g, string.([(location(i, mdp.g.w, 1).x, location(i, mdp.g.w, 1).y) for i in 1:mdp.g.w*mdp.g.h]))
savefig("gridworld_results_world.pdf")

tree_sizes = Int.(floor.(10 .^ range(2, stop=4, length=5)))
data = OrderedDict{String, OrderedDict{Int, Array{Trial}}}() # Maps method to a dictionary that maps tree size to several versions of results

maxN, minN = 0, Inf
for ts in tree_sizes
    println("Tree size: ", ts)
    Ns = 0
    for i in 1:trials_per_size
        println("    Trial: ", i)
        # MCTS - With regular UCB
        dpw_solver = DPWSolver(n_iterations=ts, depth=mdp.g.T, exploration_constant=ec, k_state = .9, alpha_state = 0., check_repeat_state = false)
        dpw_planner = solve(dpw_solver, mdp)
        action(dpw_planner, s0)
        E, σ2 = failure_prob(1, dpw_planner.tree, mdp, samples_per_leaf)

        # Get the number of rollouts for regular version
        if i == 1
            Ns = samples_per_leaf*num_leaves(dpw_planner.tree) + ts
            if Ns > maxN
                global maxN = Ns
            end
            if Ns < minN
                global minN = Ns
            end
        end

        # Save the data
        add_data!(data, Trial("MCTS-Reg", ts, Ns, E, σ2))



        # ## MCTS - With other version of UCB
        # dpw_solver = DPWSolver(n_iterations=ts, depth=mdp.g.T, exploration_constant=ec, k_state = .9, alpha_state = 0., check_repeat_state = false)
        # dpw_planner = solve(dpw_solver, mdp)
        # action(dpw_planner, s0)
        #
        # # Get the number of rollouts for the ucb-sampled version
        # N2 = samples_per_leaf*num_leaves(dpw_planner.tree) + ts
        #
        # # Save the data
        # data["MCTS-Reg"][N1] = Trial("MCTS-Reg", ts, N1, E, σ2)

        ## MC rollouts
        mc = [policy_rollout(g_fail, s0, mdp.π) for i=1:Ns]
        add_data!(data, Trial("MC", ts, Ns, mean(mc), var(mc)/Ns))

        ## IS rollouts
        uis_mean, uis_var = failure_rollout(s0, mdp, Ns)
        add_data!(data, Trial("UIS", ts, Ns, uis_mean, uis_var/Ns))
    end
end

##### Plot the results

## Show means with sample variance
plot(minN:maxN, fill(V[to_index(s0, mdp.g.w)], maxN), title="Comparison of Methods", xlabel="Rollout Budget", ylabel="Probability of Failure", label="Exact", xscale=:log)
for method in keys(data)
    Ns = []
    means = []
    sample_std = []
    for n in keys(data[method])
        trials = data[method][n]
        push!(Ns, n)
        fps = []
        for t in trials
            push!(fps, t.failure_prob)
        end
        push!(means, mean(fps))
        push!(sample_std, std(fps))
    end
    plot!(Ns, means, yerror = sample_std, label=method)
end
savefig("gridworld_results.pdf")

