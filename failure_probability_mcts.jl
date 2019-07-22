using Plots; gr()
using Statistics
include("gridworld.jl")
include("mcts.jl")
include("value_iteration.jl")
include("adversarial_simulator.jl")

w,h = 6,6
# Define a reward function
r = zeros(w*h)
r[state((3,3), w)] = 1
r[state((2,1), w)] = -1
terminals = findall(r.!= 0)

# Define a transition function
P = gen_P(w, h, terminals, [0.01, 0.01, 0.01, 0.01, 0])

# Put it together in a gridworld
g = GridWorld(w,h,r,P,0.8)

# Get a policy to evaluate
_, π0 = value_iteration(g)
display_gridworld(g,policy_to_annotations(π0))
savefig("gridworld_polciy.pdf")

# update the reward function to the new one (for failure probability)
g.r[state((3,3), w)] = 0
g.r[state((2,1), w)] = 1
g.γ = 1

V = policy_evaluation(g,π0,tol=1e-10)

display_gridworld(g, string.(round.(V, digits=2)), title = "Prob of Failure under π")

g.r[state((3,3), w)] = 1
g.r[state((2,1), w)] = -1
sim = AdversarialSimulator(g, 1, π0)

tree_size = unique(Int.(floor.(exp10.(range(1, 4, length=25)))))
total_rollouts = []
mcts_mean = []
mcts_std = []
mc_mean = []
mc_std = []
is_mean = []
is_std = []
s0 = 16
for ts in tree_size
    println("Tree size: ", ts)
    g.r[state((3,3), w)] = 1
    g.r[state((2,1), w)] = -1
    # Get the results for mcts
    root = Node(s0, s0, nothing)
    mcts(root, sim, ts)
    mean1, var1 = failure_prob(root,sim,10)
    push!(mcts_mean, mean1)
    push!(mcts_std, sqrt(var1))

    # Find out how many rollouts we used
    N = num_leaves(root)*10 + root.n
    push!(total_rollouts, N)

    # Get the results for importance sampling
    _, mean2, var2 = rollout(root, sim, N)
    push!(is_mean, mean2)
    push!(is_std, sqrt(var2/N))

    # Get the results from basic monte carlo
    g.r[state((3,3), w)] = 0
    g.r[state((2,1), w)] = 1
    mc = [rollout(g, s0, π0) for i=1:N]
    push!(mc_mean, mean(mc))
    push!(mc_std, sqrt(var(mc)/N))
end

plot(total_rollouts, mc_mean, label="Monte Carlo", xscale=:log)
xlabel!("Number of Samples")
ylabel!("Probability")
title!("Failure Probability Estimates")
# plot!(total_rollouts, is_mean, ribbon=2.78*is_std, label="Importance Sampling")

plot!(total_rollouts, mcts_mean, ribbon=(min.(mcts_mean,2.78*mcts_std), 2.78*mcts_std), label="MCTS")
plot!(total_rollouts, fill(V[s0],length(total_rollouts)), label="Exact")

savefig("results.pdf")


