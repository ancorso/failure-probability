include("../adversarial_continuum_world.jl")
include("plotting.jl")
include("../failure_prob.jl")
include("trials.jl")

# Starting state for the continuum gridworld
s0 = Vec2(5, 5)

# Construct the mdp and the result with value iteration
σ2 = 0.2 # True distribution variance
_, mdp, V, w_fail = create_adversarial_cworld(σ2 = σ2, is_σ2 = σ2, solver_m = 20, max_itrs = 20, s0 = s0)

# Get the reference probability of failure
V0 = evaluate(V, s0)

# Get the planner and construct a search tree
dpw_planner = get_planner(mdp, 1000)
action(dpw_planner, [s0])

# Compute our estimate of the failure probability
prob_fail, variance = failure_prob(dpw_planner.tree)

