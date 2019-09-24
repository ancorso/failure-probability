using Test
using Statistics
using POMDPs
using MCTS
include("../failure_prob.jl")

# Structure for storing the results of an estimation trial
mutable struct Trial
    mean_arr::Array{Float64}
    var_arr::Array{Float64}
    num_sanodes::Array{Float64}
    num_snodes::Array{Float64}
    pts::Array{Float64}
    num_failures::Array{Float64}
end

# Default constructor
Trial() = Trial([],[],[],[],[],[])

# Take the mean of an array of trials
function average(trials::Array{Trial})
    ma = mean([t.mean_arr for t in trials])
    va = mean([t.var_arr for t in trials])
    sa = mean([t.num_sanodes for t in trials])
    sn = mean([t.num_snodes for t in trials])
    pts = mean([t.pts for t in trials])
    fs = mean([t.num_failures for t in trials])
    Trial(ma, va, sa, sn, pts,fs)
end

# Run a trial with MCTS
# planner: This is the MCTS planner with the mdp.
#           Assumed to have keep_tree set to true
#           Runs the planner for n_iterations number of steps in between data acquisition
# Nmax: The maximum number of iterations
function run_mcts_trial(planner, Nmax)
    @assert planner.tree == nothing || planner.tree.s_lookup == nothing || length(planner.tree.s_lookup) == 0

    Nstep = planner.solver.n_iterations
    trial = Trial()
    last_planner = deepcopy(planner)
    last_E = 0
    for i in 1:Nstep:Nmax-Nstep
        action(planner, [s0])
        E, Var = failure_prob(planner.tree)

        if last_E != 0. && E == 0.
            println("Went to 0 from nonzero!")
            return last_planner, planner
        end
        last_planner = deepcopy(planner)
        last_E = E
        push!(trial.mean_arr, E)
        push!(trial.var_arr, Var)
        push!(trial.num_sanodes, length(planner.tree.q))
        push!(trial.num_snodes, length(planner.tree.s_lookup))
        push!(trial.pts, i + Nstep - 1)
        push!(trial.num_failures, length(planner.tree.action_sequences))
    end
    trial, planner
end

# Runs a Monte Carlo style rollout to get a mean and variance
# rollout: This is a function that returns a (potentially weighted) failure sample
# Nmax: This is the total number of iterations to run the trial for
# Nstep: This is the interval between data acquisition
function run_rollout_trial(rollout, Nmax, Nstep)
    vals, trial = [], Trial()
    for i = 1:Nstep:Nmax-Nstep
        push!(vals, [rollout() for j=1:Nstep]...)
        push!(trial.mean_arr, mean(vals))
        push!(trial.var_arr, var(vals)/length(vals))
        push!(trial.pts, i + Nstep - 1)
        push!(trial.num_failures, sum(vals))
    end
    trial
end

# Constructs a planner for the provided mdp
# Nstep is the number of iterations that MCTS is run each time action is called
function get_planner(mdp, Nstep; exploration_constant = 0.2, sample_ucb = false, k_action = 1., alpha_action = .1)
    solver = DPWSolver(n_iterations=Nstep,
                       depth=100,
                       exploration_constant= exploration_constant,
                       k_state = .9,
                       alpha_state = 0.,
                       check_repeat_state = false,
                       estimate_value = double_rollout,
                       next_action = random_action,
                       sample_ucb = sample_ucb,
                       k_action = k_action,
                       alpha_action = alpha_action,
                       double_reward = true,
                       keep_tree = true
                       )

    solve(solver, mdp)
end



### Here are some tests to make sure everything is working well.
t1 = Trial()
@test t1.mean_arr == t1.var_arr && t1.mean_arr == Float64[]
@test t1.num_sanodes == t1.num_snodes && t1.num_sanodes == Int[]

t2 = Trial([1.,2.,3.], [4.,5.,6.], [7, 8, 9], [10,11,12], [10,11,12], [0, 1, 1])
t3 = Trial([4.,5.,6.], [7, 8, 9], [10,11,12], [1.,2.,3.], [1.,2.,3.], [0, 1, 1])
t4 = average([t2, t3, t2, t3])
@test t4.mean_arr == [2.5, 3.5, 4.5]
@test t4.var_arr == [5.5, 6.5, 7.5]
@test t4.num_sanodes == [8.5, 9.5, 10.5]
@test t4.num_snodes == [5.5, 6.5, 7.5]
@test t4.pts == [5.5, 6.5, 7.5]
@test t4.num_failures == [0, 1, 1]

