using Test
using Statistics
using POMDPs
using MCTS
# using PolynomialRoots

# Rollout function that tracks a "heuristic reward" for search and a "true reward"
# for computing probability estimates
function double_rollout(mdp, s, depth)
    tot_hr, tot_tr, mul = 0, 0, discount(mdp)
    actions = []
    while !isterminal(mdp, s)
        ract, ρ = random_action(mdp, s, nothing)
        push!(actions, ract)
        sp, hr, tr = generate_sr(nothing, mdp, s, ract, Random.GLOBAL_RNG)
        tot_hr += mul*hr
        tot_tr += mul*tr
        mul *= discount(mdp)
        s = sp
    end
    return tot_hr, tot_tr, actions
end

function get_counts_mean(μ, σ2)
    β = (μ*(1-μ)^2)/σ2 + μ - 1
    α = β*μ/(1-μ)
    return α, β
end

function get_counts_mode(μ, σ2)
    r = roots([2*σ2+μ*(μ-1), 5*σ2-(1-2*μ + 2*μ^2), 4*σ2 + μ*(μ -1), σ2])
    z = maximum(real.(r[(abs.(imag.(r)) .<= 1e-12) .& (real.(r) .> 0.)]))
    β = (1-μ)*z + μ
    ϵ = μ*(β-1) / (1-μ)
    α = 1+ ϵ
    return α, β
end

function get_upper_lower_bounds(a,b, ci = 0.9)
    ϵ = (1-ci) / 2
    d = Beta(a,b)
    return quantile(d, ϵ), quantile(d, 1-ϵ)
end




# Assumes the probability of failure is Beta distributed, updates estimates based
# on effective number of samples. Returns mean and variance
function fit_beta(samples, weights; prior = 0)
    weights ./= sum(weights)

    Neff = 1/sum(weights.^2)
    pf = sum(samples .* weights)
    positives = pf * Neff
    negatives = Neff - positives

    prior = pf
    v = 0
    for i=1:10
        if pf == 0
            pα, pβ = 0.5, 0.5
        else
            pα, pβ = prior, 1-prior
        end

        b = Beta(positives + pα, negatives + pβ)
        prior, v = mean(b), var(b)
    end
    return prior, v
    # mean(weights.*samples), var(weights .* samples) / Neff
end

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
function run_mcts_trial(planner, Nmax; failure_prob_fns = [failure_prob])
    @assert planner.tree == nothing || planner.tree.s_lookup == nothing || length(planner.tree.s_lookup) == 0

    Nstep = planner.solver.n_iterations
    trials = [Trial() for t in 1:length(failure_prob_fns)]
    println("running MCTS trial to i=", Nmax, "....")
    for i in 1:Nstep:Nmax-Nstep
        print("i=",i,"    ")
        action(planner, [s0])

        for f in 1:length(failure_prob_fns)
            trial = trials[f]
            E, V = failure_prob_fns[f](planner.tree)
            # E,V = fit_beta(samples, weights)

            push!(trial.mean_arr, E)
            push!(trial.var_arr, V)
            push!(trial.num_sanodes, length(planner.tree.q))
            push!(trial.num_snodes, length(planner.tree.s_lookup))
            push!(trial.pts, i + Nstep - 1)
            push!(trial.num_failures, length(planner.tree.action_sequences))
        end
    end
    println("")
    trials, planner
end

# Runs a Monte Carlo style rollout to get a mean and variance
# rollout: This is a function that returns a (potentially weighted) failure sample
# Nmax: This is the total number of iterations to run the trial for
# Nstep: This is the interval between data acquisition
function run_rollout_trial(rollout, Nmax, Nstep)
    println("running rollout trial to i=", Nmax, "....")
    samples, weights, trial = Float64[], Float64[], Trial()
    for i = 1:Nstep:Nmax-Nstep
        print("i=",i,"    ")
        for j=1:Nstep
            samp, w = rollout()
            push!(samples, samp)
            push!(weights, w)
        end
        # E, V = fit_beta(samples, weights ./ length(weights))
        E,V = mean(samples), var(samples) / length(samples)
        push!(trial.mean_arr, E)
        push!(trial.var_arr, V)
        push!(trial.pts, i + Nstep - 1)
        push!(trial.num_failures, sum(samples .> 0))
    end
    println("")
    trial, samples, weights
end

# Constructs a planner for the provided mdp
# Nstep is the number of iterations that MCTS is run each time action is called
function get_planner(mdp, Nstep; exploration_constant = 0.2, sample_ucb = false, k_action = 1., alpha_action = .1, skip_terminal_branches = false)
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
                       keep_tree = true,
                       skip_terminal_branches = skip_terminal_branches
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

