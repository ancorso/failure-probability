using Plots
using Distributions
using Serialization
include("../adversarial_continuum_world.jl")
include("../failure_prob.jl")
include("plot_tree.jl")

normalize_1(v) = v ./ sum(v)
softmax(v) = normalize_1(exp.(v))

function myrollout_2r(mdp, s, depth)

    tot_r, tot_r2, mul = 0, 0, discount(mdp)

    while !isterminal(mdp, s)
        ract, ρ = random_action(mdp, s, nothing)
        sp, r, r2 = generate_sr(nothing, mdp, s, ract, Random.GLOBAL_RNG)
        tot_r += mul*r
        tot_r2 += mul*r2
        mul *= discount(mdp)
        s = sp
    end
    return ast_reward(s, mdp), tot_r2
end

function choose_action_model(mdp::AdversarialCWorld, s::Array{Vec2}, snode)
    tree = snode.tree
    cs = children(snode)
    N = length(cs)
    if N < 3
        return Vec2(rand(mdp.action_dist)), 1.
    end
    N = min(N, 5)
    qs = tree.q2[cs]
    if all(qs .== 0)
        return Vec2(rand(mdp.action_dist)), 1.
    end
    as = tree.a_labels[cs]

    probs = [pdf(mdp.w.disturbance_dist, asi) for asi in as]
    V = qs .* probs
    order = sortperm(V, rev = true)[1:N]
    w = softmax(10*normalize_1(V[order]))
    avec2d = convert(Array{Float64}, hcat(as[order]...))

    P = fit_mle(MvNormal, avec2d, w)
    a = SVector{2,Float64}(rand(P))
    ρ = pdf(mdp.w.disturbance_dist, a) / pdf(P, a)
    return a, ρ
end

# Starting state for the continuum gridworld
s0 = Vec2(7, 5)

# Construct the mdp and the result with value iteration
mdp, V, w_fail = create_sim(σ2 = 0.03, is_σ2 = 0.03, solver_m = 100, max_itrs = 100)

V0 = evaluate(V, s0)
# println(V0)
# V0 = 6.539515625820375e-12


# Sample Solving
# Construct the solver with different options
Nstep = 100
dpw_solver = DPWSolver(n_iterations=Nstep,
                       depth=100,
                       exploration_constant=.3,
                       k_state = .9,
                       alpha_state = 0.,
                       check_repeat_state = false,
                       estimate_value = myrollout_2r,
                       next_action = random_action,
                       sample_ucb = false,
                       k_action = 1.,
                       alpha_action = .1,
                       double_reward = true,
                       keep_tree = true
                       )

dpw_planner = solve(dpw_solver, mdp)
# action(dpw_planner, [s0])
# mean(length.(dpw_planner.tree.children))
# length(dpw_planner.tree.q)

Nmax = 50000
pts = 1:Nstep:Nmax
t = run_mcts_trial(Nmax, Nstep, dpw_planner)


nonzero = t.mean_arr .> 0.
p1 = plot(pts[nonzero], t.mean_arr[nonzero], yscale = :log, ribbon = (0, 2.98sqrt.(t.var_arr[nonzero])), label = "MCTS Estimate", ylabel = "Failure Probability")
plot!(pts[nonzero], fill(V0, sum(nonzero)), label = "true")


p2 = plot(pts, t.num_sanodes, label = "State-Action Nodes", legend = :bottomright, ylabel = "Number of Nodes")
plot!(pts, t.num_snodes, label = "State Nodes")

plot(p1, p2, layout = (2,1))
savefig("Estimate}_over_time.pdf")
# plot_path(mdp, s_list)
plot_tree(dpw_planner.tree, mdp)

# savefig("path_sampleUCB_modelA.pdf")
E, Var = failure_prob(dpw_planner.tree)
σ = sqrt(Var)
dpw_planner.tree.children
dpw_planner.tree.q2[dpw_planner.tree.children[1]]



# Loop over different MCTS types
function get_planners(Nstep)
    MCTS_planners = Dict()
    for action_model = [random_action, choose_action_model]
        for sample_ucb = [true, false]
            dpw_solver = DPWSolver(n_iterations=Nstep,
                                   keep_tree=true,
                                   depth=100,
                                   exploration_constant=.2,
                                   k_state = .9,
                                   alpha_state = 0.,
                                   check_repeat_state = false,
                                   estimate_value = myrollout_2r,
                                   next_action = action_model,
                                   sample_ucb = sample_ucb,
                                   k_action = 1.,
                                   alpha_action = .1,
                                   double_reward = true,
                                   )

             name = string("MCTS_", (sample_ucb) ? "sampleUCB" : "maxUCB", "_", (action_model == random_action) ? "randA" : "modelA")
             dpw_planner = solve(dpw_solver, mdp)
             MCTS_planners[name] = dpw_planner
         end
    end
    return MCTS_planners
end

mutable struct Trial
    mean_arr::Array{Float64}
    var_arr::Array{Float64}
    num_sanodes::Array{Int}
    num_snodes::Array{Int}
end

function run_mc_trial(Nmax, Nstep)
    vals = []
    trial = Trial([],[],[],[])

    for i = 1:Nstep:Nmax
        push!(vals, [policy_rollout(w_fail, s0, mdp.policy) for j=1:Nstep]...)
        push!(trial.mean_arr, mean(vals))
        push!(trial.var_arr, var(vals)/i)
    end
    return trial
end


function run_mcts_trial(Nmax, Nstep, planner)
    trial = Trial([],[],[],[])
    for i in 1:Nstep:Nmax
        action(planner, [s0])
        E, Var = failure_prob(planner.tree)
        push!(trial.mean_arr, E)
        push!(trial.var_arr, Var)
        push!(trial.num_sanodes, length(planner.tree.q))
        push!(trial.num_snodes, length(planner.tree.s_lookup))
    end
    return trial
end


function add_data!(data, name, trial)
    if !haskey(data, name)
        data[name] = Trial[]
    end
    push!(data[name], trial)
end

data = Dict{String, Array{Trial}}()
Nmax = 50000
Nstep = 100

# Solve the mdp
for i=1:10
    MCTS_planners = get_planners(Nstep)
    println("Trial: ", i)
    println("Running MC")
    mc_res = run_mc_trial(Nmax, Nstep)
    add_data!(data, "MC", mc_res)

    for k in keys(MCTS_planners)
        println("Running: ", k)
        mcts_res = run_mcts_trial(Nmax, Nstep, MCTS_planners[k])
        add_data!(data, k, mcts_res)
    end
end

combined_data = Dict{String, Trial}()
k = collect(keys(data))[1]
data[k]
data
for k in keys(data)
    println(typeof(k))

    Nt = length(data[k])
    combined_data[k] = data[k][1]
    for t in 2:Nt
        combined_data[k].mean_arr = combined_data[k].mean_arr .+ data[k][t].mean_arr
        combined_data[k].var_arr = combined_data[k].var_arr .+ data[k][t].var_arr
    end
    combined_data[k].mean_arr = combined_data[k].mean_arr./Nt
    combined_data[k].var_arr = combined_data[k].var_arr./Nt
end

trial = 1
x = 1:Nstep:Nmax
p= plot(x, V0*ones(length(x)),label="exact", yscale=:log, legend = :topleft)
for e in combined_data
    dy = sqrt.(e[2].var_arr)
    y = e[2].mean_arr
    y[y.==0] .= NaN
    plot!(x, y, label = e[1], ribbon = (0, dy))
end

sqrt.(data["MCTS_sampleUCB_randA"][1].var_arr)
display(p)

savefig("comparison_longer_horizon.pdf")
data

trial = 4
x = 1:Nstep:Nmax
p= plot(x, V0*ones(length(x)),label="exact", yscale=:log, legend = :topleft)
for e in data
    dy = sqrt.(e[2][trial].var_arr)
    y = e[2][trial].mean_arr
    y[y.==0] .= NaN
    plot!(x, y, label = e[1], ribbon = (0, dy))
end

display(p)

