using POMDPs
using GridInterpolations
using StaticArrays
using LinearAlgebra
using Distributions
using Parameters
using Random


const Vec2 = SVector{2, Float64}

struct CircularRegion
    center::Vec2
    radius::Float64
end

Base.in(v::Vec2, r::CircularRegion) = LinearAlgebra.norm(v-r.center) <= r.radius

const card_and_stay = [Vec2(1.0, 0.0), Vec2(-1.0, 0.0), Vec2(0.0, 1.0), Vec2(0.0, -1.0), Vec2(0.0, 0.0)]
const cardinal = [Vec2(1.0, 0.0), Vec2(-1.0, 0.0), Vec2(0.0, 1.0), Vec2(0.0, -1.0)]
const default_regions = [CircularRegion(Vec2(3.5, 2.5), 0.5),
                         CircularRegion(Vec2(3.5, 5.5), 0.5),
                         CircularRegion(Vec2(8.5, 2.5), 0.5),
                         CircularRegion(Vec2(7.5, 7.5), 0.5)]
const default_rewards = [-1.0, -1.0, 1.0, 1.0]


@with_kw struct CWorld <: MDP{Vec2, Vec2}
    xlim::Tuple{Float64, Float64}                   = (0.0, 10.0)
    ylim::Tuple{Float64, Float64}                   = (0.0, 10.0)
    reward_regions::Vector{CircularRegion}          = default_regions
    rewards::Vector{Float64}                        = default_rewards
    terminal::Vector{CircularRegion}                = default_regions
    disturbance_dist                                = MvNormal([0.,0.], [0.5 0.0; 0.0 0.5])
    actions::Vector{Vec2}                           = cardinal
    discount::Float64                               = 1
end

POMDPs.actions(w::CWorld) = w.actions
POMDPs.n_actions(w::CWorld) = length(w.actions)
POMDPs.discount(w::CWorld) = w.discount

function POMDPs.generate_s(w::CWorld, s::AbstractVector, a::AbstractVector, rng::Random.AbstractRNG)
    move(w, s, a, rand(rng, w.disturbance_dist))
end

function policy_rollout(w, s0, policy)
    s = s0
    tot_r, mul = 0, discount(mdp)
    while !isterminal(w,s)
        a = action(policy, s)
        sp = generate_s(w, s, a, Random.GLOBAL_RNG)
        r = reward(w, s, a, sp)
        tot_r += mul*r
        mul *= discount(mdp)
        s = sp
    end
    return tot_r
end

function move(w, s, a, d)
    sp = s + 0.1*a + d
    xpos = max(min(sp[1], w.xlim[2]), w.xlim[1])
    ypos = max(min(sp[2], w.ylim[2]), w.ylim[1])
    Vec2([xpos, ypos])
end

function POMDPs.reward(w::CWorld, s::AbstractVector, a::AbstractVector, sp::AbstractVector) # XXX inefficient
    rew = 0.0
    for (i,r) in enumerate(w.reward_regions)
        if sp in r
            rew += w.rewards[i]
        end
    end
    return rew
end

function POMDPs.isterminal(w::CWorld, s::Vec2) # XXX inefficient
    for r in w.terminal
        if s in r
            return true
        end
    end
    return false
end

function POMDPs.initialstate(w::CWorld, rng::Random.AbstractRNG)
    x = w.xlim[1] + (w.xlim[2] - w.xlim[1]) * rand(rng)
    y = w.ylim[1] + (w.ylim[2] - w.ylim[1]) * rand(rng)
    return Vec2(x,y)
end

## Visualization
using Plots
mutable struct CWorldVis
    w::CWorld
    s::Union{Vec2, Nothing}
    f::Union{Function, Nothing}
    g::Union{AbstractGrid, Nothing}
    title::Union{String, Nothing}
end

function CWorldVis(w::CWorld;
                   s=nothing,
                   f=nothing,
                   g=nothing,
                   title=nothing)
    return CWorldVis(w, s, f, g, title)
end

@recipe function f(v::CWorldVis)
    xlim --> v.w.xlim
    ylim --> v.w.ylim
    aspect_ratio --> 1
    title --> something(v.title, "Continuum World")
    if v.f !== nothing
        @series begin
            f = v.f
            width = v.w.xlim[2]-v.w.xlim[1]
            height = v.w.ylim[2]-v.w.ylim[1]
            n = 200 # number of pixels
            nx = round(Int, sqrt(n^2*width/height))
            ny = round(Int, sqrt(n^2*height/width))
            xs = range(v.w.xlim[1], stop=v.w.xlim[2], length=nx)
            ys = range(v.w.ylim[1], stop=v.w.ylim[2], length=ny)
            zg = Array{Float64}(undef, nx, ny)
            for i in 1:nx
                for j in 1:ny
                    zg[j,i] = f(Vec2(xs[i], ys[j]))
                end
            end
            color --> cgrad([:red, :white, :green])
            seriestype := :heatmap
            xs, ys, zg
        end
    end
    if v.g !== nothing
        @series begin
            g = v.g
            xs = collect(ind2x(g, i)[1] for i in 1:length(g))
            ys = collect(ind2x(g, i)[2] for i in 1:length(g))
            label --> "Grid"
            marker --> :+
            markercolor --> :blue
            seriestype := :scatter
            xs, ys
        end
    end
end

Base.show(io::IO, m::MIME, v::CWorldVis) = show(io, m, plot(v))
Base.show(io::IO, m::MIME"text/plain", v::CWorldVis) = println(io, v)

### Policy eval and solving
using GridInterpolations

struct GIValue{G <: AbstractGrid}
    grid::G
    gdata::Vector{Float64}
end

struct CWorldPolicy{V} <: Policy
    actions::Vector{Vec2}
    Qs::Vector{V}
end

evaluate(v::GIValue, s::AbstractVector{Float64}) = interpolate(v.grid, v.gdata, convert(Vector{Float64}, s))

@with_kw mutable struct CWorldSolver{G<:AbstractGrid, RNG<:Random.AbstractRNG} <: Solver
    grid::G                     = RectangleGrid(range(0.0, stop=10.0, length=30), range(0.0, stop=10.0, length=30))
    max_iters::Int              = 50
    tol::Float64                = 0.01
    m::Int                      = 20
    value_hist::AbstractVector  = []
    rng::RNG                    = Random.GLOBAL_RNG
end

struct CWorldPolicy{V} <: Policy
    actions::Vector{Vec2}
    Qs::Vector{V}
end

function policy_eval(sol::CWorldSolver, w::CWorld, policy)
    sol.value_hist = []
    data = zeros(length(sol.grid))
    val = GIValue(sol.grid, data)

    for k in 1:sol.max_iters
        newdata = similar(data)
        for i in 1:length(sol.grid)
            s = Vec2(ind2x(sol.grid, i))
            if isterminal(w, s)
                newdata[i] = 0.0
            else
                Qsum = 0.0
                a = action(policy, s)
                for j in 1:sol.m
                    sp, r = generate_sr(w, s, a, sol.rng)
                    Qsum += r + discount(w)*evaluate(val, sp)
                end
                newdata[i] = Qsum/sol.m
            end
        end
        push!(sol.value_hist, val)
        print("\rfinished iteration $k")
        val = GIValue(sol.grid, newdata)
    end
    return sol.value_hist
end

function POMDPs.solve(sol::CWorldSolver, w::CWorld)
    sol.value_hist = []
    data = zeros(length(sol.grid))
    val = GIValue(sol.grid, data)

    for k in 1:sol.max_iters
        newdata = similar(data)
        for i in 1:length(sol.grid)
            s = Vec2(ind2x(sol.grid, i))
            if isterminal(w, s)
                newdata[i] = 0.0
            else
                best_Qsum = -Inf
                for a in actions(w, s)
                    Qsum = 0.0
                    for j in 1:sol.m
                        sp, r = generate_sr(w, s, a, sol.rng)
                        Qsum += r + discount(w)*evaluate(val, sp)
                    end
                    best_Qsum = max(best_Qsum, Qsum)
                end
                newdata[i] = best_Qsum/sol.m
            end
        end
        push!(sol.value_hist, val)
        print("\rfinished iteration $k")
        val = GIValue(sol.grid, newdata)
    end

    print("\nextracting policy...     ")

    Qs = Vector{GIValue}(undef,n_actions(w))
    acts = collect(actions(w))
    for j in 1:n_actions(w)
        a = acts[j]
        qdata = similar(val.gdata)
        for i in 1:length(sol.grid)
            s = Vec2(ind2x(sol.grid, i))
            if isterminal(w, s)
                qdata[i] = 0.0
            else
                Qsum = 0.0
                for k in 1:sol.m
                    sp, r = generate_sr(w, s, a, sol.rng)
                    Qsum += r + discount(w)*evaluate(val, sp)
                end
                qdata[i] = Qsum/sol.m
            end
        end
        Qs[j] = GIValue(sol.grid, qdata)
    end
    println("done.")

    return CWorldPolicy(acts, Qs)
end

function POMDPs.action(p::CWorldPolicy, s::AbstractVector{Float64})
    best = action_ind(p, s)
    return p.actions[best]
end

action_ind(p::CWorldPolicy, s::AbstractVector{Float64}) = argmax([evaluate(Q, s) for Q in p.Qs])

POMDPs.value(p::CWorldPolicy, s::AbstractVector{Float64}) = maximum([evaluate(Q, s) for Q in p.Qs])



### Adversarial Continuum Worlds

struct AdversarialCWorld <: MDP{Array{Vec2}, Vec2}
    w::CWorld
    policy
    action_dist
end

function POMDPs.generate_sr(mdp::AdversarialCWorld, s::Array{Vec2}, a::Vec2, rng::Random.AbstractRNG)
    cworld_action = action(mdp.policy, s[end])
    sp = move(mdp.w, s[end], cworld_action, a)
    new_s = [s..., sp]
    return new_s, heuristic_reward(new_s, mdp)
end

function POMDPs.generate_sr(v::Nothing, mdp::AdversarialCWorld, s::Array{Vec2}, a::Vec2, rng::Random.AbstractRNG)
    cworld_action = action(mdp.policy, s[end])
    sp = move(mdp.w, s[end], cworld_action, a)
    r2 = reward(mdp.w, s[end], a, sp) == -1
    new_s = [s..., sp]
    return new_s, heuristic_reward(new_s, mdp), r2
end

function action_taken(mdp::AdversarialCWorld, s::Array{Vec2}, sp::Array{Vec2})
    cworld_action = action(mdp.policy, s[end])
    sp[end] - (s[end] + cworld_action) # Compute the disturbance
end

imp_samp_weight(mdp::AdversarialCWorld, a::Vec2) = pdf(mdp.w.disturbance_dist, a) / pdf(mdp.action_dist, a)

function random_action(mdp::AdversarialCWorld, s::Array{Vec2}, snode)
    a = Vec2(rand(mdp.action_dist))
    ρ = imp_samp_weight(mdp, a)
    return a, ρ
end

rollout_weight(mdp::AdversarialCWorld, s::Array{Vec2}, sp::Array{Vec2}) = imp_samp_weight(mdp, action_taken(mdp, s, sp))

tree_weight(mdp::AdversarialCWorld, Nc, s::Array{Vec2}, sp::Array{Vec2}) = imp_samp_weight(mdp, action_taken(mdp, s, sp))

POMDPs.discount(mdp::AdversarialCWorld) = 1

POMDPs.isterminal(mdp::AdversarialCWorld, s::Array{Vec2}) = isterminal(mdp.w, s[end])

in_E(s, mdp) = reward(mdp.w, Vec2(0., 0.), Vec2(0., 0.), s) == -1.

function shortest_distance_to_error(sp, mdp)
    min_dist = Inf
    for i in 1:length(mdp.w.rewards)
        r = mdp.w.rewards[i]
        if r == -1.
            dist = sqrt((sp[1] - mdp.w.reward_regions[i].center[1])^2 + (sp[2] - mdp.w.reward_regions[i].center[2])^2)
            if dist < min_dist
                min_dist = dist
            end
        end
    end
    return min_dist
end

function heuristic_reward(s_traj, sim)
    isterm = isterminal(sim, s_traj)
    if  isterm && in_E(s_traj[end], sim)
        return 1.
    elseif isterm && !in_E(s_traj[end], sim)
        min_distance = minimum([shortest_distance_to_error(s, sim) for s in s_traj]) / sqrt((sim.w.xlim[2] - sim.w.xlim[1])^2 + (sim.w.ylim[2] - sim.w.ylim[1])^2)
        return exp(-min_distance)
    else
        return 0
    end
end



    # reward = 1. / (1+shortest_distance_to_error(sp, sim))#log(transition_prob(s, sp, sim))
    # # reward = shortest_distance_to_error(s, sim) - shortest_distance_to_error(sp, sim)
    # if isterminal(sim.w, sp) && !in_E(sp, sim)
    #     reward += -10000
    # end
    # if isterminal(sim.w, sp) && in_E(sp, sim)
    #     reward += 10000
    # end
    # reward
# end


function create_adversarial_cworld(;σ2 = 0.5, is_σ2 = 1.0, solver_m = 500, max_itrs = 50)
    w = CWorld(disturbance_dist = MvNormal([0.,0.], [σ2 0.0; 0.0 σ2]))
    w_fail = CWorld(rewards = [1, 1, 0, 0], disturbance_dist = MvNormal([0.,0.], [σ2 0.0; 0.0 σ2]))

    # Solve for the optimal policy
    println("solving for optimal policy...")
    g = RectangleGrid(range(w.xlim[1], stop=w.xlim[2], length=30), range(w.ylim[1], stop=w.ylim[2], length=30))
    sol = CWorldSolver(max_iters=50, m=20, grid=g)
    policy = solve(sol, w)

    # Evaluate that policy in great detail (to get prob of failure)
    println("Evaluating the probability of failure")
    g = RectangleGrid(range(w.xlim[1], stop=w.xlim[2], length=50), range(w.ylim[1], stop=w.ylim[2], length=50))
    sol = CWorldSolver(max_iters=max_itrs, m=solver_m, grid=g)
    V = policy_eval(sol, w_fail, policy)

    AdversarialCWorld(w,policy,MvNormal([0.,0.], [is_σ2 0.; 0. is_σ2])), V[end], w_fail
end

