using Test
using POMDPs
using MCTS
using Random
include("../failure_prob.jl")

mutable struct TestMDP <: MDP{Tuple{Int,Int}, Int}
    win_at
    depth
end


function POMDPs.generate_sr(n::Nothing, mdp::TestMDP, s::Tuple{Int,Int}, a::Int, rng::Random.AbstractRNG)
     ((s[1]+a, s[2]+1), a == 4, s[1]+a >= mdp.win_at)
end

POMDPs.actions(mdp::TestMDP) = [1,2,3,4]
POMDPs.n_actions(mdp::TestMDP) = 4
POMDPs.discount(mdp::TestMDP) = 1
POMDPs.isterminal(mdp::TestMDP, s::Tuple{Int, Int}) = s[2] == mdp.depth
random_action(mdp::TestMDP, s::Tuple{Int,Int}, snode) = rand(actions(mdp)), 1.

dpw_solver = DPWSolver(n_iterations=5000,
                       k_state = .9,
                       alpha_state = 0.,
                       check_repeat_state = false,
                       estimate_value = double_rollout,
                       next_action = random_action,
                       double_reward = true,
                       exploration_constant = 1.,
                       k_action = 3.,
                       alpha_action = .5
                       )

mdp = TestMDP(4, 1)
dpw_planner = solve(dpw_solver, mdp)
action(dpw_planner, (0, 0))


e1, v1 = failure_prob(dpw_planner.tree)
@test e1 == 0.25


mdp = TestMDP(12, 3)
dpw_planner = solve(dpw_solver, mdp)
action(dpw_planner, (0, 0))

e1, v1 = failure_prob(dpw_planner.tree)
@test e1 == 0.25*.25*.25

