using POMDPs
using Test
include("../adversarial_continuum_world.jl")
include("../continuous_policy_evaluation.jl")
include("../cworld_vis.jl")
w = CWorld()
w_fail = CWorld(rewards = [1,1,0,0])

dummy_s = Vec2(0.0, 0.0)
dummy_a = Vec2(0.0, 0.0)
plot(CWorldVis(w, f=sp->reward(w,dummy_s,dummy_a,sp)))

# Solve for the optimal policy
nx = 30
ny = 30
g = RectangleGrid(range(w.xlim[1], stop=w.xlim[2], length=nx), range(w.ylim[1], stop=w.ylim[2], length=ny))
sol = CWorldSolver(max_iters=20, m=20, grid=g)
policy =  solve(sol, w)

# Evaluate that policy in great detail
nx = 50
ny = 50
g = RectangleGrid(range(w.xlim[1], stop=w.xlim[2], length=nx), range(w.ylim[1], stop=w.ylim[2], length=ny))
sol = CWorldSolver(max_iters=50, m=500, grid=g)
V = policy_eval(sol, w_fail, policy)
plot(CWorldVis(w_fail, f=s->evaluate(V[end],s)))

positions_x = []
positions_y = []
s = dummy_s
while !isterminal(w,s)
    println("s: ", s)
    push!(positions_x, s[1])
    push!(positions_y, s[2])
    a = action(policy, s)
    global s = generate_s(w, s, a, Random.GLOBAL_RNG)
end

xx, yy, u, v = [], [], [], []
for x in range(w.xlim[1], stop=w.xlim[2], length=nx),  y in range(w.ylim[1], stop=w.ylim[2], length=ny)
    push!(xx, x)
    push!(yy, y)
    a =
    push!(u, )

policy
plot!(positions_x, positions_y, marker = true, markersize=2)

## Test adversarial_continuum_world
acworld = AdversarialCWorld(w,policy,MvNormal([0.,0.], [1. 0.; 0. 1]))
s0 = Vec2(0,0)

s1, r = generate_sr(acworld, s0, Vec2(2.5,2.5), Random.GLOBAL_RNG)
@test shortest_distance_to_error(s1, acworld) == 0
@test r == 1e4 + 1





