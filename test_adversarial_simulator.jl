using Test
include("adversarial_simulator.jl")

w,h = 4,4
## Generate gridworld
# Define a reward function
r = zeros(w*h)
r[state((3,3), w)] = 1
r[state((2,1), w)] = -1
terminals = findall(r.!= 0)

# Define a transition function
P = gen_P(w, h, terminals, [0.001, 0.001, 0.001, 0.001, 0])

# Put it together in a gridworld
g = GridWorld(w,h,r,P,0.8)

# Get a policy to evaluate
_, πo = value_iteration(g)

g.γ = 1

sim = AdversarialSimulator(g, 1, πo)

@test sim.s == 1
@test discount(sim) == 1
@test actions_from(1, sim) == [1, 2, 5]
@test step(5, sim) == 5
@test sim.s == 5
@test !terminal(5, sim)
@test terminal(6, sim)

@test reward(5, 6, sim) == -1000 + log(0.001)
@test reward(5, 1, sim) == log(0.001)
@test reward(5, 9, sim) == log(0.997)

