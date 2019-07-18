using Test
include("mcts.jl")
include("adversarial_simulator.jl")
include("value_iteration.jl")

# Construct the adversarial simulator that we are going to use
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

# Node generation
root = Node(-1, 1, nothing)
@test isempty(root.children)
@test root.Q == 0
@test root.n == 0

# Check fully expanded
@test fully_expanded(root, sim) == false

# Check "add_child" and pick_univisted
root.children
tot = 10
first_a = nothing
while !fully_expanded(root, sim)
    new_node = pick_unvisited(root, sim)
    @test new_node.parent === root
    @test root.children[new_node.a] === new_node
    if first_a == nothing
        global first_a = new_node.a
    end
    new_node.Q = tot
    new_node.n = 1
    root.n += 1
    global tot -= 1
end
first_a
@test length(root.children) ==  length(actions_from(root.s, sim))
@test fully_expanded(root, sim)
@test num_leaves(root) == 3

# check traverse and uct
new_node = traverse(root, sim)
new_node
try
    @test new_node.parent.parent === root
    @test new_node.parent.a == first_a
    @test new_node.n == 0
catch
    @test new_node.parent === root
    @test terminal(new_node.s, sim)
end

old_n = new_node.n
backpropagate(new_node, 1, sim)
@test new_node.n == old_n + 1


root = Node(16, 16, nothing)
mcts(root, sim, 100, 1)


root

