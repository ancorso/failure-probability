using Plots; gr()
using Statistics
include("gridworld.jl")
include("value_iteration.jl")

w = 4
h = 4

# Define a reward function
r = zeros(w*h)
r[state((3,3), w)] = 1
r[state((2,1), w)] = -1
terminals = findall(r.!= 0)

# Define a transition function
P = gen_P(w, h, terminals, [0.1, 0.1, 0.1, 0.1, 0])

# Put it together in a gridworld
g = GridWorld(w,h,r,P,0.8)

# Get a policy to evaluate
_, π = value_iteration(g)
display_gridworld(g,policy_to_annotations(π))

# update the reward function to the new one (for failure probability)
g.r[state((3,3), w)] = 0
g.r[state((2,1), w)] = 1
g.γ = 1

V = policy_evaluation(g,π,tol=1e-15)

display_gridworld(g, string.(round.(V, digits=2)), title = "Prob of Failure under π")

xb, xbprv, M = 0, 0, 0
N = 1000
meanN, varN, totalReturn = Float64[], Float64[], Float64[]
for n=1:N
    x = rollout(g, 1, π)
    global xbprev = xb
    global xb = xbprev + (x - xbprev) / n
    global M = M + (x - xbprev)*(x - xb)
    s2 = M/(n-1)
    sm2 = s2/n

    push!(meanN, xb)
    push!(varN, sm2)
    if isempty(totalReturn)
        push!(totalReturn, x)
    else
        push!(totalReturn, totalReturn[end] + x)
    end
end


p1 = plot(meanN, label="Estimator", ribbon=2.78*sqrt.(varN), legend=:bottomright)
plot!(1:N, fill(V[1],N), label="Exact")
plot!([],[],color=:black, label="Total Failures")
p1_shared = twinx(p1)
plot!(p1_shared, totalReturn, label="", color=:black, linestyle=:dash, xaxis=false)

