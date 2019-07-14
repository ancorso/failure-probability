include("gridworld.jl")

mutable struct AdversarialSimulator
    g::GridWorld
    s
    π
end

discount(sim) = sim.g.γ

function actions_from(s, sim)
    a = sim.π[s]
    possible_next_states = sim.g.P[s, a]
    sort([stup[1] for stup in possible_next_states])
end

function step(a, sim)
    sim.s = a
end

terminal(s, sim) = is_terminal(s, sim.g)

transition_prob(s, sp, sim) = transition_prob(s,sim.π[s],sp, sim.g)

in_E(s, sim) = sim.g.r[s] == -1

function reward(s, sp, sim)
    reward = log(transition_prob(s,sp, sim))
    if is_terminal(sp, sim.g) && !in_E(sp, sim)
        reward += -1000
    end
    reward
end

