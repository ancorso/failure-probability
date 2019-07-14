function policy_evaluation(g::GridWorld, policy; tol=1e-3)
	vp = zeros(nS(g))
	v = vp .+ Inf
	while maximum(abs.(v .- vp)) > tol
		v .= vp
		for s in 1:nS(g)
			a = policy[s]
			next_states = g.P[s, a]
			vp[s] = 0
			for (sp, prob) in next_states
				vp[s] += prob*g.r[sp] + g.γ*prob*v[sp]
            end
        end
    end
	vp
end

function value_iteration(g::GridWorld; tol=1e-3)
    Qp = zeros(nS(g), 4)
    Q = Qp .+ Inf
    while maximum(abs.(Q .- Qp)) > tol
    	Q .= Qp
    	for s in 1:nS(g), a in 1:4
    		next_states = g.P[s,a]
    		Qp[s,a] = 0
    		for (sp, prob) in next_states
    			Qp[s,a] += prob * g.r[sp] + g.γ * prob * maximum(Q[sp,:])
            end
        end
    end
    policy = [argmax(Q[i,:]) for i in 1:size(Q,1)]

    return Qp, policy
end

