normalize_1(v) = v ./ sum(v)
softmax(v) = normalize_1(exp.(v))

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