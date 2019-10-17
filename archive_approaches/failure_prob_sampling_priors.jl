using Statistics

# Computes the failure probability from the state at the root of the tree
function failure_prob_samples(tree; s=1, Nsamps = 1000, prior = Beta(0.5, 0.5))
    sa_children = tree.children[s]
    if isempty(sa_children)
        warning("There were no state-action nodes in the tree, returning 0")
        return 0., 0.
    end
    failure_prob_sa_children_samples(sa_children, tree, Nsamps = Nsamps, prior=prior)
end

# Returns the failure probability averaging across a vector of sa nodes
function failure_prob_sa_children_samples(sa_children, tree, sa = nothing; Nsamps = 1000, prior = Beta(0.5,0.5))
    samps = zeros(Nsamps)
    for c in sa_children
        samps .+= failure_prob_sanode_samples(c, tree, Nsamps = Nsamps, prior = prior)
    end
    samps ./ length(sa_children)
end


# Returns the failure probability of an sa node in the tree
function failure_prob_sanode_samples(sa, tree; Nsamps = 1000, prior = Beta(0.5,0.5))
    # First check if there have been any transitions from this sanode, if not then return its val
    v = tree.q2[sa]
    dist = Beta(prior.α + v, prior.β + (1-v))
    (tree.n_a_children[sa] == 0) && return rand(dist, Nsamps)

    # In state s, taking action a, gives us state sp
    sp = tree.transitions2[sa][1][1]

    # if sp does not have any children then it does not have an estimate for q, so take the estimate of the sa node
    sa_children = tree.children[sp]
    isempty(sa_children) && return rand(dist, Nsamps)

    # Otherwise compute the average failure probability of the children
    failure_prob_sa_children_sampls(sa_children, tree, sa, Nsamps = Nsamps, prior=prior)
end

