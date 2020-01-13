using Plots; gr()

function get_states_from_rollout(mdp, s0, actions)
    s = [s0]
    for a in actions
        s, hr, tr = generate_sr(nothing, mdp, s, a, Random.GLOBAL_RNG)
    end
    s
end


plot_cworld(mdp) = plot(CWorldVis(mdp.w, f=sp->reward(mdp.w, Vec2(0, 0),Vec2(0, 0), sp)))

# Plots a sequence of states on the continuum world that is a part of the mdp
function plot_path!(cworld_plot, path, label="")
    x = [p[1] for p in path]
    y = [p[2] for p in path]
    plot!(cworld_plot, x,y, markersize=3, markerstrokealpha=0, label=label)
end

# Plots a bunch of paths on the same continuum world vis
function plot_action_paths(mdp, paths, s0)
    cworld_plot = plot_cworld(mdp)
    for p in paths
        plot_path!(cworld_plot, get_states_from_rollout(mdp, s0, p))
    end
    cworld_plot
end



# Plots the states in the tree on the continuum world in the mdp
function plot_tree(tree, mdp)
    # Get the state locations of each state in the tree and plot
    x = [s[end][1] for s in tree.s_labels]
    y = [s[end][2] for s in tree.s_labels]
    V = NaN.*zeros(length(x))
    for i = 1:length(tree.q)
        qval = tree.q[i]
        if length(tree.transitions2[i]) > 0
            s = tree.transitions2[i][1][1]
            V[s] = qval
        end
    end

    # Scale the value function to be between -1,1
    V[1] = minimum(V[2:end])
    V = 2 .*((V .- minimum(V)) / (maximum(V) - minimum(V)) .- 0.5)

    p = plot(CWorldVis(mdp.w, f=sp->reward(mdp.w, Vec2(0, 0),Vec2(0, 0), sp)))
    scatter!(x,y, marker_z = V , markersize=2, markerstrokealpha=0, clims = (minimum(V), maximum(V)), label = "")
end

function plot_metric_comparisons(trials, labels, V_exact)
    minpts, maxpts = extrema(hcat([t.pts for t in trials]...))
    p1 = plot(range(minpts, stop=maxpts, length=1000), fill(V_exact, 1000), title = "Failure Probabilty Estimation", ylabel = "Failure Probability", xlabel = "Number of Samples", legend = :outertopright, yscale = :log10, xscale=:log10, label = "Exact")

    p2 = plot([minpts], [0], label="", title = "Number of Failure Samples vs. Total Samples", ylabel = "Number of Failure Samples", xlabel = "Number of Samples", legend = :outertopright, xscale = :log10)

    colors = distinguishable_colors(length(trials)+1, colorant"white")[2:end]
    for t in 1:length(trials)
        trial, label = trials[t], labels[t]
        nonzero = trial.mean_arr .> 0.
        y = trial.mean_arr[nonzero]
        v = trial.var_arr[nonzero]

        bounds = [get_upper_lower_bounds(get_counts_mode(y[i], v[i])..., 0.99) for i in 1:length(y)]

        lb = [b[1] for b in bounds]
        ub = [b[2] for b in bounds]

        plot!(p1, trial.pts[nonzero], y, fillrange=ub, fillalpha = 0.2, label = label, color = colors[t])
        plot!(p1, trial.pts[nonzero], y, fillrange=lb, fillalpha = 0.2, color = colors[t], label="")
        plot!(p2, trial.pts, trial.num_failures, label = label, color = colors[t])
    end

    plot(p1, p2, layout = (2,1), size = (800,600))
end

# Plot the resuls of a trial. for now it shows the growth of the tree,
# as well as the accuracy of the prediction over time
# V_exact is the reference value for the probability of failure
function plot_trial(trial, V_exact, mdp, tree, s0)
    nonzero = trial.mean_arr .> 0.
    y = trial.mean_arr[nonzero]
    CI = 2.98*sqrt.(trial.var_arr[nonzero])
    p1 = plot(trial.pts[nonzero], y, fillrange=y.+CI, fillalpha = 0.2, yscale=:log10, label = "MCTS Estimate", ylabel = "Failure Probability", xlims = (0, trial.pts[end]), legend = :bottomleft)
    plot!(trial.pts[nonzero], fill(V0, sum(nonzero)), label = "True Value")


    p2 = plot(trial.pts, trial.num_sanodes, label = "State-Action Nodes", legend = :topright, ylabel = "Number of Nodes")
    plot!(trial.pts, trial.num_snodes, label = "State Nodes")

    p3 = plot(trial.pts, trial.num_failures, label = "Num Failures", legend = :topright, ylabel = "")

    p4 = plot(p1, p2, p3, layout = (3,1))

    p5 = plot_action_paths(mdp, tree.action_sequences, s0)
    p6 = plot_tree(tree, mdp)
    plot(p4, plot(p5,p6, layout = (2,1)), size = (2400, 800))
end

