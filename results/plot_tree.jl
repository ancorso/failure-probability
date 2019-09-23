function plot_path(mdp, path)
    x = [p[1] for p in path]
    y = [p[2] for p in path]
    plot(CWorldVis(mdp.w, f=sp->reward(mdp.w, Vec2(0, 0),Vec2(0, 0), sp)))
    plot!(x,y, markersize=3, markerstrokealpha=0)
end



function plot_tree(tree, mdp)
    # # Get the state locations of each state in the tree and plot
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
    V[1] = minimum(V[2:end])
    V = 2 .*((V .- minimum(V)) / (maximum(V) - minimum(V)) .- 0.5)


    plot(CWorldVis(mdp.w, f=sp->reward(mdp.w, Vec2(0, 0),Vec2(0, 0), sp)))
    scatter!(x,y, marker_z = V , markersize=3, markerstrokealpha=0, clims = (minimum(V), maximum(V)))
end