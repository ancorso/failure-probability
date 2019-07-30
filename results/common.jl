mutable struct Trial
    method
    tree_size
    n_iterations
    failure_prob
    var_failure_prob
end

function add_data!(data, trial)
    if !haskey(data, trial.method)
        data[trial.method] = OrderedDict{Int, Array{Trial}}()
    end
    dict = data[trial.method]
    if !haskey(dict, trial.n_iterations)
        dict[trial.n_iterations] = Trial[]
    end
    arr = dict[trial.n_iterations]
    push!(arr, trial)
end