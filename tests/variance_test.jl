using Distributions
using Plots
using Statistics


pxy = MvNormal([0.,0.], [0.5 0; 0 0.5])
px = Normal(0., 0.5)
py = Normal(0., 0.5)


f(x)  = x[1] .> 0 && x[2] .> 0

pts = [rand(pxy) for i=1:1000]
F = f.(pts)
var(F)
mean(F)

scatter([pt[1] for pt in pts], [pt[2] for pt in pts])

all_means = Float64[]
estimated_var_of_mean = Float64[]
for  trial = 1:1000
    data = Dict{Float64, Array{Float64}}()
    for i = 1:50000
        N = length(data)
        r = rand()
        if r < (1/(N+1))
            x, y = rand(px), rand(py)
            data[x] = [f([x,y])]
        else
            x, y = rand(keys(data)), rand(py)
            push!(data[x], f([x,y]))
        end
    end

    varx = []
    meanx = []
    Nys = []
    Nx = length(data)
    for k in sort(collect(keys(data)))
        v = var(data[k])
        if isnan(v)
            v = 0
        end
        push!(Nys, length(data[k]))
        push!(varx, v)
        push!(meanx, mean(data[k]))
    end

    # plot(meanx, yerror=varx)
    # plot(Nys)

    mean(varx)
    push!(all_means, mean(meanx))
    push!(estimated_var_of_mean, (mean(varx./Nys) + var(meanx)) / Nx)
end

var(all_means)
mean(estimated_var_of_mean)
var(estimated_var_of_mean)




