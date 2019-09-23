using LsqFit

Base.isinf(val::Array{Float64,1}) = any(isinf.(val))
Base.isnan(val::Array{Float64,1}) = any(isnan.(val))

function get_mvnormal_params(p)
    p[1], p[2:3], [1. / p[4]^2 0; 0 1. / p[5]^2]
end

function get_mvnormal_params(p, ndim)
    NΣ = ndim^2
    Σ_range, A_range, μ_range = 1:NΣ, NΣ + 1, NΣ+2:NΣ+1+ndim
    S = reshape(p[Σ_range], ndim, ndim)
    p[A_range], p[μ_range], inv(S*S')
end

function model(a, p)
    nparam, ndim, N = length(p), length(a[1]), length(a)
    out = zeros(N)
    A, μ, Σinv = get_mvnormal_params(p)
    for i=1:N
        out[i] =  A*exp(-0.5*(a[i] .- μ)'*Σinv*(a[i] .- μ))
    end
    out
end

as


fit_mle(MvNormal, hcat(as...),[1.,1.e-16,0.,0.,0.,0.,0.,0.,0.,0.])
as[1]

scatter3d(ax, ay, V)

