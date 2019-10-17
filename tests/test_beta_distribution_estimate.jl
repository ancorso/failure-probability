using Distributions
using Plots

b1 = Beta(200000.5,0.5)
b2 = Beta(.5,2000.5)
b3 = sum_betas([0.5, 0.5], [b1, b2])

b4 = [rand(b1) + rand(b2) for i=1:1000]
var(b4)

x = 0.00001:0.001:0.9999


v1 = [pdf(b1, xi) for xi in x]
v2 = [pdf(b2, xi) for xi in x]

v3 = conv(v1, v2)
x2 = range(0,stop=1,length = length(v3))
dx = x2[2] - x2[1]
v3 = v3 ./ (sum(v3) * dx)


plot(x, [pdf(b1, xi) for xi in x], label="B1", ylim=(0,10))
plot!(x, [pdf(b2, xi) for xi in x], label="B2")
plot!(x, [pdf(b3, xi) for xi in x], label="Approx Convolution")
plot!(x2, v3, label="numerical")

histogram!(b4./2, xlims = (0,1), normed=true, bins=20)

v3

k = -100:100
xk = tanh.(0.5*π*sinh.(k*0.01))
println(xk[1])
scatter(xk, ones(size(xk)))

sqrt(var(b1))
v1 = [pdf(b1, xi) for xi in x]
plot(v1)

plot(abs.(ifft(fft(v1).*fft(v1))))

out = abs.(ifft(fft(v1).*fft(v1)))
fft(v1)[1]

