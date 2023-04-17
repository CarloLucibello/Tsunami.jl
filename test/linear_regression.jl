

N = 1000
α = 0.5
λ = 1e-5 / round(Int, N * α)

M = round(Int, N * α)
teacher = LinearModel(N)
X = randn(Float32, N, M)
y = teacher(X)

model = LinearModel(N; λ)
trainer = SilentTrainer(max_epochs=1000, devices=[1])

Tsunami.fit!(model, trainer, [(X, y)])
