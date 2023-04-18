

N = 1000
α = 0.5
λ = 1e-5 / round(Int, N * α)

M = round(Int, N * α)
teacher = LinearModel(N)
X = randn(Float32, N, M)
y = teacher(X)

model = LinearModel(N; λ)
@test model.W isa Matrix{Float64}
@test size(model.W) == (1, N)

trainer = SilentTrainer(max_epochs=1000, devices=[1])
model, fit_state = Tsunami.fit(model, trainer, [(X, y)])
@test model.W isa Matrix{Float32} # by default precision is Float32
