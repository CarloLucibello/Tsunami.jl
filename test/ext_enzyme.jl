
@testitem "linear regression enzyme" setup=[TsunamiTest] begin
    using .TsunamiTest
    N = 1000
    α = 0.5
    λ = 1f-5 / round(Int, N * α)

    M = round(Int, N * α)
    teacher = LinearModel(N)
    X = randn(Float32, N, M)
    y = teacher(X)

    model = LinearModel(N; λ)
    trainer = SilentTrainer(max_epochs=1000, autodiff=:enzyme, accelerator=:cpu)
    @test trainer.foil.device isa CPUDevice
    Tsunami.fit!(model, trainer, [(X, y)])
    @test model.W isa Matrix{Float32}
    @test Flux.mse(model(X), y) < 1e-1
end

@testitem "mlp classification enzyme" setup=[TsunamiTest] begin
    using .TsunamiTest
    using MLUtils: DataLoader
    using Flux: onehotbatch
    din, dout = 784, 10
    n = 1000
    train_loader = DataLoader((randn(Float32, din, n), onehotbatch(rand(1:dout, n), 1:dout)), batchsize=100)

    model = MLP(din, dout, :classification)
    trainer = SilentTrainer(max_epochs=10, autodiff=:enzyme, accelerator=:cpu)
    Tsunami.fit!(model, trainer, train_loader)
end
