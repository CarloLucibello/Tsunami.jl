
@testset "linear regression enzyme" tags=[:enzyme]begin
    using .TsunamiTest
    N = 1000
    α = 0.5
    λ = 1f-4 / round(Int, N * α)

    M = round(Int, N * α)
    teacher = LinearModel(N)
    X = randn(Float32, N, M)
    y = teacher(X)

    model = LinearModel(N; λ)
    trainer = SilentTrainer(max_epochs=100, autodiff=:enzyme, accelerator=:cpu)
    @test trainer.foil.device isa CPUDevice
    Tsunami.fit!(model, trainer, [(X, y)])
    @test model.W isa Matrix{Float32}
    @test Flux.mse(model(X), y) < 1e-1
end

@testset "mlp classification enzyme" tags=[:enzyme]begin
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

@testset "mlp fashion mnist enzyme" tags=[:enzyme] begin
    using .TsunamiTest
    using MLUtils: DataLoader, mapobs, getobs
    using MLDatasets

    function mnist_transform(batch)
        x, y = batch
        y = Flux.onehotbatch(y, 0:9)
        return (x, y)
    end

    train_data = FashionMNIST(split=:train)
    train_data = mapobs(mnist_transform, train_data)[:]
    train_loader = DataLoader(train_data, batchsize=128, shuffle=true)

    test_data = FashionMNIST(split=:test)
    test_data = mapobs(mnist_transform, test_data)[:]
    test_loader = DataLoader(test_data, batchsize=128)

    model = MLP(28*28, 10, :classification)
    trainer = SilentTrainer(max_epochs=5, autodiff=:enzyme, accelerator=:cpu)
    Tsunami.fit!(model, trainer, train_loader)
    n = 100
    x, y = getobs(train_data, 1:n)
    @test Tsunami.accuracy(model(x), y) > 0.85
end
