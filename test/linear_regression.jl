
@testitem "linear regression" setup=[TsunamiTest] begin
    using .TsunamiTest
    N = 1000
    α = 0.5
    λ = 1f-4 / round(Int, N * α)

    M = round(Int, N * α)
    teacher = LinearModel(N)
    X = randn(Float32, N, M)
    y = teacher(X)

    model = LinearModel(N; λ)
    @test model.W isa Matrix{Float32}
    @test size(model.W) == (1, N)
    trainer = SilentTrainer(max_epochs=100)
    Tsunami.fit!(model, trainer, [(X, y)])
    @test model.W isa Matrix{Float32} # by default precision is Float32
    @test Flux.mse(model(X), y) < 1e-1
end

@testitem "GPU linear regression" setup=[TsunamiTest] tags=[:gpu] begin
    using .TsunamiTest
    N = 1000
    α = 0.5
    λ = 1f-4 / round(Int, N * α)

    M = round(Int, N * α)
    teacher = LinearModel(N)
    X = randn(Float32, N, M)
    y = teacher(X)

    model = LinearModel(N; λ)
    @test model.W isa Matrix{Float32}
    @test size(model.W) == (1, N)
    trainer = SilentTrainer(max_epochs=100, accelerator=:gpu)
    Tsunami.fit!(model, trainer, [(X, y)])
    @test model.W isa Matrix{Float32} # by default precision is Float32
    @test Flux.mse(model(X), y) < 1e-1
end
