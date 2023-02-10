mutable struct NotModule
    net
end

mutable struct TestModule1 <: FluxModule
    net
end

TestModule1() = TestModule1(Flux.Chain(Flux.Dense(4, 3, relu), Flux.Dense(3, 2)))
(m::TestModule1)(x) = m.net(x)

function training_step(m::TestModule1, batch, batch_idx)
    x, y = batch
    y_hat = m(x)
    loss = Flux.mse(y_hat, y)
    return loss
end

@testset "abstract type FluxModule" begin
    @test isabstracttype(FluxModule)
end

@testset "functor" begin
    m = TestModule1()
end


