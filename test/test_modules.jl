mutable struct NotModule
    net
end

############ TestModule1 ############
mutable struct TestModule1 <: FluxModule
    net
    tuple_field::Tuple{Int, Int}
end

TestModule1() = TestModule1(Flux.Chain(Flux.Dense(4, 3, relu), Flux.Dense(3, 2)), (1, 2))

(m::TestModule1)(x) = m.net(x)

function Tsunami.training_step(m::TestModule1, batch, batch_idx)
    x, y = batch
    y_hat = m(x)
    loss = Flux.mse(y_hat, y)
    return loss
end

function Tsunami.configure_optimisers(m::TestModule1)
    return Optimisers.setup(Optimisers.Adam(1e-3), m)
end

# utility returning input and output sizes
io_sizes(m::TestModule1) = 4, 2

############ TestModule2 ############