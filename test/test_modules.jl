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

############ LinearModel ############


mutable struct LinearModel{Tw, Tm, F} <: FluxModule
    W::Tw
    mask::Tm
    λ::F  # L2 regularization
end

function LinearModel(N::Int; λ = 0.0f0)
    W = randn(Float32, 1, N) ./ sqrt(N) 
    mask = fill(true, size(W))
    return LinearModel(W, mask, λ)
end

function (m::LinearModel)(x::AbstractMatrix)
    return (m.W .* m.mask) * x
end

function Tsunami.training_step(model::LinearModel, batch, batch_idx)
    x, y = batch
    ŷ = model(x)
    loss_data = Flux.mse(ŷ, y)
    loss_reg = model.λ * sum(abs2.(model.W)) 
    # Zygote.ignore_derivatives() do
    #     @show loss_data loss_reg
    # end
    return loss_data + loss_reg
end

function Tsunami.configure_optimisers(model::LinearModel)
    return Optimisers.setup(Optimisers.Adam(1e-1), model)
end

