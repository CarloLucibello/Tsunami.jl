# # Flow Matching Example: Chessboard

# This example is ported the pytorch notebook at 
# https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_flow_matching.ipynb

# We train and evaluate a simple 2D FM model with a linear scheduler.

# ## Imports 

using Statistics, Random
using Tsunami, Flux, Optimisers
using Plots, ConcreteStructs
using MLUtils
using Metal

# ## Data Generation

# We define a generator of 2D points on a chessboard pattern.

function inf_train_gen(batch_size::Int = 200)
    n = 4
    x1 = rand(Float32, batch_size) .* n
    x2 = rand(Float32, batch_size) .+ rand(0:2:n-2, batch_size)
    x2 = x2 .+ floor.(x1) .% 2 
    data = vcat(x1', x2')
    data .-= n / 2    
    return data
end

# Let's visualize a data sample.

data = inf_train_gen(1000)

scatter(data[1,:], data[2,:], xlim=(-2,2), ylim=(-2,2),
    markersize=3, legend=false, widen=false, framestyle=:box)

# We also define a generator of points for the moon dataset.

function make_moons(n_samples::Int = 100, noise = 0.1, shuffle::Bool = true)
    n_samples_out = n_samples ÷ 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = cos.(range(0, π, length=n_samples_out))
    outer_circ_y = sin.(range(0, π, length=n_samples_out))
    inner_circ_x = 1 .- cos.(range(0, π, length=n_samples_in))
    inner_circ_y = 1 .- sin.(range(0, π, length=n_samples_in)) .- 0.5

    X = vcat(hcat(outer_circ_x', inner_circ_x'), hcat(outer_circ_y', inner_circ_y'))
    y = vcat(zeros(Int, n_samples_out), ones(Int, n_samples_in))

    if shuffle
        perm = randperm(n_samples)
        X = X[:, perm]
        y = y[perm]
    end

    if noise !== nothing
        X += noise * randn(size(X))
    end

    return X, y
end

data = make_moons(1000, 0.1)

scatter(data[1][1,:], data[1][2,:], xlim=(-2,3), ylim=(-2,3),
    markersize=3, legend=false, widen=false, framestyle=:box)

# ## Model

@concrete struct FlowModel <: FluxModule
    net
    hparams
end

function FlowModel(; input_dim::Int = 2, time_dim::Int = 1, hidden_dim::Int = 128, lr = 0.001)
    net = Chain(
        Dense(input_dim + time_dim, hidden_dim, elu),
        Dense(hidden_dim, hidden_dim, elu),
        Dense(hidden_dim, hidden_dim, elu),
        Dense(hidden_dim, input_dim)
    )
    hparams = (; lr)
    return FlowModel(net, hparams)
end

(m::FlowModel)(x::AbstractMatrix, t::Number) = m(x, fill_like(x, t, size(x, 2)))
(m::FlowModel)(x::AbstractMatrix, t::AbstractVector) = m(x, reshape(t, 1, :))
(m::FlowModel)(x::AbstractMatrix, t::AbstractMatrix) = m.net(vcat(x, t))

function Tsunami.configure_optimisers(m::FlowModel, trainer)
    opt = Optimisers.setup(Optimisers.Adam(m.hparams.lr), m)
    return opt
end

function Tsunami.train_step(m::FlowModel, trainer, batch, batch_idx)
    x1 = batch
    batch_size = size(x1, 2)
    x0 = randn_like(x1)
    t = rand_like(x1, (1, batch_size))
    xt = @. (1 - t) * x0 + t * x1
    v = x1 .- x0
    v̂ = m(xt, t)
    loss = Flux.mse(v̂, v)
    Tsunami.log(trainer, "loss/train", loss)
    return loss
end

function train(; lr = 1e-4, batch_size = 256, iterations = 50000, hidden_dim = 512)
    # train_loader = (inf_train_gen(batch_size) for _ in 1:iterations)
    train_loader = (make_moons(batch_size, 0.05)[1] for _ in 1:iterations)
    
    model = FlowModel(; input_dim=2, hidden_dim, lr)
    trainer = Trainer(max_epochs=1, log_every_n_steps=50, accelerator=:gpu)
    Tsunami.fit!(model, trainer, train_loader)
    return model
end

# Let's train the model.

m = train(lr=1e-2, hidden_dim=64)

# ## Sampling

function step(m::FlowModel, x_t::AbstractMatrix, t_start::Number, t_end::Number)
    vt = m(x_t, t_start)
    xhalf = @. x_t + vt * (t_end - t_start) / 2
    vhalf = m(xhalf, (t_start + t_end) / 2)
    return @. x_t + vhalf * (t_end - t_start)
end

function sample(m::FlowModel, x0::AbstractMatrix, n_steps::Int)
    ts = Float32.(range(0, 1, length=n_steps+1))
    x = x0
    for i in 1:n_steps
        x = step(m, x, ts[i], ts[i+1])
    end
    return x
end


samples = sample(m, randn(Float32, 2, 1000), 10)

scatter(samples[1,:], samples[2,:], xlim=(-2,3), ylim=(-2,3),
    markersize=3, legend=false, widen=false, framestyle=:box)
