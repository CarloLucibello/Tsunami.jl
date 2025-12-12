# # Flow Matching Example: Chessboard

# This example is ported the pytorch notebook at 
# https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_flow_matching.ipynb
# See also 
# https://github.com/keishihara/flow-matching

# We train and evaluate a simple 2D FM model with a linear scheduler.

# ## Imports 

using Statistics, Random
using Tsunami, Flux, Optimisers
using Plots, ConcreteStructs
using MLUtils
using CUDA

# ## Data Generation

# We define a generator of 2D points on a chessboard pattern.

function inf_train_gen(batch_size::Int = 200)
    n = 4
    x1 = rand(Float32, batch_size) .* n
    x2 = rand(Float32, batch_size) .+ rand(0:2:n-2, batch_size)
    x2 = x2 .+ floor.(x1) .% 2 
    data = vcat(x1', x2')
    data .-= n / 2    
    return data .* 2 # scale to [-4,4]
end

# Let's visualize a data sample.

function plot_checkboard(points)
    scatter(points[1,:], points[2,:], xlim=(-4,4), ylim=(-4,4),
        markersize=3, legend=false, widen=false, framestyle=:box)
end

data = inf_train_gen(1000)
plot_checkboard(data)

# ## Model Definition
@concrete struct FlowModel <: FluxModule
    net
    hparams
end

function FlowModel(; input_dim::Int = 2, time_dim::Int = 1, hidden_dim::Int = 128, lr = 0.001)
    net = Chain(
        Dense(input_dim + time_dim, hidden_dim, swish),
        Dense(hidden_dim, hidden_dim, swish),
        Dense(hidden_dim, hidden_dim, swish),
        Dense(hidden_dim, input_dim)
    )
    hparams = (; lr)
    return FlowModel(net, hparams)
end

(m::FlowModel)(x::AbstractMatrix, t::Number) = m(x, fill_like(x, t, size(x, 2)))
(m::FlowModel)(x::AbstractMatrix, t::AbstractVector) = m(x, reshape(t, 1, :))
(m::FlowModel)(x::AbstractMatrix, t::AbstractMatrix) = m.net(vcat(x, t))
# (m::FlowModel)(x::AbstractMatrix, t::AbstractMatrix) = m.net(x)

function Tsunami.configure_optimisers(m::FlowModel, trainer)
    return Optimisers.Adam(m.hparams.lr)
end

function Tsunami.train_step(m::FlowModel, trainer, batch, batch_idx)
    x1 = batch
    batch_size = size(x1, 2)
    x0 = randn_like(x1)
    t = rand_like(x1, (1, batch_size))
    xt = @. (1 - t) * x0 + t * x1
    v = x1 .- x0
    v̂ = m(xt, t)
    # loss = Flux.mse(zeros_like(v), x1)
    loss = Flux.mse(v̂, v)
    Tsunami.log(trainer, "loss/train", loss)
    return loss
end

struct ShowCudaAllocCallback end

function Tsunami.on_train_batch_end(callback::ShowCudaAllocCallback, model, trainer, out, batch, batch_idx)
    if batch_idx % 1_000_000 == 0 # set period to a lower value to monitor cuda memory usage
        CUDA.pool_status()
    end
end

function train(; lr = 1e-4, batch_size = 256, iterations = 20_000, hidden_dim = 512)
    train_loader = (inf_train_gen(batch_size) for _ in 1:iterations)
    
    model = FlowModel(; input_dim=2, hidden_dim, lr)

    trainer = Trainer(max_epochs=1, log_every_n_steps=50, 
               accelerator=:auto, autodiff=:zygote, 
               devices=[1], # change this to train on specific GPU
               callbacks=[ShowCudaAllocCallback()])
    
    Tsunami.fit!(model, trainer, train_loader)
    
    return model
end

# Let's train the model.

model = train(lr=1e-3, hidden_dim=512, batch_size=4096, iterations=20_000)

# ## Sampling

# We sample using midpoint-method integration of the learned flow field.

function step(m::FlowModel, x_t::AbstractMatrix, t_start::Number, t_end::Number)
    vt = m(x_t, t_start)
    xhalf = @. x_t + vt * (t_end - t_start) / 2
    vhalf = m(xhalf, (t_start + t_end) / 2)
    return @. x_t + vhalf * (t_end - t_start)
end

function sample(m::FlowModel; n::Int , steps::Int, return_history::Bool = true)
    x = randn(Float32, 2, n)
    ts = Float32.(range(0, 1, length=steps+1))
    history = [x]
    for i in 1:steps
        x = step(m, x, ts[i], ts[i+1])
        push!(history, x)
    end
    if return_history
        return history
    else
        return x
    end
end

history = sample(model, n=1000, steps=20, return_history=true)
anim = @animate for xt in history
    plot_checkboard(xt)
end
gif(anim, "chessboard_sampling_animation.gif", fps=5)

plot_checkboard(history[end])
savefig("chessboard_final_samples.png")
