using Flux, Functors, Optimisers, Flurry, MLDatasets
using Flux: DataLoader

mutable struct MLP <: FluxModule
    net
end

@functor MLP

function MLP()
    net = Chain(
        flatten_input,
        Dense(28^2 => 100, relu), 
        Dense(100 => 10))
    return MLP(net)
end

flatten_input(x) = reshape(x, :, size(x, ndims(x)))

function (m::MLP)(x)
    m.net(x)
end

function Flurry.training_step(m::MLP, batch, batch_idx)
    x, y = batch
    y_hat = m(x)
    y = Flux.onehotbatch(y, 0:9)
    loss = Flux.Losses.logitcrossentropy(y_hat, y)
    acc = Flurry.accuracy(y_hat, y)
    return (; loss, acc)
end

function Flurry.configure_optimisers(m::MLP)
    return Optimisers.setup(Optimisers.AdamW(1e-3), m)
end

train_loader = DataLoader(MNIST(:train), batchsize=128, shuffle=true)
test_loader = DataLoader(MNIST(:test), batchsize=128)

# TRAIN FROM SCRATCH
model = MLP()
trainer = Trainer(max_epochs=10, default_root_dir=@__DIR__)
Flurry.fit!(model, trainer; train_dataloader=train_loader, val_dataloader=test_loader)

# RESUME TRAINING
trainer.max_epochs = 20
Flurry.fit!(model, trainer; train_dataloader=train_loader, val_dataloader=test_loader,
    ckpt_path = joinpath(@__DIR__, "ckpt_epoch=0010.bson"))