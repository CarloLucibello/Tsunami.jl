using Flux, Functors, Optimisers, Flurry, MLDatasets
using Flux: DataLoader

struct MLP <: FluxModule
    net
end

@functor MLP

function MLP()
    net = Chain(
        x -> reshape(x, :, size(x, ndims(x))),
        Dense(28^2 => 100, relu), 
        Dense(100 => 10))
    return MLP(net)
end

function (m::MLP)(x)
    m.net(x)
end

function Flurry.training_step(m::MLP, batch, batch_idx)
    x, y = batch
    y = Flux.onehotbatch(y, 0:9)
    y_hat = m(x)
    loss = Flux.Losses.logitcrossentropy(y_hat, y)
    return loss
end

function Flurry.configure_optimisers(m::MLP)
    return Optimisers.setup(Optimisers.AdamW(1e-3), m)
end

train_loader = DataLoader(MNIST(:train), batchsize=128, shuffle=true)
test_loader = DataLoader(MNIST(:test), batchsize=128)

model = MLP()
trainer = Trainer(max_epochs=10, default_root_dir=@__DIR__)
Flurry.fit!(model, trainer; train_dataloader=train_loader, val_dataloader=test_loader)
