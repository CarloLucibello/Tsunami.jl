using Flux, Optimisers, Statistics, Tsunami, MLDatasets
using MLUtils: DataLoader, flatten, mapobs

## Uncomment one of the following for GPU acceleration

# using CUDA
# using AMDGPU
# using Metal

## Define the model 

struct MLP{T} <: FluxModule
    net::T
end

MLP() = MLP(Chain(Dense(28^2 => 512, relu), Dense(512 => 10)))

(model::MLP)(x) = model.net(flatten(x))

function loss_and_accuracy(model::MLP, batch)
    x, y = batch
    ŷ = model(x)
    return Flux.logitcrossentropy(ŷ, y), Tsunami.accuracy(ŷ, y)
end

function Tsunami.train_step(model::MLP, trainer, batch)
    loss, acc = loss_and_accuracy(model, batch)
    Tsunami.log(trainer, "loss/train", loss, prog_bar=true)
    Tsunami.log(trainer, "accuracy/train", acc, prog_bar=true)
    return loss
end

function Tsunami.val_step(model::MLP, trainer, batch)
    loss, acc = loss_and_accuracy(model, batch)
    Tsunami.log(trainer, "loss/val", loss)
    Tsunami.log(trainer, "accuracy/val", acc)
end

Tsunami.configure_optimisers(model::MLP, trainer) = 
    Optimisers.setup(Optimisers.AdamW(1e-3), model)

## Prepare the data

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

## Create and train the model

model = MLP()
trainer = Trainer(max_epochs=5, accelerator=:cpu)
Tsunami.fit!(model, trainer, train_loader, test_loader)
