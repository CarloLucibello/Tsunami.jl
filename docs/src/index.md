# Tsunami.jl

![the_great_wave](https://raw.githubusercontent.com/CarloLucibello/Tsunami.jl/main/docs/src/assets/the_great_wave.jpg)

A high-level deep learning framework for the Julia language 
that helps you focus and organize the relevant part of your code
while removing the boilerplate. 

Tsunami  is built on top of [Flux.jl](https://github.com/FluxML/Flux.jl)
and it is heavily inspired by [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

## Installation 

Tsunami is still in an early development change and it is not a registered package yet. 
Things can break without any notice. 

Install Tsunami with 
```julia
pkg> add https://github.com/CarloLucibello/Tsunami.jl
```

## Features

- Use `fit!` instead of implementing a training loop.
- Logging (tensorboard).
- Checkpoints (save and resume training).
- GPU movement.

## Usage Examples

Define your model subtyping the `FluxModule` abstract type, implement a few required methods, then let the `Trainer`
train the model on your dataset with `fit!`. Tsunami will handle all of the boilerplate (training loop, logging, gpu movement, validation, ...).

```julia
using Flux, Optimisers, Statistics, Tsunami, HuggingFaceDatasets, ImageCore
using Flux: DataLoader, flatten, mapobs

## Define the model 

mutable struct MLP <: FluxModule
    net
end

MLP() = MLP(Chain(flatten,
                Dense(28^2 => 512, relu), 
                Dense(512 => 10)))

(model::MLP)(x) = model.net(x)

function Tsunami.train_step(model::MLP, trainer, batch, batch_idx)
    x, y = batch
    ŷ = model(x)
    loss = Flux.Losses.logitcrossentropy(ŷ, y)
    Tsunami.log(trainer, "loss/train", loss, prog_bar=true)
    Tsunami.log(trainer, "accuracy/train", Tsunami.accuracy(ŷ, y), prog_bar=true)
    return loss
end


function Tsunami.val_step(model::MLP, trainer, batch, batch_idx)
    x, y = batch
    ŷ = model(x)
    loss = Flux.Losses.logitcrossentropy(ŷ, y)
    Tsunami.log(trainer, "loss/val", loss)
    Tsunami.log(trainer, "accuracy/val", Tsunami.accuracy(ŷ, y))
    return loss
end

Tsunami.configure_optimisers(model::MLP, trainer) = 
    Optimisers.setup(Optimisers.AdamW(1e-3), model)

## Prepare the data

function mnist_transform(batch)
    image = ImageCore.channelview.(batch["image"])
    image = Flux.batch(image) ./ 255f0
    label = Flux.onehotbatch(batch["label"], 0:9)
    return (; image, label)
end

train_data = load_dataset("fashion_mnist", split="train").with_format("julia")
train_data = mapobs(mnist_transform, train_data)[:]
train_loader = DataLoader(train_data, batchsize=128, shuffle=true)

test_data = load_dataset("fashion_mnist", split="test").with_format("julia")
test_data = mapobs(mnist_transform, test_data)[:]
test_loader = DataLoader(test_data, batchsize=128)

## Create and train the model

model = MLP()
trainer = Trainer(max_epochs=5)
Tsunami.fit!(model, trainer, train_loader, test_loader);
```

<img src="https://raw.githubusercontent.com/CarloLucibello/Tsunami.jl/main/docs/src/assets/readme_output.png">

See the folder [examples/](https://github.com/CarloLucibello/Tsunami.jl/tree/main/examples) for usage examples.

## Similar libraries 

- [FastAI.jl](https://github.com/FluxML/FastAI.jl)
- [FluxTraining.jl](https://github.com/FluxML/FluxTraining.jl)
