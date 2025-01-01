<img align="right" width="200px" src="https://raw.githubusercontent.com/CarloLucibello/Tsunami.jl/main/docs/src/assets/the_great_wave.jpg">

# Tsunami.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://CarloLucibello.github.io/Tsunami.jl/dev)
![](https://github.com/CarloLucibello/Tsunami.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/CarloLucibello/Tsunami.jl/branch/main/graph/badge.svg?token=UhgCzsHqhM)](https://codecov.io/gh/CarloLucibello/Tsunami.jl)

A high-level deep learning framework for the Julia language that helps you focus and organize the relevant part of your code while removing the boilerplate. 

Tsunami  is built on top of [Flux.jl](https://github.com/FluxML/Flux.jl) and it is heavily inspired by [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/) (although [LightningAI](https://www.pytorchlightning.ai/index.html) is not involved in this project).


## Installation 

Install Tsunami with 
```julia
pkg> add Tsunami
```

## Usage

Define your model subtyping the `FluxModule` abstract type, implement a few required methods, then let the `Trainer` train the model on your dataset with `Tsunami.fit!`. Tsunami will handle all of the boilerplate (training loop, loggin, gpu movement, validation, ...).

In the following script we train a Multilayer Perceptron on the FashionMNIST dataset using Tsunami:
```julia
using Flux, Optimisers, Statistics, Tsunami, MLDatasets
using CUDA # or AMDGPU, Metal, ... for GPU support
using MLUtils: DataLoader, flatten, mapobs

## Define the model 

struct MLP <: FluxModule
    net
end

MLP() = MLP(Chain(flatten,
                Dense(28^2 => 512, relu), 
                Dense(512 => 10)))

(model::MLP)(x) = model.net(x)

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
trainer = Trainer(max_epochs=5)
fit_state = Tsunami.fit!(model, trainer, train_loader, test_loader)
```

What follows is the final output of the script. The script will train the model on CUDA gpus if available and will also write tensorboard logs and and model checkpoints on disk.

<img src="https://raw.githubusercontent.com/CarloLucibello/Tsunami.jl/main/docs/src/assets/readme_output.png">

See the [documentation](https://carlolucibello.github.io/Tsunami.jl/dev/) and check the [examples](https://github.com/CarloLucibello/Tsunami.jl/tree/main/examples) folder to learn more.

## Features

- Use `Tsunami.fit!` instead of implementing a training loop.
- Logging (tensorboard).
- Checkpoints (save and resume training).
- Hyperparameters' schedulers.
- CUDA, AMDGPU, Metal GPU support.

## Contributions are welcome!

If you want to contribute to Tsunami, please open an issue or a pull request.
Any help is appreciated!

## Similar julia libraries 

- [FastAI.jl](https://github.com/FluxML/FastAI.jl)
- [FluxTraining.jl](https://github.com/FluxML/FluxTraining.jl)
- [Ignite.jl](https://github.com/jondeuce/Ignite.jl)
