# Flurry.jl

A high-level deep learning framework for the Julia language 
that helps you focus and organize the relevant part of your code
while removing the boilerplate. 

Flurry  is built on top of [Flux.jl](https://github.com/FluxML/Flux.jl)
and it is heavily inspired by [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/).


## Installation 

Flurry is still in an early development change and it is not a registered package yet. 
Things can break without any notice. 

Install Flurry with 
```julia
pkg> add https://github.com/CarloLucibello/Flurry.jl
```

## Features

- Use `fit!` instead of implementing a training loop.
- Logging (tensorboard).
- Checkpoints (save and resume training).
- GPU movement.

## Usage Examples

Define your model subtyping the `FluxModule` abstract type, implement a few required methods, then let the `Trainer`
train the model on your dataset with `fit!`. Flurry will handle all of the boilerplate (training loop, loggin, gpu movement, validation, ...).

See the folder [examples/](https://github.com/CarloLucibello/Flurry.jl/tree/main/examples) for usage examples.

## Similar libraries 

- [FastAI.jl](https://github.com/FluxML/FastAI.jl)
- [FluxTraining.jl](https://github.com/FluxML/FluxTraining.jl)
