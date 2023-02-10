<img align="right" width="300px" src="https://raw.githubusercontent.com/CarloLucibello/Tsunami.jl/master/docs/src/assets/the_great_wave.jpg">

# Tsunami.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://CarloLucibello.github.io/Tsunami.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://CarloLucibello.github.io/Tsunami.jl/dev)
![](https://github.com/CarloLucibello/Tsunami.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/CarloLucibello/Tsunami.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/CarloLucibello/Tsunami.jl)


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
train the model on your dataset with `fit!`. Tsunami will handle all of the boilerplate (training loop, loggin, gpu movement, validation, ...).

See the folder [examples/](https://github.com/CarloLucibello/Tsunami.jl/tree/main/examples) for usage examples.

## Similar libraries 

- [FastAI.jl](https://github.com/FluxML/FastAI.jl)
- [FluxTraining.jl](https://github.com/FluxML/FluxTraining.jl)
