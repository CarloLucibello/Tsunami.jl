# Guides 

## Selecting a backend

Tsunami supports both CPU and GPU devices. To select a device, use the `accelerator` and `devices` keyword arguments in the `Trainer` constructor.

```julia
trainer = Trainer(accelerator = :auto) # default, selects CPU or GPU depending on availability
trainer = Trainer(accelerator = :gpu) # forces selection to GPU, errors if no GPU is available
```
Currently supported accelerators are `:auto`, `:gpu`, and `:cpu`.
See the [`Trainer`](@ref) documentation for more details.

By default, Tsunami will use the first of the available GPUs and the CPU if no GPUs are present. 
To select a specific GPU, use the `devices` keyword argument:

```julia
trainer = Trainer(devices = [1])
```

Devices are indexed starting from 1, as in the `MLDataDevices.get_device` method used by Flux.

## Selecting trainable components

TODO

## Freezing parameters

TODO

## Gradient accumulation

Gradient accumulation is a technique that allows you to simulate larger batch sizes by accumulating gradients over multiple batches. This is useful when you want to use a large batch size but your GPU does not have enough memory.

Optimisers.jl supports gradient accumulation the `AccumGrad` rule:

```
    AccumGrad(n::Int)

A rule constructed `OptimiserChain(AccumGrad(n), Rule())` will accumulate for `n` steps, before applying Rule to the mean of these `n` gradients.

This is useful for training with effective batch sizes too large for the available memory. Instead of computing the gradient for batch size `b` at once, compute it for size `b/n` and accumulate `n` such gradients.
```

AccumGrad can be easily integrated into Tsunami's `configure_optimisers`:

```julia
using Optimisers

function Tsunami.configure_optimisers(model::Model, trainer)
    opt = OptimiserChain(AccumGrad(5), AdamW(1e-3))
    opt_state = Optimiser.setup(opt, model)
    return opt_state
end
```

## Gradient clipping

Gradient clipping is a technique that allows you to limit the range or the norm of the gradients. This is useful to prevent exploding gradients and improve training stability.

Optimisers.jl supports gradient clipping with the `ClipNorm` and `ClipGrad` rule:

```
    ClipGrad(δ = 10f0)

Restricts every gradient component to obey -δ ≤ dx[i] ≤ δ.
```
```
    ClipNorm(ω = 10f0, p = 2; throw = true)

Scales any gradient array for which norm(dx, p) > ω to stay at this threshold (unless p==0).
Throws an error if the norm is infinite or NaN, which you can turn off with throw = false.
```

Gradient clipping can be easily integrated into Tsunami's `configure_optimisers`:

```julia
using Optimisers

function Tsunami.configure_optimisers(model::Model, trainer)
    opt = OptimiserChain(ClipGrad(0.1), AdamW(1e-3))
    opt_state = Optimiser.setup(opt, model)
    return opt_state
end
```

## Mixed precision training

TODO
