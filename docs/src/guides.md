# Guides 

## Selecting a GPU backend

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

Devices are indexed starting from 1, as in the `MLDataDevices.gpu_device` method used by Flux.

## Selecting an automatic differentiation engine

[Zygote](https://fluxml.ai/Zygote.jl/stable/) is the default Automatic Differentiation (AD) engine in Tsunami,
used for computing gradients during training. [Enzyme](https://enzymead.github.io/Enzyme.jl/stable/) is an alternative AD engine that can sometimes provide faster performance and differentiate through mutating functions.

To select an AD engine, use the `autodiff` keyword argument in the `Trainer` constructor:

```julia
trainer = Trainer(autodiff = :enzyme) # options are :zygote (default) and :enzyme
```

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
    return OptimiserChain(AccumGrad(5), AdamW(1e-3))
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
    return OptimiserChain(ClipGrad(0.1), AdamW(1e-3))
end
```

## Freezing model parameters

Tsunami allows you to freeze specific parameters of your model during training, preventing them from being updated. This is useful for transfer learning or fine-tuning pre-trained models. To freeze parameters, use the `freeze` keyword argument in the `Trainer` constructor. The `freeze` argument takes a vector of `KeyPath`s that specify the parameters to freeze.

For instance, assming you have a model (inheriting from `FluxModule`) with layers stored in a field called `layers`, you can freeze all the parameters in the first layer and only the bias of the second layer as follows:

```julia
using Tsunami
trainer = Trainer(freeze = [KeyPath(:layers, 1), KeyPath(:layers, 2, :bias)])
Tsunami.fit!(model, trainer, train_loader, val_loader)
```

Read more about `KeyPath`s in the [Functors.jl documentation](https://fluxml.ai/Functors.jl/stable/api/#KeyPath).

When resuming training from a checkpoint, Tsunami will automatically unfreeze the parameters before applying the freeze settings specified in the `Trainer`.

