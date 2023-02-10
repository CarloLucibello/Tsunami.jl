"""
    FluxModule

An abstract type for Flux models.
A `FluxModule` helps orgainising you code and provides a standard interface for training.

A `FluxModule` comes with `functor` already implemented.
You can change the trainables by implementing `Optimisers.trainables`.

Types inheriting from `FluxModule` have to be mutable. They also
have to implement the following methods in order to interact with a [`Trainer`](@ref):
- `configure_optimisers(model)`
- `training_step(model, batch, batch_idx)`

Optionally also:
- `validation_step(model, batch, batch_idx)`
- `test_step(model, batch, batch_idx)`
- `training_epoch_end(model, outs)`
- `validation_epoch_end(model, outs)`
- `test_epoch_end(model, outs)`

# Examples

```julia
using Flux, Tsunami, Optimisers

# Define a Multilayer Perceptron implementing the FluxModule interface

mutable struct MLP <: FluxModule
    net
end

function MLP()
    net = Chain(Dense(4 => 32, relu), Dense(32 => 2))
    return MLP(net)
end

(model::MLP)(x) = model.net(x)

function Tsunami.training_step(model::MLP, batch, batch_idx)
    x, y = batch
    y_hat = model(x)
    loss = Flux.Losses.mse(y_hat, y)
    return loss
end

function Tsunami.configure_optimisers(model::MLP)
    return Optimisers.setup(Optimisers.Adam(1e-3), model)
end

# Prepare the dataset and the DataLoader

X, Y = rand(4, 100), rand(2, 100)
train_dataloader = Flux.DataLoader((x, y), batchsize=10)


# Create and Train the model

model = MLP()
trainer = Trainer(max_epochs=10)
Tsunami.fit!(model, trainer; train_dataloader)
```
"""
abstract type FluxModule end

function Functors.functor(::Type{<:FluxModule}, m::T) where T
    childr = (; (f => getfield(m, f) for f in fieldnames(T))...)
    re = x -> T(x...)
    return childr, re
end

not_implemented_error(name) = error("You need to implement the method `$(name)`")

"""
    configure_optimisers(model)

Return an optimisers' state,  `Optimisers`

# Examples

```julia
using Optimisers

function configure_optimisers(model::MyFluxModule)
    return Optimisers.setup(AdamW(1e-3), model)
end
```
"""
function configure_optimisers(model::FluxModule)
    not_implemented_error("configure_optimisers")
end

"""
    training_step(model, batch, batch_idx)

Should return either a scalar loss or a `NamedTuple` with a scalar 'loss' field.
"""
function training_step(model::FluxModule, batch, batch_idx)
    not_implemented_error("training_step")
end

"""
    validation_step(model, batch, batch_idx)

If not implemented, the default is to use [`training_step`](@ref).
The return type has to be a `NamedTuple`.
"""
function validation_step(model::FluxModule, batch, batch_idx)
    out = training_step(model, batch, batch_idx)
    if out isa NamedTuple
        return out
    else
        return (; loss = out)
    end
end

"""
    test_step(model, batch, batch_idx)

If not implemented, the default is to use [`validation_step`](@ref).
"""
test_step(model::FluxModule, batch, batch_idx) = validation_step(model::FluxModule, batch, batch_idx)

"""
    training_epoch_end(model, outs)

If not implemented, do nothing. 
"""
function training_epoch_end(::FluxModule, outs::Vector{<:NamedTuple})
    return nothing
end

"""
    validation_epoch_end(model::MyModule, outs)

If not implemented, the default is to compute the mean of the 
scalar outputs of [`validation_step`](@ref).
""" 
function validation_epoch_end(model::FluxModule, outs::Vector{<:NamedTuple})
    ks = keys(outs[1])
    ks = filter(k -> outs[1][k] isa Number, ks)
    mean_out = (; (k => mean(x[k] for x in outs) for k in ks)...)
    return mean_out
end

"""
    test_epoch_end(model::MyModule, outs)

If not implemented, the default is to use [`validation_epoch_end`](@ref).
"""
test_epoch_end(model::FluxModule, outs::Vector{<:NamedTuple}) = validation_epoch_end(model, outs)

"""
    copy!(dest::FluxModule, src::FluxModule)

Shallow copy of all fields of `src` to `dest`.
"""
function Base.copy!(dest::T, src::T) where T <: FluxModule
    for f in fieldnames(T)
        setfield!(dest, f, getfield(src, f))
    end
    return dest
end

function check_fluxmodule(m::FluxModule)
    @assert ismutable(m) "FluxModule has to be a `mutable struct`."
end

function check_training_step(m::FluxModule, batch)
    out = training_step(m, batch, 1)
    errmsg = "The output of `training_step` has to be a scalar or a `NamedTuple` with a `loss` field."
    if out isa NamedTuple
        @assert haskey(out, :loss) errmsg
    else
        @assert out isa Number errmsg
    end
end

function check_validation_step(m::FluxModule, batch)
    out = validation_step(m, batch, 1)
    @assert out isa NamedTuple "The output of `validation_step` has to be a `NamedTuple`."
    @assert haskey(out, :loss) "The output of `validation_step` has to have a `loss` field."
end
