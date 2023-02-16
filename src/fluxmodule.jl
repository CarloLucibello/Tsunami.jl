"""
    FluxModule

An abstract type for Flux models.
A `FluxModule` helps orgainising you code and provides a standard interface for training.

A `FluxModule` comes with `functor` already implemented.
You can change the trainables by implementing `Optimisers.trainables`.

Types inheriting from `FluxModule` have to be mutable. They also
have to implement the following methods in order to interact with a [`Trainer`](@ref).

# Required methods

- [`configure_optimisers`](@ref)`(model)`
- [`training_step`](@ref)`(model, batch, batch_idx)`

# Optional Methods 

- [`validation_step`](@ref)`(model, batch, batch_idx)`
- [`test_step`](@ref)`(model, batch, batch_idx)`
- [`training_epoch_end`](@ref)`(model, outs)`
- [`validation_epoch_end`](@ref)`(model, outs)`
- [`test_epoch_end`](@ref)`(model, outs)`

# Examples

```julia
using Flux, Tsunami, Optimisers

# Define a Multilayer Perceptron implementing the FluxModule interface

mutable struct Model <: FluxModule
    net
end

function Model()
    net = Chain(Dense(4 => 32, relu), Dense(32 => 2))
    return Model(net)
end

(model::Model)(x) = model.net(x)

function Tsunami.training_step(model::Model, batch, batch_idx)
    x, y = batch
    y_hat = model(x)
    loss = Flux.Losses.mse(y_hat, y)
    return loss
end

function Tsunami.configure_optimisers(model::Model)
    return Optimisers.setup(Optimisers.Adam(1e-3), model)
end

# Prepare the dataset and the DataLoader
X, Y = rand(4, 100), rand(2, 100)
train_dataloader = Flux.DataLoader((x, y), batchsize=10)

# Create and Train the model
model = Model()
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

Return an optimiser's state initialized for the `model`.
It can also return a tuple of `(scheduler, optimiser)`,
where `scheduler` is any callable object that takes 
the current epoch as input and returns a scalar that will be 
set as the learning rate for the next epoch.

# Examples

```julia
using Optimisers, ParameterScheduler

function Tsunami.configure_optimisers(model::Model)
    return Optimisers.setup(AdamW(1e-3), model)
end

# Now with a scheduler dropping the learning rate by a factor 10 
# at epochs [50, 100, 200] starting from the initial value of 1e-2
function Tsunami.configure_optimisers(model::Model)

    function lr_scheduler(epoch)
        if epoch <= 50
            return 1e-2
        elseif epoch <= 100
            return 1e-3
        elseif epoch <= 200
            return 1e-4
        else
            return 1e-5
        end
    end
    
    opt = Optimisers.setup(AdamW(), model)
    return lr_scheduler, opt
end

# Same as above but using the ParameterScheduler package.
function Tsunami.configure_optimisers(model::Model)
    lr_scheduler = ParameterScheduler.Step(1e-2, 1/10, [50, 50, 100])
    opt = Optimisers.setup(AdamW(), model)
    return lr_scheduler, opt
end
```
"""
function configure_optimisers(model::FluxModule)
    not_implemented_error("configure_optimisers")
end

"""
    training_step(model, batch, batch_idx)

The method called at each training step during `Tsunami.fit!`.
It should compute the forward pass of the model and return the loss 
corresponding to the minibatch `batch`. 

It is also possible to return multiple values by returning a `NamedTuple`.
The values will be logged in the `Trainer`'s `progress_bar`, by the logger, and
will be available in the `outs` argument of the `*_epoch_end` methods.

The training loop in `Tsunami.fit!` approximately looks like this:
```julia
for epoch in 1:epochs
    for (batch_idx, batch) in enumerate(train_dataloader)
        grads = gradient(model) do m
            out = training_step(m, batch, batch_idx)
            # ...
            return out.loss
        end
        Optimisers.update!(opt, model, grads[1])
    end
end
```



# Examples

```julia
function Tsunami.training_step(model::Model, batch, batch_idx)
    x, y = batch
    ŷ = model(x)
    loss = Flux.Losses.logitcrossentropy(ŷ, y)
    accuracy = Tsunami.accuracy(ŷ, y)
    return (; loss, accuracy)
end
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
    validation_epoch_end(model, outs)

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
    test_epoch_end(model, outs)

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
    losserrmsg = "The output of `training_step` has to be a scalar or a `NamedTuple` with a `loss` field."
    if out isa NamedTuple
        @assert haskey(out, :loss) losserrmsg
    else
        @assert out isa Number losserrmsg
    end
end

function check_validation_step(m::FluxModule, batch)
    out = validation_step(m, batch, 1)
    @assert out isa NamedTuple "The output of `validation_step` has to be a `NamedTuple`."
end

function Base.show(io::IO, ::MIME"text/plain", m::T) where T <: FluxModule
    if get(io, :compact, false)
        return print(io, T)
    end
    print(io, "$T")
    for f in sort(fieldnames(T) |> collect)
        startswith(string(f), "_") && continue
        v = getfield(m, f)
        if v isa Chain
            s = "  $f = "
            print(io, "\n$s")
            tsunami_big_show(io, v, length(s))
        else
            print(io, "\n  $f = ")
            show(IOContext(io, :compact=>true), v)
        end
    end
end
