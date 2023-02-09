"""
    abstract type FluxModule end

Type inheriting from `FluxModule` have to be mutable.
A `FluxModule` comes with `functor` already implemented.
You can change the trainables by implementing `Optimisers.trainables`.

`FluxModule`'s subtypes need to implement the following methods:
- `configure_optimisers(model)`
- `training_step(model, batch, batch_idx)`

Optionally also:
- `validation_step(model, batch, batch_idx)`
- `test_step(model, batch, batch_idx)`
- `training_epoch_end(model, outs)`
- `validation_epoch_end(model, outs)`
- `test_epoch_end(model, outs)`

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

"""
function training_epoch_end(::FluxModule, outs::Vector{<:NamedTuple})
    names = keys(first(outs)) 
    y =  (; (name => mean(x->x[name], outs) for name in names)...)
    return y
end

"""
    validation_epoch_end(model::MyModule, outs)

If not implemented, the default is to use [`training_epoch_end`](@ref).
""" 
validation_epoch_end(model::FluxModule, outs::Vector{<:NamedTuple}) = training_epoch_end(model, outs)

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
    @assert ismutable(m) "FluxModule has to be a `mutable struct`"
end
