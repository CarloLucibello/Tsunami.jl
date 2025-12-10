hook(f, model::EnzymeCore.Duplicated, args...) = hook(f, model.val, args...)

function hook(f, model::FluxModule, trainer::Trainer, args...)
    GPUArrays.@cached trainer.cache begin
        f(model, trainer, args...)
        for callback in trainer.callbacks
            f(callback, model, trainer, args...)
        end
    end
end

"""
    on_before_update([callback,] model, trainer, out, grad)

Called before the call to `Optimisers.update!` that 
applies the gradient `grad` to update the `model`'s parameters.
`out` is the output of the last call to [`train_step`](@ref).
"""
on_before_update(model, trainer, out, grad) = nothing
on_before_update(cb, model, trainer, out, grad) = nothing

"""
    on_train_epoch_start([callback,] model, trainer)

Called at the beginning of each training epoch.
"""
on_train_epoch_start(model, trainer) = nothing
on_train_epoch_start(cb, model, trainer) = nothing

"""
    on_val_epoch_start([callback,] model, trainer)

Called  at the beginning of each validation epoch.
""" 
on_val_epoch_start(model, trainer) = nothing
on_val_epoch_start(cb, model, trainer) = nothing

"""
    on_test_epoch_start([callback,] model, trainer)

Called at the beginning of each test epoch.
"""
on_test_epoch_start(model, trainer) = nothing
on_test_epoch_start(cb, model, trainer) = nothing

"""
    on_train_epoch_end([callback,] model, trainer)

Called at the end of each training epoch.

To access all batch outputs at the end of the epoch, 
you can cache step outputs as an attribute of the model and access them in this hook:

See also [`on_train_epoch_start`](@ref).

# Examples 

```julia
struct Callback
    training_step_outputs::Vector{Float32}
    # other fields...
end

function Tsunami.train_step(model::MyModel, trainer, batch)
    ...
    return (loss = loss, accuracy = accuracy)
end

function Tsunami.on_train_epoch_start(cb::Callback, model, trainer)
    empty!(cb.training_step_outputs)
end

function Tsunami.on_train_batch_end(cb::Callback, model, trainer, out, batch, batch_idx)
    push!(cb.training_step_outputs, out.accuracy)
end

function Tsunami.on_train_epoch_end(cb::Callback, model, trainer)
    println("Mean accuracy: ", mean(cb.training_step_outputs))
end
```
""" 
on_train_epoch_end(model, trainer) = nothing
on_train_epoch_end(cb, model, trainer) = nothing

"""
    on_val_epoch_end([callback,] model, trainer)

Called at the end of each validation epoch.
"""
on_val_epoch_end(model, trainer) = nothing
on_val_epoch_end(cb, model, trainer) = nothing

"""
    on_test_epoch_end([callback,] model, trainer)

Called at the end of each test epoch.
"""
on_test_epoch_end(model, trainer) = nothing
on_test_epoch_end(cb, model, trainer) = nothing

"""
    on_train_batch_start([callback,] model, trainer, batch, batch_idx)

Called at the beginning of each training batch.
"""
on_train_batch_start(model, trainer, batch, batch_idx) = nothing
on_train_batch_start(cb, model, trainer, batch, batch_idx) = nothing

"""
    on_val_batch_start([callback,] model, trainer, batch, batch_idx)

Called at the beginning of each validation batch.
"""
on_val_batch_start(model, trainer, batch, batch_idx) = nothing
on_val_batch_start(cb, model, trainer, batch, batch_idx) = nothing

"""
    on_test_batch_start([callback,] model, trainer, batch, batch_idx)

Called at the beginning of each test batch.
"""
on_test_batch_start(model, trainer, batch, batch_idx) = nothing
on_test_batch_start(cb, model, trainer, batch, batch_idx) = nothing

"""
    on_train_batch_end([callback,] model, trainer, out, batch, batch_idx)

Called at the end of each iteration in the training loop.
`out` is the output of [`train_step`](@ref).
"""
on_train_batch_end(model, trainer, out, batch, batch_idx) = nothing
on_train_batch_end(cb, model, trainer, out, batch, batch_idx) = nothing

"""
    on_val_batch_end([callback,] model, trainer, out, batch, batch_idx)

Called at the end of each iteration in the validation loop.
`out` is the output of [`val_step`](@ref).
"""
on_val_batch_end(model, trainer, out, batch, batch_idx) = nothing
on_val_batch_end(cb, model, trainer, out, batch, batch_idx) = nothing

"""
    on_test_batch_end([callback,] model, trainer, out, batch, batch_idx)

Called at the end of each iteration in the test loop.
`out` is the output of [`test_step`](@ref).
"""
on_test_batch_end(model, trainer, out, batch, batch_idx) = nothing
on_test_batch_end(cb, model, trainer, out, batch, batch_idx) = nothing

