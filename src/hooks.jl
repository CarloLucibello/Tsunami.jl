hook(f, model::EnzymeCore.Duplicated, args...) = hook(f, model.val, args...)

function hook(f, model::FluxModule, trainer::Trainer, args...)
    f(model, trainer, args...)
    for callback in trainer.callbacks
        f(callback, model, trainer, args...)
    end
end

"""
    on_before_update([callback,] model, trainer, grad)

Called before the call to `Optimisers.update!` that 
applies the gradient `grad` to update the `model`'s parameters.
"""
on_before_update(model, trainer, grad) = nothing
on_before_update(cb, model, trainer, grad) = nothing

"""
    on_before_backprop([callback,] model, trainer, loss)

Called after the model's forward, where also the pullback is created, 
but before the call to the pullback (the backward pass computing the gradient).
"""
on_before_backprop(model, trainer, loss) = nothing
on_before_backprop(cb, model, trainer, loss) = nothing

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
    on_train_batch_end([callback,] model, trainer)

Called at the end of each training batch.
"""
on_train_batch_end(model, trainer) = nothing
on_train_batch_end(cb, model, trainer) = nothing

"""
    on_val_batch_end([callback,] model, trainer)

Called at the end of each validation batch.
"""
on_val_batch_end(model, trainer) = nothing
on_val_batch_end(cb, model, trainer) = nothing

"""
    on_test_batch_end([callback,] model, trainer)

Called at the end of each test batch.
"""
on_test_batch_end(model, trainer) = nothing
on_test_batch_end(cb, model, trainer) = nothing

