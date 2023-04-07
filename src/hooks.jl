
"""
    on_before_update([callback,] model, trainer, grad)

Called in `Tsunami.fit!` before the call to `Optimisers.update!` that 
applies the gradient `grad` to update the `model`'s parameters.
"""
on_before_update(model, trainer, grad) = nothing
on_before_update(cb, model, trainer, grad) = nothing

"""
    on_train_epoch_end([callback,] model, trainer)

Called in `Tsunami.fit!` at the end of each training epoch.
""" 
on_train_epoch_end(model, trainer) = nothing

"""
    on_val_epoch_end([callback,] model, trainer)

Called in `Tsunami.fit!` at the end of each validation epoch.
"""
on_val_epoch_end(model, trainer) = nothing

"""
    on_test_epoch_end([callback,] model, trainer)

Called in `Tsunami.fit!` at the end of each test epoch.
"""
on_test_epoch_end(model, trainer) = nothing