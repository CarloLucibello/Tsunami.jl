"""
    Tsunami.log(trainer::Trainer, name::AbstractString, value; 
        [on_step, on_epoch, prog_bar, batchsize])

Log a `value` with name `name`. Can be called from any function in the training loop
or from a callback. Logs to the loggers specified in `trainer.loggers`.

See the [Logging](@ref) docs for more details.

If `on_step` is `true`, the value will be logged on each step.
If `on_epoch` is `true`, the value will be accumulated and logged on each epoch.
In this case, the default reduction is the `mean` over the batches, which will also
take into account the batch size.
If both `on_step` and `on_epoch` are `true`, the values will be logged as 
`"$(name)_step"` and `"$(name)_epoch"`

# Arguments

- `trainer::Trainer`: The trainer object.
- `name::AbstractString`: The name of the value.
- `value`: The value to log.
- `batchsize`: The size of the current batch. Used only when `on_epoch == True`
              to compute the aggregate the batches. Defaults to `trainer.fit_state.batchsize`.
- `on_epoch::Bool`: Whether to log the value on each epoch. 
                    Defaults to `true` if `stage` is `:train_epoch_end` or `:val_epoch_end`, 
                    `false` otherwise.
- `on_step::Bool`: Whether to log the value on each step. 
                  Defaults to `true` if `stage` is `:training`, `false` for `:validation` and `:testing`.
- `prog_bar::Bool`: Whether to log the value to the progress bar. Defaults to `false`.

# Examples

```julia
function val_step(model::Model, trainer, batch, batch_idx)
    # log the validation loss
    ...
    Tsunami.log(trainer, "val/loss", val_loss)
end
```
"""
function log(trainer::Trainer, name::AbstractString, value; 
        on_step = nothing, 
        on_epoch = nothing,
        prog_bar = false, 
        batchsize = trainer.fit_state.batchsize)

    @unpack fit_state, metalogger  = trainer
    @unpack stage, step, epoch = fit_state
    
    if on_step === nothing
        if stage ∈ (:training,)
            on_step = true
        else
            on_step = false
        end
    end
    if on_epoch === nothing
        if stage ∈ (:validation, :testing, :train_epoch_end, :val_epoch_end, :test_epoch_end)
            on_epoch = true
        else
            on_epoch = false
        end
    end

    if on_step && on_epoch
        name_step = "$(name)_step"
        name_epoch = "$(name)_epoch"
    else
        name_step = name
        name_epoch = name
    end

    if on_step && (step % trainer.log_every_n_steps == 0) 
        # TODO log_every_n_steps should apply to both train and validate?
        log_step(metalogger, name_step, value, fit_state)
    end
    if on_epoch
        accumulate_epoch!(metalogger, name_epoch, value, stage, batchsize)
    end

    if prog_bar && stage ∈ (:training,) && on_step 
        store_for_train_prog_bar!(metalogger, name_step, value)
    end
    if prog_bar && stage ∈ (:validation, :testing) && on_epoch 
        store_for_val_prog_bar!(metalogger, name_step, value)
    end
end

@non_differentiable log(::Any...)