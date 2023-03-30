

"""
    log(trainer::Trainer, name::AbstractString, value; 
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
                  Defaults to `true` if `stage` is `:training` or `:validation`, `false` otherwise.
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
        if stage ∈ (:validation, :train_epoch_end, :val_epoch_end)
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
    if prog_bar && stage ∈ (:validation,) && on_epoch 
        store_for_val_prog_bar!(metalogger, name_step, value)
    end
end

### MetaLogger ###########################

mutable struct MetaLogger
    loggers::Vector
    train_epoch_stats::Stats
    val_epoch_stats::Stats
    values_for_train_progressbar::Dict{String, Any}
    values_for_val_progressbar::Dict{String, Any}
    last_step_printed::Int
end

function MetaLogger(loggers)
    return MetaLogger(loggers, Stats(), Stats(), Dict{String, Any}(), Dict{String, Any}(), -1)
end

function log_step(metalogger::MetaLogger, name::AbstractString, value, fit_state::FitState)
    @unpack step, epoch = fit_state
    for logger in metalogger.loggers
        log_scalar(logger, name, value; step)
    end
    if step > metalogger.last_step_printed
        metalogger.last_step_printed = step
        for logger in metalogger.loggers
            log_scalar(logger, "epoch", epoch; step)
        end
    end
end

function accumulate_epoch!(metalogger::MetaLogger, name::AbstractString, value, stage, batchsize)
    stats = stage ∈ (:training, :train_epoch_end)  ? metalogger.train_epoch_stats : 
                                                     metalogger.val_epoch_stats


    add_obs!(stats, name, value, batchsize)
end

function log_epoch(metalogger::MetaLogger, fit_state)
    @unpack stage, step, epoch = fit_state
    stats = stage ∈ (:training, :train_epoch_end)  ? metalogger.train_epoch_stats : 
                                                     metalogger.val_epoch_stats

    for (name, value) in pairs(stats)
        for logger in metalogger.loggers
            log_scalar(logger, name, value; step)
        end
    end
    clean_stats!(metalogger, stage)
end

function clean_stats!(metalogger::MetaLogger, stage)
    if stage ∈ (:training, :train_epoch_end)
        empty!(metalogger.train_epoch_stats)
    else
        empty!(metalogger.val_epoch_stats)
    end
end

function store_for_train_prog_bar!(metalogger::MetaLogger, name::AbstractString, value)
    metalogger.values_for_train_progressbar[name] = value
end

function store_for_val_prog_bar!(metalogger::MetaLogger, name::AbstractString, value)
    metalogger.values_for_val_progressbar[name] = value
end

function values_for_train_progbar(metalogger::MetaLogger)
    dict = metalogger.values_for_train_progressbar
    ks = sort(collect(keys(dict)))
    return [(k, roundval(dict[k])) for k in ks]
end

function values_for_val_progressbar(metalogger::MetaLogger)
    stats = metalogger.val_epoch_stats
    ks = sort(collect(keys(stats)))
    return [(k, roundval(stats[k])) for k in ks]
end


@non_differentiable log(::Any...)
@non_differentiable log_epoch(::Any...)
@non_differentiable log_step(::Any...)
@non_differentiable accumulate_epoch!(::Any...)
