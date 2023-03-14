

""""
    log(trainer::Trainer, name::AbstractString, value; 
        on_step = nothing, on_epoch = nothing, prog_bar = false)

Log a value to the logger. Can be called from any function in the training loop
or from a callback. Logs to the loggers specified in `trainer.loggers`.

See the [Logging](TODO) docs for more details.

# Arguments

- `trainer::Trainer`: The trainer object.
- `name::AbstractString`: The name of the value.
- `value`: The value to log.
- `on_step::Bool`: Whether to log the value on each step. 
                   Defaults to `true` if `stage` is `:train` or `:validate`, `false` otherwise.
- `on_epoch::Bool`: Whether to log the value on each epoch. 
                    Defaults to `true` if `stage` is `:training_epoch_end` or `:validation_epoch_end`, 
                    `false` otherwise.
- `prog_bar::Bool`: Whether to log the value to the progress bar. Defaults to `false`.

# Examples

```julia
function validation_step(model::Model, trainer, batch, batch_idx)
    # log the validation loss
    ...
    Tsunami.log(trainer, "val/loss", val_loss)
end
```
"""
function log(trainer::Trainer, name::AbstractString, value; 
        on_step = nothing, 
        on_epoch = nothing,
        prog_bar = false)

    @unpack fit_state, metalogger  = trainer
    @unpack stage, step, epoch = fit_state
    
    if on_step === nothing
        if stage ∈ (:train,)
            on_step = true
        else
            on_step = false
        end
    end
    if on_epoch === nothing
        if stage ∈ (:validate, :training_epoch_end, :validation_epoch_end)
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

    if on_step && (fit_state.step % trainer.log_every_n_steps == 0) 
        # TODO log_every_n_steps should apply to both train and validate?
        log_step(metalogger, name_step, value, step)
    end
    if on_epoch
        accumulate_epoch!(metalogger, name_epoch, value, stage)
    end

    if prog_bar && stage ∈ (:train,)
        store_for_prog_bar!(metalogger, name_step, value)
    end
end

### MetaLogger ###########################

mutable struct MetaLogger
    loggers::Vector
    training_epoch_stats::Stats
    validation_epoch_stats::Stats 
    values_for_train_progressbar::Dict{String, Any}
    last_step_last_epoch::Int
end

function MetaLogger(loggers)
    return MetaLogger(loggers, Stats(), Stats(), Dict{String, Any}(), 0)
end

function log_step(metalogger::MetaLogger, name::AbstractString, value, step)
    for logger in metalogger.loggers
        log_scalar(logger, name, value; step)
    end
end

function accumulate_epoch!(metalogger::MetaLogger, name::AbstractString, value, stage)
    stats = stage ∈ (:train, :training_epoch_end)  ? metalogger.training_epoch_stats : 
                                                     metalogger.validation_epoch_stats
    OnlineStats.fit!(stats, Dict(name => value))
end

function log_epoch(metalogger::MetaLogger, fit_state)
    @unpack stage, step, epoch = fit_state
    stats = stage ∈ (:train, :training_epoch_end)  ? metalogger.training_epoch_stats : 
                                                     metalogger.validation_epoch_stats
    if stage ∈ (:train, :training_epoch_end)
        for logger in metalogger.loggers
            for s in metalogger.last_step_last_epoch+1:step
                log_scalar(logger, "epoch", epoch; step=s)
            end
        end
        metalogger.last_step_last_epoch = step
    end
    if epoch == 0 
        @assert step == 0
        for logger in metalogger.loggers
            log_scalar(logger, "epoch", 0; step=0)
        end
    end

    for (name, value) in pairs(stats)
        for logger in metalogger.loggers
            log_scalar(logger, name, OnlineStats.value(value); step)
        end
    end
    clean_stats!(metalogger, stage)
end

function clean_stats!(metalogger::MetaLogger, stage)
    if stage ∈ (:train, :training_epoch_end)
        metalogger.training_epoch_stats = Stats()
    else
        metalogger.validation_epoch_stats = Stats()
    end
end

function store_for_prog_bar!(metalogger::MetaLogger, name::AbstractString, value)
    metalogger.values_for_train_progressbar[name] = value
end

function values_for_train_progressbar(metalogger::MetaLogger)
    return [(k, roundval(v)) for (k, v) in pairs(metalogger.values_for_train_progressbar)]
end

@non_differentiable log(::Any...)
@non_differentiable log_epoch(::Any...)
@non_differentiable log_step(::Any...)
@non_differentiable accumulate_epoch!(::Any...)

# function log_validation(tblogger, nsteps::Int, validation_epoch_out::NamedTuple)
#     if tblogger !== nothing
#         TensorBoardLogger.set_step!(tblogger, nsteps)
#         with_logger(tblogger) do
#             @info "Validation" validation_epoch_out...
#         end
#     end

#     #TODO customize with https://github.com/JuliaLogging/MiniLoggers.jl
#     f(k, v) = "$(k) = $(roundval(v))"

#     val_crayon = Crayon(foreground=:light_cyan, bold=true)
#     print(val_crayon, "Validation: ")
#     println(Crayon(foreground=:white, bold=false), "$(join([f(k, v) for (k, v) in pairs(validation_epoch_out)], ", "))")
# end

# function log_training_step(tblogger, epoch, step, out::NamedTuple)
#     if tblogger !== nothing
#         TensorBoardLogger.set_step!(tblogger, step)
#         with_logger(tblogger) do
#             @info "Training" epoch out...
#         end
#     end
# end