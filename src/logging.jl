

""""
    log(trainer::Trainer, name::AbstractString, value; 
        on_step = nothing, on_epoch = nothing, prog_bar = false)

Log a value to the logger. 

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
"""
function log(trainer::Trainer, name::AbstractString, value; 
        on_step = nothing, 
        on_epoch = nothing,
        prog_bar = false)

    fit_state  = trainer.fit_state
    @unpack stage, step, epoch = fit_state
    loggers = trainer.loggers
    isempty(loggers) && return

    if on_step === nothing
        if stage ∈ (:train, :validate)
            on_step = true
        else
            on_step = false
        end
    end
    if on_epoch === nothing
        if stage ∈ (:training_epoch_end, :validation_epoch_end)
            on_epoch = true
        else
            on_epoch = false
        end
    end

    if on_step && (fit_state.step % trainer.log_every_n_steps == 0) 
        # TODO log_every_n_steps should apply to both train and validate?
        for logger in loggers
            log_step(logger, name, value, step)
        end
    end
    if on_epoch
        for logger in loggers
            accumulate_epoch!(logger, name, value, stage)
        end
    end
end

function log_epoch(trainer::Trainer)
    for logger in trainer.loggers
        log_epoch(logger, trainer.fit_state.stage, trainer.fit_state.epoch)
    end 
end

## LOGGER API 
function log_step end
function log_epoch end
function accumulate_epoch! end  

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