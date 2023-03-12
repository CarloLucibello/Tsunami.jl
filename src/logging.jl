function log(trainer::Trainer, name::AbstractString, value; 
        on_step = nothing, 
        on_epoch = nothing,
        prog_bar = false)

    fit_state  = trainer.fit_state
    @unpack stage, step, epoch = fit_state
    logger = trainer.logger
    if on_step === nothing
        if stage === :train || stage === :validate
            on_step = true
        else
            on_step = false
        end
    end
    if on_epoch === nothing
        if stage === :training_epoch_end || stage === :validation_epoch_end
            on_epoch = true
        else
            on_epoch = false
        end
    end

    if on_step
        log_any(logger, name, value; step=step)
    end
    if on_epoch
        ## accumulate statistics
    end
end


log_any(logger::TBLogger, name::AbstractString, value::Number; step) = 
    TensorBoardLogger.log_value(logger, name, value; step=step(logger))



function log_validation(tblogger, nsteps::Int, validation_epoch_out::NamedTuple)
    if tblogger !== nothing
        TensorBoardLogger.set_step!(tblogger, nsteps)
        with_logger(tblogger) do
            @info "Validation" validation_epoch_out...
        end
    end

    #TODO customize with https://github.com/JuliaLogging/MiniLoggers.jl
    f(k, v) = "$(k) = $(roundval(v))"

    val_crayon = Crayon(foreground=:light_cyan, bold=true)
    print(val_crayon, "Validation: ")
    println(Crayon(foreground=:white, bold=false), "$(join([f(k, v) for (k, v) in pairs(validation_epoch_out)], ", "))")
end

function log_training_step(tblogger, epoch, step, out::NamedTuple)
    if tblogger !== nothing
        TensorBoardLogger.set_step!(tblogger, step)
        with_logger(tblogger) do
            @info "Training" epoch out...
        end
    end
end