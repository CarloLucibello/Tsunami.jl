mutable struct TensorBoardLogger
    tblogger::TBLogger
    training_epoch_stats
    validation_epoch_stats
end

function TensorBoardLogger(run_dir)
    tblogger = TBLogger(run_dir, tb_append, step_increment=0)
    return TensorBoardLogger(tblogger, Stats(), Stats())
end

function log_step(logger::TensorBoardLogger, name::AbstractString, value, step)
    TensorBoardLoggers.log_value(logger.tblogger, name, value; step)
end

function accumulate_epoch!(logger::TensorBoardLogger, name::AbstractString, value, stage)
    stats = stage ∈ (:train, :training_epoch_end)  ? logger.training_epoch_stats : 
                                                     logger.validation_epoch_stats
    OnlineStats.fit!(stats[name], value)
end

function log_epoch(logger::TensorBoardLogger, stage, epoch)
    stats = stage ∈ (:train, :training_epoch_end)  ? logger.training_epoch_stats : 
                                                     logger.validation_epoch_stats
    for (name, value) in pairs(stats)
        TensorBoardLoggers.log_value(logger.tblogger, name, value; step=epoch)
    end
    clean_stats!(logger, stage)
end

function clean_stats!(logger::TensorBoardLogger, stage)
    if stage ∈ (:train, :training_epoch_end)
        logger.training_epoch_stats = Stats()
    else
        logger.validation_epoch_stats = Stats()
    end
end

function reset_run_dir!(logger::TensorBoardLogger, run_dir = logger.tblogger.logdir)
    logger.tblogger.logdir = run_dir
    TensorBoardLoggers.reset!(logger.tblogger)
end
