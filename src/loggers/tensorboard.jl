mutable struct TensorBoardLogger
    tblogger::TBLogger
end

function TensorBoardLogger(run_dir::AbstractString)
    tblogger = TBLogger(run_dir, tb_append, step_increment=0)
    return TensorBoardLogger(tblogger)
end

log_scalar(logger::TensorBoardLogger, name::Symbol, value; step::Int) = 
    log_scalar(logger, string(name), value; step=step)

function log_scalar(logger::TensorBoardLogger, name::AbstractString, value; step::Int)
    TensorBoardLoggers.log_value(logger.tblogger, name, value; step)
end

