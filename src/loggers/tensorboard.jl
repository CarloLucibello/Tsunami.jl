"""
    TensorBoardLogger(run_dir)

A logger that writes to writes tensorboard events to the
`run_dir` directory. Relies on the `TensorBoardLogger.jl` package.
"""
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

##############################################################################

"""
    read_tensorboard_logs(logdir)

Reads all tensorboard events from the `logdir` path and returns them as a list of 
`(name, step, value)` tuples.

# Example

```julia
julia> fit_state = Tsunami.fit!(model, trainer, train_dataloader);

julia> Tsunami.read_tensorboard_logs(fit_state.run_dir)
24-element Vector{Tuple{String, Int64, Any}}:
("train/loss", 1, 2.509954f0)
 ("epoch", 1, 1.0f0)
 ("train/acc", 1, 0.0f0)
 ("train/loss", 2, 2.2748244f0)
 ("epoch", 2, 1.0f0)
 ("train/acc", 2, 0.5f0)
 ...

# Convert to a DataFrame
julia> df = DataFrame([(; name, step, value) for (name, step, value) in events]);

julia> unstack(df, :step, :name, :value)
8×4 DataFrame
 Row │ step   train/loss  epoch     train/acc 
     │ Int64  Float32?    Float32?  Float32?  
─────┼────────────────────────────────────────
   1 │     1     2.50995       1.0   0.0
   2 │     2     2.27482       1.0   0.5
   3 │     3     2.06172       2.0   0.333333
   4 │     4     1.72649       2.0   1.0
   5 │     5     1.57971       3.0   1.0
   6 │     6     1.39933       3.0   1.0
   7 │     7     1.17671       4.0   1.0
   8 │     8     1.17483       4.0   1.0
```
"""
function read_tensorboard_logs(logdir)
    events = Tuple{String,Int, Any}[]
    TensorBoardLoggers.map_summaries((x...) -> push!(events, x), logdir)
    return events
end
