"""
    FitState

A type storing the state of execution during a call to [`fit!`](@ref). 

A `FitState` object is part of a [`Trainer`](@ref) object.

# Fields

- `epoch`: the current epoch number.
- `run_dir`: the directory where the logs and checkpoints are saved.
- `stage`: the current stage of execution. One of `:training`, `:train_epoch_end`, `:validation`, `:val_epoch_end`.
- `step`: the current step number.
- `batchsize`: number of samples in the current batch.
- `should_stop`: set to `true` to stop the training loop.
"""
@kwdef mutable struct FitState
    epoch::Int = 0
    run_dir::String = ""
    stage::Symbol = :training # [:training, :train_epoch_end, :validation, :val_epoch_end]
    step::Int = 0
    batchsize::Int = 0
    should_stop::Bool = false
end

# Base.show(io::IO, fit_state::FitState) = print(io, "FitState()")
Base.show(io::IO, ::MIME"text/plain", fit_state::FitState) = container_show(io, fit_state)

"""
    Trainer(; kws...)

A type storing the training options to be passed to [`fit!`](@ref).

A `Trainer` object also contains a field `fit_state` of type [`FitState`](@ref) mantaining updated information about 
the fit state during the execution of `fit!`.

# Constructor Arguments

- **autodiff**: The automatic differentiation engine to use. 
                Possible values are `:zygote` and `:enzyme` . Default: `:zygote`.
- **callbacks**: Pass a single or a list of callbacks. Default `nothing`.
- **checkpointer**: If `true`, enable checkpointing.
                    Default: `true`.

- **default\\_root\\_dir** : Default path for logs and weights.
                      Default: `pwd()`.
                    
- **fast\\_dev\\_run**: If set to `true` runs a single batch for train and validation to find any bugs. 
             Default: `false`.

- **log\\_every\\_n\\_steps**: How often to log within steps. See also `logger`.
             Default: `50`.

- **logger**: If `true` use tensorboard for logging.
            Every output of the `train_step` will be logged every 50 steps by default.
            Set `log_every_n_steps` to change this.
            Default: `true`.

- **max\\_epochs**: Stop training once this number of epochs is reached. 
                Disabled by default (`nothing`). 
                If both `max_epochs` and `max_steps` are not specified, 
                defaults to `max_epochs = 1000`. To enable infinite training, set `max_epochs` = -1.
                Default: `nothing`.

- **max\\_steps**: Stop training after this number of steps. 
               Disabled by default (`-1`). 
               If `max_steps = -1` and `max_epochs = nothing`, will default to `max_epochs = 1000`. 
               To enable infinite training, set `max_epochs` to `-1`.
               Default: `-1`.

- **progress\\_bar**: It `true`, shows a progress bar during training. 
                  Default: `true`.

- **val\\_every\\_n\\_epochs**: Perform a validation loop every after every N training epochs.
                        The validation loop is in any case performed at the end of the last training epoch.
                        Set to 0 or negative to disable validation.
                        Default: `1`.

The constructor also take any of the [`Foil`](@ref)'s constructor arguments:

$FOIL_CONSTRUCTOR_ARGS

# Fields

Besides most of the constructor arguments, a `Trainer` object also contains the following fields:

- **fit\\_state**: A [`FitState`](@ref) object storing the state of execution during a call to [`fit!`](@ref).
- **foil**: A [`Foil`](@ref) object.
- **loggers**: A list of loggers.
- **lr\\_schedulers**: The learning rate schedulers used for training.
- **optimisers**: The optimisers used for training.

# Examples

```julia
trainer = Trainer(max_epochs = 10, 
                  accelerator = :cpu,
                  checkpointer = true,
                  logger = true)

Tsunami.fit!(model, trainer, train_dataloader, val_dataloader)
```
"""
mutable struct Trainer
    autodiff::Symbol
    callbacks::Vector
    default_root_dir::AbstractString
    fast_dev_run::Bool
    log_every_n_steps::Int
    loggers::Vector
    metalogger::MetaLogger
    max_epochs::Int
    max_steps::Int
    progress_bar::Bool
    val_every_n_epochs::Int

    fit_state::FitState
    foil::Foil
    lr_schedulers
    optimisers
end

function Trainer(;
            autodiff::Symbol = :zygote,
            callbacks = [],
            checkpointer = true,
            default_root_dir = pwd(),
            fast_dev_run = false,
            log_every_n_steps = 50,
            logger = true,
            loggers = [],
            max_epochs = nothing,
            max_steps = -1,
            progress_bar = true,
            val_every_n_epochs = 1,
            foil_kws...
         )
    if autodiff == :zygote
        is_loaded(:Zygote) || throw(ArgumentError("Zygote.jl must be loaded to use autodiff=:zygote. Run `using Zygote`."))
    elseif autodiff == :enzyme
        is_loaded(:Enzyme) || throw(ArgumentError("Enzyme.jl must be loaded to use autodiff=:enzyme. Run `using Enzyme`."))
    else
        throw(ArgumentError("autodiff must be either :zygote or :enzyme"))
    end

    fit_state = FitState()
    foil = Foil(; foil_kws...)
    lr_schedulers = nothing
    optimisers = nothing
    loggers = copy(loggers)  # copy to avoid mutating the original list
    callbacks = copy(callbacks)

    if checkpointer && !any(x -> x isa Checkpointer, callbacks)
        push!(callbacks, Checkpointer())
    end
    if logger && isempty(loggers)
        push!(loggers, TensorBoardLogger())
    end
    metalogger = MetaLogger(loggers)

    if max_steps == -1 && max_epochs === nothing
        max_epochs = 1000
    end
    if max_epochs === nothing  || max_epochs < 0
        max_epochs = typemax(Int)
    end

    if fast_dev_run
        val_every_n_epochs = 1
        loggers = []
        metalogger = MetaLogger(loggers)
    end
    
    return Trainer(autodiff, callbacks, default_root_dir, fast_dev_run, 
                    log_every_n_steps, loggers, metalogger, 
                    max_epochs, max_steps, progress_bar, val_every_n_epochs, 
                    fit_state, foil, lr_schedulers, optimisers)
end

Base.show(io::IO, ::MIME"text/plain", trainer::Trainer) = 
    container_show(io, trainer, brief=[:metalogger, :optimisers, :callbacks, :loggers])
