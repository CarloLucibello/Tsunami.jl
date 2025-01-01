"""
    FitState

A type storing the state of execution during a call to [`fit`](@ref). 

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

A type storing the training options to be passed to [`fit`](@ref).

A `Trainer` object also contains a field `fit_state` of type [`FitState`](@ref) mantaining updated information about 
the fit state during the execution of `fit`.

# Constructor Arguments

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
            Every output of the `train_step` will be logged every 50 steps.
            See also `log_every_n_steps`.
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

- **fit\\_state**: A [`FitState`](@ref) object storing the state of execution during a call to [`fit`](@ref).
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

model, fitstate = Tsunami.fit(model, trainer, train_dataloader, val_dataloader)
```
"""
mutable struct Trainer
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
    
    return Trainer(callbacks, default_root_dir, fast_dev_run, log_every_n_steps, loggers, metalogger, 
                    max_epochs, max_steps, progress_bar, val_every_n_epochs, 
                    fit_state, foil, lr_schedulers, optimisers)
end

Base.show(io::IO, ::MIME"text/plain", trainer::Trainer) = 
    container_show(io, trainer, brief=[:metalogger, :optimisers, :callbacks, :loggers])

"""
    fit!(model, trainer, train_dataloader, [val_dataloader]; [ckpt_path, ...]) -> fit_state

Mutating version of [`fit`](@ref), copying back the trained model into the input `model`.
If `ckpt_path` is not `nothing`, training is resumed from the checkpoint.

See [`fit`](@ref) for more details.
"""
function fit!(model, args...; ckpt_path = nothing, kws...)
    if ckpt_path !== nothing
        newmodel, fit_state = fit(ckpt_path, args...; kws...)
    else
        newmodel, fit_state = fit(model, args...; kws...)
    end
    copy!(model, newmodel)
    return fit_state
end

"""
    fit([ckpt_path,] model, trainer, train_dataloader, [val_dataloader]) -> (new_model, fit_state)

Train `model` using the configuration given by `trainer`.
If `ckpt_path` is given, training is resumed from the checkpoint.

Returns the trained model and a [`FitState`](@ref) object.

See also [`fit!`](@ref) for a mutating version.

# Arguments

- **ckpt\\_path**: Path of the checkpoint from which training is resumed.
- **model**: A Flux model subtyping [`FluxModule`](@ref).
- **trainer**: A [`Trainer`](@ref) object storing the configuration options for `fit`.
- **train\\_dataloader**: An iterator over the training dataset, typically a `Flux.DataLoader`.
- **val\\_dataloader**: An iterator over the validation dataset, typically a `Flux.DataLoader`. Default: `nothing`.

# Examples

```julia
model = ...
trainer = Trainer(max_epochs = 10)
model, fit_state = Tsunami.fit(model, trainer, train_dataloader, val_dataloader)

# Resume training from checkpoint
trainer = Trainer(max_epochs = 20) # train for 10 more epochs
ckpt_path = joinpath(fit_state.run_dir, "checkpoints", "ckpt_last.bson")
model′, fit_state′ = Tsunami.fit(ckpt_path, model, trainer, train_dataloader, val_dataloader)
```
"""
function fit(ckpt_path::AbstractString, model::FluxModule, trainer, args...; kws...)
    ckpt = load_checkpoint(ckpt_path)
    if haskey(ckpt, :model) # for backward compatibility
        model = ckpt.model
    else
        model = deepcopy(model)
        Flux.loadmodel!(model, ckpt.model_state)
    end
    trainer.fit_state = ckpt.fit_state
    trainer.lr_schedulers = ckpt.lr_schedulers
    trainer.optimisers = ckpt.optimisers
    return fit(model, trainer, args...; kws..., _resuming_from_ckpt = true)
end

function fit(
        model::FluxModule,
        trainer::Trainer,
        train_dataloader,
        val_dataloader = nothing;
        _resuming_from_ckpt = false
    )
    
    if !_resuming_from_ckpt
        model = deepcopy(model)
        trainer.fit_state = FitState()
    end
    fit_state = trainer.fit_state
    fit_state.should_stop = false
    
    tsunami_dir = joinpath(trainer.default_root_dir, "tsunami_logs")
    run_dir = dir_with_version(joinpath(tsunami_dir, "run"))
    fit_state.run_dir = run_dir
    set_run_dir!(trainer.metalogger, run_dir)

    print_fit_initial_summary(model, trainer)

    if _resuming_from_ckpt
        lr_schedulers = trainer.lr_schedulers
        optimisers = trainer.optimisers 
        start_epoch = fit_state.epoch + 1
    else # train from scratch
        optimisers, lr_schedulers = configure_optimisers(model, trainer) |> process_out_configure_optimisers
        start_epoch = 1
        fit_state.step = 0
    end
    fit_state.epoch = start_epoch - 1

    model, optimisers = setup(trainer.foil, model, optimisers)

    trainer.optimisers = optimisers
    trainer.lr_schedulers = lr_schedulers
 
    if trainer.fast_dev_run
        check_fluxmodule(model)
        check_train_step(model, trainer, first(train_dataloader))
        if val_dataloader !== nothing
            check_val_step(model, trainer, first(val_dataloader))
        end
        return model, fit_state
    end

    val_loop(model, trainer, val_dataloader; progbar_keep=false, progbar_print_epoch=true)

    for epoch in start_epoch:trainer.max_epochs # TODO turn into while loop
        fit_state.epoch = epoch
        train_loop(model, trainer, train_dataloader, val_dataloader)
        fit_state.should_stop && break
    end

    return model |> cpu, fit_state
end

function val_loop(model, trainer, val_dataloader; progbar_offset = 0, 
            progbar_keep = true, progbar_print_epoch = false)
    val_dataloader === nothing && return
    fit_state = trainer.fit_state
    fit_state.stage = :validation

    hook(on_val_epoch_start, model, trainer)

    progbar_desc = progbar_print_epoch ?  "Val Epoch $(fit_state.epoch): " : "Validation: "
    valprogressbar = Progress(_length(val_dataloader); desc=progbar_desc, 
        showspeed=true, enabled=trainer.progress_bar, color=:green, offset=progbar_offset, keep=progbar_keep)
    for (batch_idx, batch) in enumerate(val_dataloader)
        fit_state.batchsize = MLUtils.numobs(batch)

        hook(on_val_batch_start, model, trainer, batch, batch_idx)

        batch = setup_batch(trainer.foil, batch)
        val_step(model, trainer, batch, batch_idx)
        ProgressMeter.next!(valprogressbar, 
                showvalues = values_for_val_progressbar(trainer.metalogger),
                valuecolor = :green)

        hook(on_val_batch_end, model, trainer)
    end

    fit_state.stage = :val_epoch_end

    hook(on_val_epoch_end, model, trainer)

    val_results = log_epoch(trainer.metalogger, fit_state)
    return val_results
end

function train_loop(model, trainer, train_dataloader, val_dataloader)
    @unpack fit_state = trainer
    
    fit_state.stage = :training

    hook(on_train_epoch_start, model, trainer)

    if trainer.lr_schedulers !== nothing
        lr = trainer.lr_schedulers(fit_state.epoch)
        Optimisers.adjust!(trainer.optimisers, lr)
    end

    train_progbar = Progress(_length(train_dataloader); desc="Train Epoch $(fit_state.epoch): ", 
                        showspeed=true, enabled = trainer.progress_bar, color=:yellow)

    ## SINGLE EPOCH TRAINING LOOP
    for (batch_idx, batch) in enumerate(train_dataloader)
        fit_state.step += 1
        fit_state.batchsize = MLUtils.numobs(batch)
        
        hook(on_train_batch_start, model, trainer, batch, batch_idx)
        
        batch = setup_batch(trainer.foil, batch)
        
        loss, pb = pullback(model, trainer.foil) do model
            loss = train_step(model, trainer, batch, batch_idx)
            return loss
        end

        hook(on_before_backprop, model, trainer, loss)
        
        grad = pb()

        hook(on_before_update, model, trainer, grad)

        Optimisers.update!(trainer.optimisers, model, grad)

        if fit_state.step == trainer.max_steps
            fit_state.should_stop = true
        end
        
        ProgressMeter.next!(train_progbar,
            showvalues = values_for_train_progbar(trainer.metalogger),
            valuecolor = :yellow, 
            final = fit_state.should_stop || batch_idx == _length(train_dataloader),
            keep = fit_state.should_stop || fit_state.epoch == trainer.max_epochs
        )

        hook(on_train_batch_end, model, trainer)
        
        fit_state.should_stop && break
    end
    if fit_state.epoch == trainer.max_epochs
        fit_state.should_stop = true
    end

    ## EPOCH END
    fit_state.stage = :train_epoch_end
    hook(on_train_epoch_end, model, trainer)
    log_epoch(trainer.metalogger, fit_state)
    fit_state.stage = :training

    ## VALIDATION
    if  val_dataloader !== nothing && trainer.val_every_n_epochs > 0
        if  (fit_state.epoch % trainer.val_every_n_epochs == 0) || fit_state.should_stop
            val_loop(model, trainer, val_dataloader; 
                    progbar_offset = fit_state.should_stop ? 0 : train_progbar.numprintedvalues + 1, 
                    progbar_keep = fit_state.should_stop, 
                    progbar_print_epoch=true)
        end
    end
end

function process_out_configure_optimisers(out::Tuple)
    opt, lr_scheduler = out
    return opt, lr_scheduler
end

function process_out_configure_optimisers(out)
    opt = out
    lr_scheduler = nothing
    return opt, lr_scheduler
end

function print_fit_initial_summary(model, trainer)
    cuda_available = is_cuda_functional()
    amdgpu_available = is_amdgpu_functional()
    metal_available = is_metal_functional()
    if cuda_available
        str_gpuavail = "true (CUDA)"
        str_gpuused = is_using_gpu(trainer.foil)
    elseif amdgpu_available
        str_gpuavail = "true (AMDGPU)"
        str_gpuused = is_using_gpu(trainer.foil)
    elseif metal_available
        str_gpuavail = "true (Metal)"
        str_gpuused = is_using_gpu(trainer.foil)
    else
        str_gpuavail = "false"
        str_gpuused = "false"
    end
    @info "GPU available: $(str_gpuavail), used: $(str_gpuused)"
    @info "Model Summary:"
    show(stdout, MIME("text/plain"), model)
    println()
end

"""
    test(model::FluxModule, trainer, dataloader)

Run the test loop, calling the [`test_step`](@ref) method on the model for each batch returned by the `dataloader`.
Returns the aggregated results from the values logged in the `test_step` as a dictionary.

# Examples

```julia
julia> struct Model <: FluxModule end 

julia> function Tsunami.test_step(::Model, trainer, batch)
    Tsunami.log(trainer, "test/loss", rand())
end

julia> model, trainer = Model(), Trainer();

julia> test_results = Tsunami.test(model, trainer, [rand(2) for i=1:3]);
Testing: 100%|████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00 (6.04 μs/it)
  test/loss:  0.675

julia> test_results
Dict{String, Float64} with 1 entry:
  "test/loss" => 0.674665
```
"""
function test(model::FluxModule, trainer::Trainer, dataloader)
    model = setup_batch(trainer.foil, model)
    return test_loop(model, trainer, dataloader; progbar_keep=true)
end

function test_loop(model, trainer, dataloader; progbar_offset = 0, progbar_keep = true)
    dataloader === nothing && return
    fit_state = trainer.fit_state
    fit_state.stage = :testing

    hook(on_test_epoch_start, model, trainer)


    testprogressbar = Progress(_length(dataloader); desc="Testing: ", 
                                showspeed=true, enabled=trainer.progress_bar, 
                                color=:green, offset=progbar_offset, keep=progbar_keep)
    for (batch_idx, batch) in enumerate(dataloader)
        fit_state.batchsize = MLUtils.numobs(batch)

        hook(on_test_batch_start, model, trainer, batch, batch_idx)

        batch = setup_batch(trainer.foil, batch)
        test_step(model, trainer, batch, batch_idx)
        ProgressMeter.next!(testprogressbar, 
                showvalues = values_for_val_progressbar(trainer.metalogger),
                valuecolor = :green
                )

        hook(on_test_batch_end, model, trainer)
    end

    fit_state.stage = :test_epoch_end
    hook(on_test_epoch_end, model, trainer)
    test_results = log_epoch(trainer.metalogger, fit_state)
    return test_results
end

"""
    validate(model::FluxModule, trainer, dataloader)

Run the validation loop, calling the [`val_step`](@ref) method on the model for each batch returned by the `dataloader`.
Returns the aggregated results from the values logged in the `val_step` as a dictionary.

See also [`Tsunami.test`](@ref) and [`Tsunami.fit`](@ref).
"""
function validate(model::FluxModule, trainer::Trainer, dataloader)
    model = setup_batch(trainer.foil, model)
    return val_loop(model, trainer, dataloader; progbar_keep=true)
end
