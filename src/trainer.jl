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
"""
@kwdef mutable struct FitState
    epoch::Int = 0
    run_dir::String = ""
    stage::Symbol = :training # [:training, :train_epoch_end, :validation, :val_epoch_end]
    step::Int = 0
    batchsize::Int = 0
end

Functors.@functor FitState

function Base.show(io::IO, ::MIME"text/plain", fit_state::FitState)
    container_show(io, fit_state)
end


"""
    Trainer(; kws...)

A type storing the training options to be passed to [`fit!`](@ref).

A `Trainer` object also contains a field `fit_state` of type [`FitState`](@ref) mantaining updated information about 
the fit state during the execution of `fit!`.

# Constructor Arguments

- **accelerator**: Supports passing different accelerator types `(:cpu, :gpu,  :auto)`.
                `:auto` will automatically select a gpu if available.
                See also the `devices` option.
                 Default: `:auto`.
- **callbacks**: Pass a single or a list of callbacks. Default `nothing`.
- **checkpointer**: If `true`, enable checkpointing.
                    Default: `true`.

- **default\\_root\\_dir** : Default path for logs and weights.
                      Default: `pwd()`.
                    
- **devices**: Pass an integer `n` to train on `n` devices, 
            or a list of devices ids to train on specific devices.
            If `nothing`, will use all available devices. 
            Default: `nothing`.

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

# Additional Fields

- **fit\\_state**: A [`FitState`](@ref) object storing the state of execution during a call to [`fit!`](@ref).
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
@kwdef mutable struct Trainer
    accelerator::Symbol = :auto
    callbacks = []
    checkpointer::Bool = true
    default_root_dir::String = pwd()
    devices::Union{Int, Nothing} = nothing
    fast_dev_run::Bool = false
    log_every_n_steps::Int = 50
    logger::Bool = true
    loggers = []
    metalogger = nothing
    max_epochs::Union{Int, Nothing} = nothing
    max_steps::Int = -1
    progress_bar::Bool = true
    val_every_n_epochs::Int = 1

    fit_state::FitState = FitState()
    lr_schedulers = nothing
    optimisers = nothing
end

function val_loop(model, trainer, val_dataloader; device, progbar_offset = 0, progbar_keep = true)
    val_dataloader === nothing && return
    fit_state = trainer.fit_state
    oldstage = fit_state.stage
    fit_state.stage = :validation

    valprogressbar = Progress(length(val_dataloader); desc="Val Epoch $(fit_state.epoch): ", 
        showspeed=true, enabled=trainer.progress_bar, color=:green, offset=progbar_offset, keep=progbar_keep)
    for (batch_idx, batch) in enumerate(val_dataloader)
        fit_state.batchsize = MLUtils.numobs(batch)
        batch = batch |> device
        val_step(model, trainer, batch, batch_idx)
        ProgressMeter.next!(valprogressbar, 
                showvalues = values_for_val_progressbar(trainer.metalogger),
                valuecolor = :green
                )
    end
    ProgressMeter.finish!(valprogressbar)

    fit_state.stage = :val_epoch_end
    on_val_epoch_end(model, trainer)
    for cbk in trainer.callbacks
        on_val_epoch_end(cbk, model, trainer)
    end
    log_epoch(trainer.metalogger, fit_state)
    fit_state.stage = oldstage
end

function train_loop(model, trainer, train_dataloader, val_dataloader; device, max_steps)
    @unpack fit_state = trainer
    
    oldstage = fit_state.stage
    fit_state.stage = :training
    islastepoch = fit_state.epoch == trainer.max_epochs

    if trainer.lr_schedulers !== nothing
        lr = trainer.lr_schedulers(fit_state.epoch)
        Optimisers.adjust!(trainer.optimisers, lr)
    end

    train_progbar = Progress(length(train_dataloader); desc="Train Epoch $(fit_state.epoch): ", 
                        showspeed=true, enabled = trainer.progress_bar, color=:yellow,
                        keep = islastepoch)

    ## SINGLE EPOCH TRAINING LOOP
    for (batch_idx, batch) in enumerate(train_dataloader)
        fit_state.step += 1
        fit_state.batchsize = MLUtils.numobs(batch)
        
        batch = batch |> device

        grads = Flux.gradient(model) do model
            loss = train_step(model, trainer, batch, batch_idx)
            return loss
        end

        Optimisers.update!(trainer.optimisers, model, grads[1])

        ProgressMeter.next!(train_progbar,
            showvalues = values_for_train_progbar(trainer.metalogger),
            valuecolor = :yellow)

        fit_state.step == max_steps && break
    end
    ProgressMeter.finish!(train_progbar)

    ## EPOCH END
    fit_state.stage = :train_epoch_end
    on_train_epoch_end(model, trainer)
    for cbk in trainer.callbacks
        on_train_epoch_end(cbk, model, trainer)
    end
    log_epoch(trainer.metalogger, fit_state)
    fit_state.stage = :training

    ## VALIDATION
    if  val_dataloader !== nothing && trainer.val_every_n_epochs > 0
        if  (fit_state.epoch % trainer.val_every_n_epochs == 0) || islastepoch
            val_loop(model, trainer, val_dataloader; device, 
                    progbar_offset = islastepoch ? 0 : train_progbar.numprintedvalues + 1, 
                    progbar_keep = islastepoch)
        end
    end

    fit_state.stage = oldstage
end

"""
    fit!(model::FluxModule, trainer::Trainer, train_dataloader,
        [val_dataloader]; [ckpt_path, resume_run])

Train a `model` using the configuration given by `trainer`.
If `ckpt_path` is not `nothing`, training is resumed from the checkpoint.

# Arguments

- **model**: A Flux model subtyping [`FluxModule`](@ref).
- **trainer**: A [`Trainer`](@ref) object storing the configuration options for `fit!`.
- **train\\_dataloader**: An iterator over the training dataset, typically a `Flux.DataLoader`.
- **val\\_dataloader**: An iterator over the validation dataset, typically a `Flux.DataLoader`. Default: `nothing`.
- **ckpt\\_path**: Path of the checkpoint from which training is resumed (if given). Default: `nothing`.

# Examples

```julia
trainer = Trainer(max_epochs = 10)
Tsunami.fit!(model, trainer, train_dataloader, val_dataloader)
```
"""
function fit!(
        model::FluxModule,
        trainer::Trainer,
        train_dataloader,
        val_dataloader = nothing;
        ckpt_path = nothing,
    )
    
    input_model = model
    trainer.fit_state = FitState()
    fit_state = trainer.fit_state

    tsunami_dir = joinpath(trainer.default_root_dir, "tsunami_logs")
    run_dir = dir_with_version(joinpath(tsunami_dir, "run"))
    fit_state.run_dir = run_dir
    
    if trainer.checkpointer && !any(x -> x isa Checkpointer, trainer.callbacks)
        push!(trainer.callbacks, Checkpointer())
    end

    device = select_device(trainer.accelerator, trainer.devices)
    
    if trainer.logger && isempty(trainer.loggers)
        push!(trainer.loggers, TensorBoardLogger(run_dir))
    end
    trainer.metalogger = MetaLogger(trainer.loggers) # TODO move to trainer constructor
    
    max_steps, max_epochs = compute_max_steps_and_epochs(trainer.max_steps, trainer.max_epochs)
    
    if trainer.fast_dev_run
        # max_steps = 1
        # max_epochs = 1
        trainer.val_every_n_epochs = 1
        empty!(trainer.loggers)

        check_fluxmodule(model)
        # check forwards on cpu
        check_train_step(model, trainer, first(train_dataloader))
        if val_dataloader !== nothing
            check_val_step(model, trainer, first(val_dataloader))
        end
        return fit_state
    end

    print_fit_initial_summary(model, trainer, device)

    fit_state.step = 0
    if ckpt_path !== nothing # load checkpoint and resume training
        model, ckpt_fit_state, lr_schedulers, optimisers = load_checkpoint(ckpt_path)
        start_epoch = ckpt_fit_state.epoch + 1
        fit_state.step = ckpt_fit_state.step
    else # train from scratch
        optimisers, lr_schedulers = configure_optimisers(model, trainer) |> process_out_configure_optimisers
        start_epoch = 1
    end
    fit_state.epoch = start_epoch - 1

    model = model |> device
    trainer.optimisers = optimisers |> device
    trainer.lr_schedulers = lr_schedulers
 
    val_loop(model, trainer, val_dataloader; device, progbar_keep=false)

    for epoch in start_epoch:max_epochs
        fit_state.epoch = epoch

        train_loop(model, trainer, train_dataloader, val_dataloader; device, max_steps)

        (fit_state.step == max_steps || epoch == max_epochs) && break
    end

    model = model |> cpu
    if model !== input_model
        copy!(input_model, model)
    end

    return fit_state
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

function compute_max_steps_and_epochs(max_steps, max_epochs)
    if max_steps == -1 && max_epochs === nothing
        max_epochs = 1000
    end
    if max_epochs == -1
        max_epochs = nothing
    end
    return max_steps, max_epochs
end

function select_device(accelerator::Symbol, devices)
    if accelerator == :auto
        if CUDA.functional()
            return select_cuda_device(devices)
        else
            return cpu
        end
    elseif accelerator == :cpu
        return cpu
    elseif accelerator == :gpu
        if !CUDA.functional()
            @warn "CUDA is not available"
            return cpu
        else
            return select_cuda_device(devices)
        end
    else
        throw(ArgumentError("accelerator must be one of :auto, :cpu, :gpu"))
    end
end

select_cuda_device(devices::Nothing) = gpu

function select_cuda_device(devices::Int)
    @assert devices == 1 "Only one device is supported"
    return gpu
end

function select_cuda_device(devices::Union{Vector{Int}, Tuple})
    @assert length(devices) == 1 "Only one device is supported"
    CUDA.device!(devices[1])
    return gpu
end
 
function print_fit_initial_summary(model, trainer, device)
    cuda_available = CUDA.functional()
    use_cuda = cuda_available && device === gpu
    str_gpuavail = cuda_available ? "true (CUDA)" : "false"
    @info "GPU available: $(str_gpuavail), used: $use_cuda"
    @info "Model Summary:"
    show(stdout, MIME("text/plain"), model)
    println()
end

"""
    test(model::FluxModule, trainer, dataloader)

Run the test loop, calling the [`test_step`](@ref) method on the model for each batch returned by the `dataloader`.
Return the aggregated results from the values logged in the `test_step` as a dictionary.

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
function test(
        model::FluxModule,
        trainer::Trainer,
        test_dataloader,
    )
    
    device = select_device(trainer.accelerator, trainer.devices)
    

    # tsunami_dir = joinpath(trainer.default_root_dir, "tsunami_logs")
    # run_dir = dir_with_version(joinpath(tsunami_dir, "run"))
    # fit_state.run_dir = run_dir
    # if trainer.logger && isempty(trainer.loggers)
    #     push!(trainer.loggers, TensorBoardLogger(run_dir))
    # end
    trainer.metalogger = MetaLogger(trainer.loggers) # TODO move to trainer constructor
    
    model = model |> device
    
    test_loop(model, trainer, test_dataloader; device, progbar_keep=true)
end

function test_loop(model, trainer, test_dataloader; device, progbar_offset = 0, progbar_keep = true)
    test_dataloader === nothing && return
    fit_state = trainer.fit_state
    oldstage = fit_state.stage
    fit_state.stage = :testing


    testprogressbar = Progress(length(test_dataloader); desc="Testing: ", 
        showspeed=true, enabled=trainer.progress_bar, color=:green, offset=progbar_offset, keep=progbar_keep)
    for (batch_idx, batch) in enumerate(test_dataloader)
        fit_state.batchsize = MLUtils.numobs(batch)
        batch = batch |> device
        test_step(model, trainer, batch, batch_idx)
        ProgressMeter.next!(testprogressbar, 
                showvalues = values_for_val_progressbar(trainer.metalogger),
                valuecolor = :green
                )
    end
    ProgressMeter.finish!(testprogressbar)

    fit_state.stage = :test_epoch_end
    on_test_epoch_end(model, trainer)
    for cbk in trainer.callbacks
        on_test_epoch_end(cbk, model, trainer)
    end
    test_results = log_epoch(trainer.metalogger, fit_state)
    fit_state.stage = oldstage
    return test_results
end
