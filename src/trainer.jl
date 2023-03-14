## TODO from PL ("train", "sanity_check", "validate", "test", "predict", "tune") 
# abstract type AbstractStage end
# struct TrainStage end 
# struct ValidateStage end 

"""
    FitState

A type storing the state of execution during a call to [`fit!`](@ref). 

A `FitState` object is part of a [`Trainer`](@ref) object.

# Fields

- `epoch`: current epoch.
- `run_dir`
- `stage`
- `step`: current step.
- `training_epoch_out`
- `validation_epoch_out`
"""
@kwdef mutable struct FitState  # TODO make all field const except for e.g. last_epoch?
    epoch::Int = 0
    run_dir::String = ""
    stage::Symbol = :train # [:train, :training_epoch_end, :validate, :validation_epoch_end]
    step::Int = 0
    optimisers = nothing  # TODO move to trainer?
    schedulers = nothing  # TODO move to trainer? in that case the trainer should be part of the checkpoint
end

Functors.@functor FitState

function Base.show(io::IO, ::MIME"text/plain", fit_state::FitState)
    # TODO exclude optimisers from print, too verbose
    container_show(io, fit_state)
end


"""
    Trainer(; kws...)

A type storing the training options to be passed to [`fit!`](@ref).

A `Trainer` object also contains a field `fit_state` of type [`FitState`](@ref) mantaining updated information about 
the fit state during the execution of `fit!`.

# Arguments

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
            Every output of the `training_step` will be logged every 50 steps.
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
                        Default: `1`.

# Examples

```julia
trainer = Trainer(max_epochs = 10, 
                  accelerator = :cpu,
                  checkpointer = true,
                  logger = true)

Tsunami.fit!(model, trainer; train_dataloader, val_dataloader)
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
end

# function  Trainer(;
#         accelerator::Symbol = :auto,
#         callbacks = [],
#         default_root_dir::String = pwd(),
#         checkpointer::Bool = true,
#         devices::Union{Int, Nothing} = nothing,
#         fast_dev_run::Bool = false,
#         log_every_n_steps::Int = 50,
#         logger::Bool = true,
#         max_epochs::Union{Int, Nothing} = nothing,
#         max_steps::Int = -1,
#         progress_bar::Bool = true,
#         val_every_n_epochs::Int = 1,
#         fit_state::FitState = FitState(),
#     )

#     Trainer(accelerator, default_root_dir, callbacks, checkpointer, devices, fast_dev_run, log_every_n_steps, logger, max_epochs, max_steps, progress_bar, val_every_n_epochs, fit_state)
# end


function validation_loop(model, trainer; val_dataloader, device)
    val_dataloader === nothing && return
    fit_state = trainer.fit_state
    oldstage = fit_state.stage
    fit_state.stage = :validate
    valprogressbar = Progress(length(val_dataloader); desc="Validation: ", showspeed=true, enabled=false) # TODO doesn't work
    for (batch_idx, batch) in enumerate(val_dataloader)
        batch = batch |> device
        validation_step(model, trainer, batch, batch_idx)
        ProgressMeter.next!(valprogressbar)
    end

    fit_state.stage = :validation_epoch_end
    on_validation_epoch_end(model, trainer)
    for cbk in trainer.callbacks
        on_validation_epoch_end(cbk, model, trainer)
    end
    log_epoch(trainer.metalogger, fit_state)
    fit_state.stage = oldstage
end

function training_loop(model, trainer; train_dataloader, val_dataloader, device, max_steps)
    fit_state = trainer.fit_state
    @unpack epoch, optimisers = fit_state
    opt = optimisers
    oldstage = fit_state.stage
    fit_state.stage = :train

    if fit_state.schedulers !== nothing
        lr = fit_state.schedulers(epoch)
        Optimisers.adjust!(opt, lr)
    end
    
    progressbar = Progress(length(train_dataloader); desc="Train Epoch $epoch: ", 
                        showspeed=true, enabled = trainer.progress_bar, color=:yellow)

    # SINGLE EPOCH TRAINING LOOP
    for (batch_idx, batch) in enumerate(train_dataloader)
        fit_state.step += 1
        
        batch = batch |> device

        grads = Zygote.gradient(model) do model
            loss = training_step(model, trainer, batch, batch_idx)
            return loss
        end
        opt, model = Optimisers.update!(opt, model, grads[1])

        ProgressMeter.next!(progressbar,
            # showvalues = process_out_for_progress_bar(last(training_step_outs), training_step_out_avg),
            valuecolor=:yellow)

        fit_state.step == max_steps && break
    end

    fit_state.stage = :training_epoch_end
    on_train_epoch_end(model, trainer)
    for cbk in trainer.callbacks
        on_train_epoch_end(cbk, model, trainer)
    end
    log_epoch(trainer.metalogger, fit_state)
    fit_state.stage = :train

    if  (val_dataloader !== nothing && epoch % trainer.val_every_n_epochs == 0)
        validation_loop(model, trainer; val_dataloader, device)
    end

    fit_state.stage = oldstage
end

"""
    fit!(model::FluxModule, trainer::Trainer; 
         train_dataloader, [val_dataloader, ckpt_path, resume_run])

Train a `model` using the configuration given by `trainer`.
If `ckpt_path` is not `nothing`, training is resumed from the checkpoint.

# Arguments

- **model**: A Flux model subtyping [`FluxModule`](@ref).
- **trainer**: A [`Trainer`](@ref) object storing the configuration options for `fit!`.
- **train\\_dataloader**: An iterator over the training dataset, typically a `Flux.DataLoader`. Required argument.
- **val\\_dataloader**: An iterator over the validation dataset, typically a `Flux.DataLoader`. Default: `nothing`.
- **ckpt\\_path**: Path of the checkpoint from which training is resumed (if given). Default: `nothing`.
- **resume\\_run**: If `true` and `ckpt_path` is given, continue the loggin of the run in the same
                   directory instead of creating a new one. Default: `false`.

# Examples

```julia
trainer = Trainer(max_epochs = 10)
Tsunami.fit!(model, trainer; train_dataloader, val_dataloader)
```
"""
function fit!(
        model::FluxModule,
        trainer::Trainer;
        ckpt_path = nothing,
        resume_run = false,
        train_dataloader,
        val_dataloader = nothing,
    )
    @assert !resume_run "resume_run=true is not supported yet." # TODO
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
    trainer.metalogger = MetaLogger(trainer.loggers)

    # for logger in trainer.loggers
    #     reset_run_dir!(logger, run_dir)
    # end
    
    max_steps, max_epochs = compute_max_steps_and_epochs(trainer.max_steps, trainer.max_epochs)
    
    if trainer.fast_dev_run
        max_epochs = 1
        max_steps = 1
        trainer.val_every_n_epochs = 1
        trainer.loggers = []
        trainer.metalogger = nothing

        check_fluxmodule(model)
        # check forwards on cpu
        check_training_step(model, trainer, first(train_dataloader))
        if val_dataloader !== nothing
            check_validation_step(model, trainer, first(val_dataloader))
        end
    end

    print_fit_initial_summary(model, trainer, device)

    fit_state.step = 0
    if ckpt_path !== nothing
        model, ckpt_fit_state = load_checkpoint(ckpt_path)
        start_epoch = ckpt_fit_state.epoch + 1
        opt = ckpt_fit_state.optimisers
        lr_scheduler = ckpt_fit_state.schedulers
        fit_state.step = ckpt_fit_state.step
    else
        opt, lr_scheduler = configure_optimisers(model, trainer) |> process_out_configure_optimisers
        start_epoch = 1
    end
    
    model = model |> device
    opt = opt |> device
    fit_state.optimisers = opt
    fit_state.schedulers = lr_scheduler
 
    validation_loop(model, trainer; val_dataloader, device)

    for epoch in start_epoch:max_epochs
        fit_state.epoch = epoch

        training_loop(model, trainer; train_dataloader, val_dataloader, device, max_steps)

        (fit_state.step == max_steps || epoch == max_epochs) && break
    end

    model = model |> cpu
    if model !== input_model
        copy!(input_model, model)
    end
    return fit_state
end

# process_out_training_step(training_step_out::Number) = training_step_out, (; loss=training_step_out)
# process_out_training_step(training_step_out::NamedTuple) = training_step_out.loss, training_step_out

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

function process_out_for_progress_bar(out::NamedTuple, s::Stats)
    f(k, v) = "$(roundval(v)) (last)  $(roundval(OnlineStats.value(s[k]))) (expavg)"
    return [(k, f(k, v)) for (k, v) in pairs(out)]
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
