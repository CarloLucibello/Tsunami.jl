
"""
    Trainer(; kws...)

A type storing the training options to be passed to [`fit!`](@ref).

# Arguments

- **accelerator**: Supports passing different accelerator types `(:cpu, :gpu,  :auto)`.
                `:auto` will automatically select a gpu if available.
                See also the `devices` option.
                 Default: `:auto`.

- **val\_every\_n\_epoch**: Perform a validation loop every after every N training epochs. 
                       Default: `1`.

- **checkpointer**:   If `true`, enable checkpointing.
                         Default: `true`.

- **default\_root\_dir**: Default path for logs and weights.
                      Default: `pwd()`.

- **devices**: Devices identificaiton number(s). 
            Use an integer `n` to train on `n` devices, 
            or a list to train on specific devices.
            If `nothing`, will use all available devices. 
            Default: `nothing`.

- **fast\_dev\_run**: If set to `true` runs a single batch for train and validation to find any bugs. 
             Default: `false`.

- **logger**: If `true` use tensorboard for logging.
            Every output of the `training_step` will be logged every 50 steps.
            Default: `true`.

- **max\_epochs**: Stop training once this number of epochs is reached. 
                Disabled by default (`nothing`). 
                If both `max_epochs` and `max_steps` are not specified, 
                defaults to `max_epochs = 1000`. To enable infinite training, set `max_epochs` = -1.
                Default: `nothing`.

- **max\_steps**: Stop training after this number of steps. 
               Disabled by default (`-1`). 
               If `max_steps = -1` and `max_epochs = nothing`, will default to `max_epochs = 1000`. 
               To enable infinite training, set `max_epochs` to `-1`.
               Default: `-1`.

- **progress\_bar**: It `true`, shows a progress bar during training. 
                  Default: `true`.

# Examples

```julia
trainer = Trainer(max_epochs = 10, 
                  default_root_dir = @__DIR__,
                  accelerator = :cpu,
                  checkpointer = true,
                  logger = true,
                  )

Tsunami.fit!(model, trainer; train_dataloader, val_dataloader)
```
"""
@kwdef mutable struct Trainer
    accelerator::Symbol = :auto
    default_root_dir::String = pwd()
    checkpointer::Bool = true
    devices::Union{Int, Nothing} = nothing
    fast_dev_run::Bool = false
    logger::Bool = true
    max_epochs::Union{Int, Nothing} = nothing
    max_steps::Int = -1
    progress_bar::Bool = true
    val_every_n_epoch::Int = 1
end


"""
    fit!(model::FluxModule, trainer::Trainer; train_dataloader, val_dataloader = nothing, ckpt_path = nothing)

Train a `model` using the [`Trainer`](@ref) configuration.
If `ckpt_path` is not `nothing`, training is resumed from the checkpoint.

# Arguments

- **model**: A Flux model subtyping [`FluxModule`](@ref).
- **trainer**: A [`Trainer`](@ref) object storing the configuration options for `fit!`.
- **train\_dataloader**: A `DataLoader` used for training. Required dargument.
- **val\_dataloader**: A `DataLoader` used for validation. Default: `nothing`.
- **ckpt\_path**: Path of the checkpoint from which training is resumed. Default: `nothing`.

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
        train_dataloader::DataLoader,
        val_dataloader = nothing,
    )

    input_model = model

    tsunami_dir = joinpath(trainer.default_root_dir, "tsunami_logs")
    run_dir = dir_with_version(joinpath(tsunami_dir, "run"))
    checkpoints_dir = joinpath(run_dir, "checkpoints")

    checkpointer = trainer.checkpointer ? Checkpointer(checkpoints_dir) : nothing 
    device = select_device(trainer.accelerator, trainer.devices)
    logger = trainer.logger ? TBLogger(run_dir, tb_append, step_increment=0) : nothing
    logger_infotime = 50
    max_steps, max_epochs = compute_max_steps_and_epochs(trainer.max_steps, trainer.max_epochs)
    val_every_n_epoch = trainer.val_every_n_epoch
    
    if trainer.fast_dev_run
        max_epochs = 1
        max_steps = 1
        val_every_n_epoch = 1

        check_fluxmodule(model)
        check_forward(model, first(train_dataloader))
        if val_dataloader !== nothing
            check_forward(model, first(val_dataloader))
        end
    end

    training_step_outs = NamedTuple[]
    training_step_out_avg = Stats()

    nsteps = 0
    if ckpt_path !== nothing
        ckpt = load_checkpoint(ckpt_path)
        model = ckpt.model
        start_epoch = ckpt.epoch + 1
        opt = ckpt.opt
        nsteps = ckpt.step
    else
        opt = configure_optimisers(model)
        start_epoch = 1
    end
    model = model |> device
    opt = opt |> device

    for epoch in start_epoch:max_epochs
        progressbar = Progress(length(train_dataloader); desc="Train Epoch $epoch: ", 
            showspeed=true, enabled = trainer.progress_bar, color=:yellow)
		
        # SINGLE EPOCH TRAINING LOOP
        for (batch_idx, batch) in enumerate(train_dataloader)
            nsteps += 1
            batch = batch |> device

            grads = Zygote.gradient(model) do model
                training_step_out = training_step(model, batch, batch_idx)
                loss, training_step_out = unwrap_loss(training_step_out)
                Zygote.ignore_derivatives() do
                    push!(training_step_outs, training_step_out)
                    OnlineStats.fit!(training_step_out_avg, training_step_out)
                end
                return loss
            end
            opt, model = Optimisers.update!(opt, model, grads[1])

            ProgressMeter.next!(progressbar,
                showvalues = process_out_for_progress_bar(last(training_step_outs), training_step_out_avg),
                valuecolor=:yellow)
            
            if logger !== nothing && nsteps % logger_infotime == 0
                TensorBoardLogger.set_step!(logger, nsteps)
                with_logger(logger) do
                    @info "Training" epoch last(training_step_outs)...
                end
            end

            nsteps == max_steps && break
        end
        training_epoch_out = training_epoch_end(model, training_step_outs)
        if checkpointer !== nothing
            checkpointer(model, opt; epoch, step=nsteps)
        end
        
        # VALIDATION LOOP
        if  (val_dataloader !== nothing && 
            val_every_n_epoch !== nothing && 
            epoch % val_every_n_epoch == 0)

            valprogressbar = Progress(length(val_dataloader); desc="Validation: ", showspeed=true, enabled=false) # TODO doesn't work
            validation_step_outs = NamedTuple[]
            for (batch_idx, batch) in enumerate(val_dataloader)
                batch = batch |> device
                validation_step_out = validation_step(model, batch, batch_idx)
                push!(validation_step_outs, validation_step_out)
                ProgressMeter.next!(valprogressbar)
            end
            validation_epoch_out = validation_epoch_end(model, validation_step_outs)
            log_validation(logger, nsteps, validation_epoch_out)
         end

         (nsteps == max_steps || epoch == max_epochs) && break
    end

    model = model |> cpu
    if model !== input_model
        copy!(input_model, model)
    end
    return nothing
end

unwrap_loss(training_step_out::Number) = training_step_out, (; loss=training_step_out)
unwrap_loss(training_step_out::NamedTuple) = training_step_out.loss, training_step_out

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

function log_validation(tblogger, nsteps::Int, validation_epoch_out::NamedTuple)
    if tblogger !== nothing
        TensorBoardLogger.set_step!(tblogger, nsteps)
        with_logger(tblogger) do
            @info "Validation" validation_epoch_out...
        end
    end

    #TODO customize with https://github.com/JuliaLogging/MiniLoggers.jl
    f(k, v) = "$(k) = $(roundval(v))"
    @info "Validation: $(join([f(k, v) for (k, v) in pairs(validation_epoch_out)], ", "))"
end
