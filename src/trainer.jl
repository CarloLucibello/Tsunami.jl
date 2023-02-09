
"""
    Trainer(; kws...)

# Arguments

- `accelerator`: Supports passing different accelerator types `(:cpu, :gpu,  :auto)`.
                `:auto` will automatically select a gpu if available.
                See also the `devices` option.
                 Default: `:auto`.

- `check_val_every_n_epoch`: Perform a validation loop every after every N training epochs. 
                             TODO[If `nothing`, validation will be done solely based on the number of training batches, 
                             requiring `val_check_interval` to be an integer value.]
                             Default: `1`.
            
- `enable_checkpointing`: If `true`, enable checkpointing.
        TODO[It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
        callbacks.]
        Default: `true`.

- `default_root_dir`: Default path for logs and weights TODO[when no logger/ckpt_callback passed].
                      Default: `pwd()`.

- `devices`: Devices identificaiton number(s). Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
             based on the `accelerator` type.
             Default: `nothing`.

- `max_epochs`: Stop training once this number of epochs is reached. 
                Disabled by default (`nothing`). 
                If both `max_epochs` and `max_steps` are not specified, 
                defaults to `max_epochs = 1000`. To enable infinite training, set `max_epochs` = -1.
                Default: `nothing`.

- `max_steps`: Stop training after this number of steps. 
               Disabled by default (`-1`). 
               If `max_steps = -1` and `max_epochs = nothing`, will default to `max_epochs = 1000`. 
               To enable infinite training, set `max_epochs` to `-1`.
               Default: `-1`.

- `progress_bar`: It `true`, shows a progress bar during training. 
                  Default: `true`.

- `val_check_interval`: TODO[How often to check the validation set. 
                         Pass a float in the range [0.0, 1.0] to check after a fraction of the training epoch. 
                         Pass an int to check after a fixed number of training batches. 
                         An Int value can only be higher than the number of training batches when 
                         `check_val_every_n_epoch=None`, which validates after every N training batches across epochs or during iteration-based training. 
                         Default: `1.0`.]
"""
@kwdef mutable struct Trainer
    accelerator::Symbol = :auto
    check_val_every_n_epoch::Union{Int, Nothing} = 1
    default_root_dir::String = pwd()
    enable_checkpointing::Bool = true
    devices::Union{Int, Nothing} = nothing
    max_epochs::Union{Int, Nothing} = nothing
    max_steps::Int = -1
    progress_bar::Bool = true
end


"""
    fit!(model::Trainer, trainer::FluxModule; train_dataloader = nothing, val_dataloader = nothing, ckpt_path = nothing)

# Arguments

- `model`: A `FluxModule` object.
- `trainer`: A `Trainer` object.
- `train_dataloader`: A `DataLoader` used for training. Require argument.
- `val_dataloader`: A `DataLoader` used for validation. Default: `nothing`.
- `ckpt_path`: Path of the checkpoint from which training is resumed. Default: `nothing`.
"""
function fit!(
        model::FluxModule,
        trainer::Trainer;
        ckpt_path = nothing,
        train_dataloader::DataLoader,
        val_dataloader = nothing,
    )
    
    check_fluxmodule(model)
    checkpointer = trainer.enable_checkpointing ? Checkpointer(trainer.default_root_dir) : nothing 
    device = select_device(trainer.accelerator, trainer.devices)
    # @assert train_dataloader !== nothing "train_dataloaders must be specified"
    
    max_steps, max_epochs = compute_max_steps_and_epochs(trainer.max_steps, trainer.max_epochs)
    training_step_outs = NamedTuple[]
    training_step_out_avg = Stats()

    nsteps = 0
    if ckpt_path !== nothing
        ckpt = load_checkpoint(ckpt_path)
        copy!(model, ckpt.model)
        start_epoch = ckpt.epoch + 1
        opt = ckpt.opt
    else
        opt = configure_optimisers(model)
        start_epoch = 1
    end
    model = model |> device
    opt = opt |> device

    for epoch in start_epoch:max_epochs
        progressbar = Progress(length(train_dataloader); desc="Train Epoch $epoch: ", 
            showspeed=true, enabled = trainer.progress_bar)
		
        for (batch_idx, batch) in enumerate(train_dataloader)
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

            nsteps += 1
			ProgressMeter.next!(progressbar,
                showvalues = process_out_for_progress_bar(last(training_step_outs), training_step_out_avg),
                valuecolor=:green)

            nsteps == max_steps && break
        end
        training_epoch_out = training_epoch_end(model, training_step_outs)
        if checkpointer !== nothing
            checkpointer(model, opt; epoch, nsteps)
        end
        
        if  (val_dataloader !== nothing && 
            trainer.check_val_every_n_epoch !== nothing && 
            epoch % trainer.check_val_every_n_epoch == 0)

            valprogressbar = Progress(length(val_dataloader); desc="Validation: ", showspeed=true, enabled = false) # TODO doesn't work
		
            validation_step_outs = NamedTuple[]
            for (batch_idx, batch) in enumerate(val_dataloader)
                batch = batch |> device
                validation_step_out = validation_step(model, batch, batch_idx)
                push!(validation_step_outs, validation_step_out)
                ProgressMeter.next!(valprogressbar)
            end
            validation_epoch_out = validation_epoch_end(model, validation_step_outs)
            print_validation_epoch_end(validation_epoch_out)
         end

         (nsteps == max_steps || epoch == max_epochs) && break
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
    CUDA.device!(devices)
    return gpu
end

function process_out_for_progress_bar(out::NamedTuple, s::Stats)
    f(k, v) = "$(round4(v)) (last)  $(round4(OnlineStats.value(s[k]))) (expavg)"
    return [(k, f(k, v)) for (k, v) in pairs(out)]
end

function print_validation_epoch_end(out::NamedTuple)
    f(k, v) = "$(k) = $(round4(v))"
    @info "Validation: $(join([f(k, v) for (k, v) in pairs(out)], ", "))"
end

