# 
# 
### pytorch lightning trainer
# """
#     Trainer(; kwargs...)

# Customize every aspect of training via flags.

# # Args:

#     accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto")
#         as well as custom accelerator instances.

#     accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.
#         Default: ``None``.

#     auto_lr_find: If set to True, will make trainer.tune() run a learning rate finder,
#         trying to optimize initial learning for faster convergence. trainer.tune() method will
#         set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
#         To use a different key set a string instead of True with the key name.
#         Default: ``False``.

#     auto_scale_batch_size: If set to True, will `initially` run a batch size
#         finder trying to find the largest batch size that fits into memory.
#         The result will be stored in self.batch_size in the LightningModule
#         or LightningDataModule depending on your setup.
#         Additionally, can be set to either `power` that estimates the batch size through
#         a power search or `binsearch` that estimates the batch size through a binary search.
#         Default: ``False``.

#     benchmark: The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
#         The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
#         (``False`` if not manually set). If :paramref:`~pytorch_lightning.trainer.Trainer.deterministic` is set
#         to ``True``, this will default to ``False``. Override to manually set a different value.
#         Default: ``None``.

#     callbacks: Add a callback or list of callbacks.
#         Default: ``None``.

#     check_val_every_n_epoch: Perform a validation loop every after every `N` training epochs. If ``None``,
#         validation will be done solely based on the number of training batches, requiring ``val_check_interval``
#         to be an integer value.
#         Default: ``1``.

#     detect_anomaly: Enable anomaly detection for the autograd engine.
#         Default: ``False``.

#     deterministic: If ``True``, sets whether PyTorch operations must use deterministic algorithms.
#         Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
#         that don't support deterministic mode (requires PyTorch 1.11+). If not set, defaults to ``False``.
#         Default: ``None``.

#     devices: Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
#         based on the accelerator type.

#     fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
#         of train, val and test to find any bugs (ie: a sort of unit test).
#         Default: ``False``.

#     gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
#         gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
#         Default: ``None``.

#     gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
#         to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
#         be set to ``"norm"``.

#     limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches).
#         Default: ``1.0``.

#     limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches).
#         Default: ``1.0``.

#     limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches).
#         Default: ``1.0``.

#     limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches).
#         Default: ``1.0``.

#     logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
#         the default ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``.
#         ``False`` will disable logging. If multiple loggers are provided, local files
#         (checkpoints, profiler traces, etc.) are saved in the ``log_dir`` of he first logger.
#         Default: ``True``.

#     log_every_n_steps: How often to log within steps.
#         Default: ``50``.

#     enable_progress_bar: Whether to enable to progress bar by default.
#         Default: ``True``.

#     profiler: To profile individual steps during training and assist in identifying bottlenecks.
#         Default: ``None``.

#     overfit_batches: Overfit a fraction of training/validation data (float) or a set number of batches (int).
#         Default: ``0.0``.

#     plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
#         Default: ``None``.

#     precision: Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16).
#         Can be used on CPU, GPU, TPUs, HPUs or IPUs.
#         Default: ``32``.

#     max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
#         If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
#         To enable infinite training, set ``max_epochs = -1``.

#     min_epochs: Force training for at least these many epochs. Disabled by default (None).

#     max_steps: Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
#         and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
#         ``max_epochs`` to ``-1``.

#     min_steps: Force training for at least these number of steps. Disabled by default (``None``).

#     max_time: Stop training after this amount of time has passed. Disabled by default (``None``).
#         The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
#         :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
#         :class:`datetime.timedelta`.

#     num_nodes: Number of GPU nodes for distributed training.
#         Default: ``1``.

#     num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
#         Set it to `-1` to run all batches in all validation dataloaders.
#         Default: ``2``.

#     reload_dataloaders_every_n_epochs: Set to a non-negative integer to reload dataloaders every n epochs.
#         Default: ``0``.

#     replace_sampler_ddp: Explicitly enables or disables sampler replacement. If not specified this
#         will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
#         train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
#         you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.

#     strategy: Supports different training strategies with aliases
#         as well custom strategies.
#         Default: ``None``.

#     sync_batchnorm: Synchronize batch norm layers between process groups/whole world.
#         Default: ``False``.

#     track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm. If using
#         Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.
#         Default: ``-1``.

#     val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
#         after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
#         batches. An ``int`` value can only be higher than the number of training batches when
#         ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
#         across epochs or during iteration-based training.
#         Default: ``1.0``.

#     enable_model_summary: Whether to enable model summarization by default.
#         Default: ``True``.

#     move_metrics_to_cpu: Whether to force internal logged metrics to be moved to cpu.
#         This can save some gpu memory, but can make training slower. Use with attention.
#         Default: ``False``.

#     multiple_trainloader_mode: How to loop over the datasets when there are multiple train loaders.
#         In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is traversed,
#         and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets
#         reload when reaching the minimum length of datasets.
#         Default: ``"max_size_cycle"``.

#     inference_mode: Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during
#         evaluation (``validate``/``test``/``predict``).
# """

"""
    Trainer(; kws...)

# Arguments

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
    check_val_every_n_epoch::Union{Int, Nothing} = 1
    default_root_dir::String = pwd()
    enable_checkpointing::Bool = true
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
    
    checkpointer = trainer.enable_checkpointing ? Checkpointer(trainer.default_root_dir) : nothing 
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

    for epoch in start_epoch:max_epochs
        progressbar = Progress(length(train_dataloader); desc="Train Epoch $epoch: ", 
            showspeed=true, enabled = trainer.progress_bar)
		
        for (batch_idx, batch) in enumerate(train_dataloader)
            grads = Zygote.gradient(model) do model
                training_step_out = training_step(model, batch, batch_idx)
                loss, training_step_out = unwrap_loss(training_step_out)
                Zygote.ignore_derivatives() do
                    push!(training_step_outs, training_step_out)
                    OnlineStats.fit!(training_step_out_avg, training_step_out)
                end
                return loss
            end
            Optimisers.update!(opt, model, grads[1])

            nsteps += 1
			ProgressMeter.next!(progressbar,
                showvalues = process_out_for_progress_bar(last(training_step_outs), training_step_out_avg),
                valuecolor=:green)

            nsteps == max_steps && break
        end
        training_epoch_out = training_epoch_end(model, training_step_outs)
        checkpointer(model, opt; epoch)
        
        if  (val_dataloader !== nothing && 
            trainer.check_val_every_n_epoch !== nothing && 
            epoch % trainer.check_val_every_n_epoch == 0)

            valprogressbar = Progress(length(val_dataloader); desc="Validation: ", showspeed=true, enabled = false) # TODO doesn't work
		
            validation_step_outs = NamedTuple[]
            for (batch_idx, batch) in enumerate(val_dataloader)
                validation_step_out = validation_step(model, batch, batch_idx)
                push!(validation_step_outs, validation_step_out)
                ProgressMeter.next!(valprogressbar)
            end
            validation_epoch_out = validation_epoch_end(model, validation_step_outs)
            print_validation_epoch_end(validation_epoch_out)
         end

         (nsteps == max_steps || epoch == max_epochs) && break
    end
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

function process_out_for_progress_bar(out::NamedTuple, s::Stats)
    f(k, v) = "$(round4(v)) (last)  $(round4(OnlineStats.value(s[k]))) (expavg)"
    return [(k, f(k, v)) for (k, v) in pairs(out)]
end

function print_validation_epoch_end(out::NamedTuple)
    f(k, v) = "$(k) = $(round4(v))"
    @info "Validation: $(join([f(k, v) for (k, v) in pairs(out)], ", "))"
end

