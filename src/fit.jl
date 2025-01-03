"""
    fit!(model, trainer, train_dataloader, [val_dataloader]; [ckpt_path]) -> fit_state

Train `model` using the configuration given by `trainer`.
If `ckpt_path` is given, training is resumed from the checkpoint.

Return a [`FitState`](@ref) object.

# Arguments

- **model**: A Flux model subtyping [`FluxModule`](@ref).
- **trainer**: A [`Trainer`](@ref) object storing the configuration options for `fit!`.
- **train\\_dataloader**: An iterator over the training dataset, typically a `Flux.DataLoader`.
- **val\\_dataloader**: An iterator over the validation dataset, typically a `Flux.DataLoader`. Default: `nothing`.
- **ckpt\\_path**: Path of the checkpoint from which training is resumed. Default: `nothing`.

# Examples

```julia
model = ...
trainer = Trainer(max_epochs = 10)
fit_state = Tsunami.fit!(model, trainer, train_dataloader, val_dataloader)

# Resume training from checkpoint
trainer = Trainer(max_epochs = 20) # train for 10 more epochs
ckpt_path = joinpath(fit_state.run_dir, "checkpoints", "ckpt_last.jld2")
fit_state′ = Tsunami.fit!(model, trainer, train_dataloader, val_dataloader; ckpt_path)
```
"""
function fit!(model::FluxModule, trainer::Trainer, train_dataloader, val_dataloader=nothing; 
            ckpt_path = nothing)
    @assert get_device(model) isa CPUDevice

    if ckpt_path !== nothing
        ckpt = load_checkpoint(ckpt_path)
        Flux.loadmodel!(model, ckpt.model_state)
        fit_state = ckpt.fit_state
        lr_schedulers = ckpt.lr_schedulers
        optimisers = ckpt.optimisers
        start_epoch = fit_state.epoch + 1
    else # train from scratch
        fit_state = FitState()
        optimisers, lr_schedulers = configure_optimisers(model, trainer) |> process_out_configure_optimisers
        start_epoch = 1
        fit_state.step = 0
    end
    fit_state.epoch = start_epoch - 1
    fit_state.should_stop = false

    
    tsunami_dir = joinpath(trainer.default_root_dir, "tsunami_logs")
    run_dir = dir_with_version(joinpath(tsunami_dir, "run"))
    fit_state.run_dir = run_dir
    set_run_dir!(trainer.metalogger, run_dir)

    print_fit_initial_summary(model, trainer)

    # setup could create a copy on device, therefore we keep a reference to the original model
    model_orig = model
    model, optimisers = setup(trainer.foil, model, optimisers)
    if trainer.autodiff == :enzyme
        model = EnzymeCore.Duplicated(model)
    end
    
    trainer.fit_state = fit_state
    trainer.optimisers = optimisers
    trainer.lr_schedulers = lr_schedulers
 
    if trainer.fast_dev_run
        check_train_step(model, trainer, first(train_dataloader))
        if val_dataloader !== nothing
            check_val_step(model, trainer, first(val_dataloader))
        end
        return fit_state
    end

    val_loop(model, trainer, val_dataloader; progbar_keep=false, progbar_print_epoch=true)

    for epoch in start_epoch:trainer.max_epochs # TODO turn into while loop
        fit_state.epoch = epoch
        train_loop(model, trainer, train_dataloader, val_dataloader)
        fit_state.should_stop && break
    end

    Flux.loadmodel!(model_orig, Flux.state(model))
    return fit_state
end

val_loop(m::EnzymeCore.Duplicated, args...; kws...) = val_loop(m.val, args...; kws...)

function val_loop(model::FluxModule, trainer::Trainer, val_dataloader; progbar_offset = 0, 
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

function train_loop(model, trainer::Trainer, train_dataloader, val_dataloader)
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
        
        loss, pb = pullback_train_step(model, trainer, batch, batch_idx)
        hook(on_before_backprop, model, trainer, loss)
        grad = pb()
        ## Alternative directly computing the gradient
        # loss, grad = gradient_train_step(model, trainer, batch, batch_idx)

        hook(on_before_update, model, trainer, grad)

        update!(trainer.optimisers, model, grad)

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

update!(optimisers, m::EnzymeCore.Duplicated, grad) = update!(optimisers, m.val, grad)
update!(optimisers, m::FluxModule, grad) = Optimisers.update!(optimisers, m, grad)

function pullback_train_step(model::FluxModule, trainer::Trainer, batch, batch_idx::Int)
    loss, z_pb = Zygote.pullback(model) do model
        loss = train_step(model, trainer, batch, batch_idx)
        return loss
    end
    # zygote returns a Ref with immutable, so we need to unref it
    pb = () -> unref(z_pb(one(loss))[1])
    return loss, pb
end

function gradient_train_step(model::FluxModule, trainer::Trainer, batch, batch_idx::Int)
    loss, z_grad = Zygote.withgradient(model) do model
        loss = train_step(model, trainer, batch, batch_idx)
        return loss
    end
    return loss, unref(z_grad[1])
end

# TODO remove when Optimisers.jl is able to handle gradients with (nested) Refs
unref(x::Ref) = x[]
unref(x) = x

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
    model = setup(trainer.foil, model)
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

See also [`Tsunami.test`](@ref) and [`Tsunami.fit!`](@ref).
"""
function validate(model::FluxModule, trainer::Trainer, dataloader)
    model = setup(trainer.foil, model)
    return val_loop(model, trainer, dataloader; progbar_keep=true)
end
