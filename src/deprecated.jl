#### v0.2 DEPRECATIONS #####

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

