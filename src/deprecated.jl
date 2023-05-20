function fit(ckpt_path::AbstractString, trainer::Trainer, args...; kws...)
    Base.depwarn("`fit(ckpt_path, trainer, ...)` is deprecated, use `fit(ckpt_path, model, trainer, ...; kws...)` instead.", :fit)
    ckpt = load_checkpoint(ckpth_path)
    model = ckpt.model 
    trainer.fit_state = ckpt.fit_state
    trainer.lr_schedulers = ckpt.lr_schedulers
    trainer.optimisers = ckpt.optimisers
    return fit(model, trainer, args...; kws..., _resuming_from_ckpt = true)
end
