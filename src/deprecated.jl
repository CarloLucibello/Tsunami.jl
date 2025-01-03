#### v0.2 DEPRECATIONS #####

function fit(ckpt_path::AbstractString, model::FluxModule, trainer::Trainer, args...; kws...)
    @warn "Tsunami.fit is deprecated. Use Tsunami.fit! instead."
    newmodel = deepcopy(model)
    fit!(newmodel, trainer, args...; ckpt_path, kws...)
    return newmodel, trainer.fit_state
end

function fit(model::FluxModule, trainer::Trainer, args...; kws...)
    @warn "Tsunami.fit is deprecated. Use Tsunami.fit! instead."
    newmodel = deepcopy(model)
    fit!(newmodel, trainer, args...; kws...)
    return newmodel, trainer.fit_state
end

