#### v0.2 DEPRECATIONS #####

function fit(ckpt_path::AbstractString, model::FluxModule, trainer::Trainer, args...; kws...)
    @warn "Tsunami.fit is deprecated. Use Tsunami.fit! instead."
    newmodel = deepcopy(model)
    fit_state = fit!(newmodel, trainer, args...; ckpt_path, kws...)
    return newmodel, fit_state
end

function fit(model::FluxModule, trainer::Trainer, args...; kws...)
    @warn "Tsunami.fit is deprecated. Use Tsunami.fit! instead."
    newmodel = deepcopy(model)
    fit_state = fit!(newmodel, trainer, args...; kws...)
    return newmodel, fit_state
end

