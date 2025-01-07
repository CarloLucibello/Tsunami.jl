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

function setup_batch(foil::Foil, batch)
    @warn "setup_batch is deprecated. Setup the dataloader with the Foil instead."
    return batch |> to_precision(foil) |> to_device(foil)
end


