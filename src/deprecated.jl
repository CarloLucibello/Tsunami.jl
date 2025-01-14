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

##### v0.3 DEPRECATIONS #####
#TODO deprecate properly
on_train_batch_end(model, trainer) = nothing
on_train_batch_end(cb, model, trainer) = nothing
on_val_batch_end(model, trainer) = nothing
on_val_batch_end(cb, model, trainer) = nothing
on_test_batch_end(model, trainer) = nothing
on_test_batch_end(cb, model, trainer) = nothing
on_before_backprop(model, trainer, loss) = nothing
on_before_backprop(cb, model, trainer, loss) = nothing


