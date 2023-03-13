abstract type AbstractCallback end

function on_train_epoch_end(c::AbstractCallback, model::FluxModule, trainer::Trainer)
    return nothing
end

function on_validation_epoch_end(c::AbstractCallback,  model::FluxModule, trainer::Trainer)
    return nothing
end




