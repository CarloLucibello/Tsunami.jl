abstract type AbstractCallback end

function on_training_epoch_end(c::AbstractCallback, model::FluxModule, trainer::Trainer)
    return nothing
end

function on_validation_epoch_end(c::AbstractCallback,  model::FluxModule, trainer::Trainer)
    return nothing
end




