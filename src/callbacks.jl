abstract type AbstractCallback end

function on_training_epoch_end(c::AbstractCallback, trainer::Trainer, model::FluxModule)
    return nothing
end

function on_validation_epoch_end(c::AbstractCallback, trainer::Trainer, model::FluxModule)
    return nothing
end



