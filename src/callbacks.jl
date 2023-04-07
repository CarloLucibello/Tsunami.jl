abstract type AbstractCallback end

on_train_epoch_end(::AbstractCallback, ::FluxModule, ::Trainer) = nothing
on_val_epoch_end(::AbstractCallback, ::FluxModule, ::Trainer) = nothing
on_test_epoch_end(::AbstractCallback, ::FluxModule, ::Trainer) = nothing
