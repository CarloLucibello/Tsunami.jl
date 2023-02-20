# Hooks 

Hooks are a way to extend the functionality of Tsunami. They are a way to inject custom code into the FluxModule or
into a Callback at various points in the training, testing, and validation loops.

At a high level, and omitting function imputs and outputs, a simplified version of the `Tsunami.fit!` method looks like this:
```julia
# commented out lines are not yet implemented

function fit!()
    
    configure_optimizers()
    
    # on_train_start()
    for epoch in epochs
        fit_loop()
    # on_train_end()
end

function fit_loop()
    # on_train_epoch_start()

    for batch in train_dataloader
        # on_train_batch_start()

        # on_before_batch_transfer()
        batch = transfer_batch_to_device(batch)
        # on_after_batch_transfer()

        out = training_step()
        # training_step_end(out)

        # on_before_backward()
        compute_gradient()
        # on_after_backward()

        # on_before_optimizer_step()
        # configure_gradient_clipping()
        optimizer_step()

        # on_train_batch_end()

        if should_check_val
            val_loop()
        end
    end
    on_train_epoch_end()
end

function val_loop()
    # on_validation_epoch_start()

    for (batch_idx, batch) in enumerate(val_dataloader)
        # on_validation_batch_start(batch, batch_idx)

        # batch = on_before_batch_transfer(batch)
        batch = transfer_batch_to_device(batch)
        # batch = on_after_batch_transfer(batch)

        out = validation_step(batch, batch_idx)
        # out = validation_step_end(out)

        # on_validation_batch_end(batch, batch_idx)
    end
    on_validation_epoch_end()
end
```


## Hooks API

```@docs
Tsunami.on_training_epoch_end
Tsunami.on_validation_epoch_end
```
