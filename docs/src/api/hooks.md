```@meta
CollapsedDocStrings = true
```

# Hooks 

Hooks are a way to extend the functionality of Tsunami. They are a way to inject custom code into the FluxModule or into a Callback at various points in the training, testing, and validation loops.

At a high level, and omitting function imputs and outputs, a simplified version of the [`Tsunami.fit!`](@ref) method looks like this:

```julia
function fit!()
    configure_optimizers()
    
    for epoch in epochs
        train_loop()
    end
end

function train_loop()
    on_train_epoch_start()
    set_learning_rate(lr_scheduler, epoch)

    for (batch, batch_idx) in enumerate(train_dataloader)
        batch = transfer_batch_to_device(batch)
        on_train_batch_start(batch, batch_idx)
        out, grad = out_and_gradient(train_step, model, trainer, batch, batch_idx)
        on_before_update(out, grad)
        update!(opt_state, model, grad)
        on_train_batch_end(out)
        if should_check_val
            val_loop()
        end
    end
    on_train_epoch_end()
end

function val_loop()
    on_val_epoch_start()
    for (batch, batch_idx) in val_dataloader
        on_val_batch_start()
        batch = transfer_batch_to_device(batch)
        out = val_step(model, trainer, batch, batch_idx)
        on_val_batch_end(out)
    end
    on_val_epoch_end()
end
```

Each `on_something` hook takes as input the model and the trainer.

## Hooks API

```@docs
Tsunami.on_before_backprop
Tsunami.on_before_update
Tsunami.on_train_batch_start
Tsunami.on_train_batch_end
Tsunami.on_train_epoch_start
Tsunami.on_train_epoch_end
Tsunami.on_test_batch_start
Tsunami.on_test_batch_end
Tsunami.on_test_epoch_start
Tsunami.on_test_epoch_end
Tsunami.on_val_batch_start
Tsunami.on_val_batch_end
Tsunami.on_val_epoch_start
Tsunami.on_val_epoch_end
```
