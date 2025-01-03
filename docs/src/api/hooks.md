```@meta
CollapsedDocStrings = true
```

# Hooks 

Hooks are a way to extend the functionality of Tsunami. They are a way to inject custom code into the FluxModule or
into a Callback at various points in the training, testing, and validation loops.

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

    for batch in train_dataloader
        on_train_batch_start()
        batch = transfer_batch_to_device(batch)
        loss, pb = pullback(m -> train_step(m, batch),  model)
        on_before_backprop()
        grad = pb()
        on_before_update()
        update!(opt_state, model, grad)
        on_train_batch_end()
        if should_check_val
            val_loop()
        end
    end
    on_train_epoch_end()
end

function val_loop()
    on_val_epoch_start()
    for batch in val_dataloader
        on_val_batch_start()
        batch = transfer_batch_to_device(batch)
        val_step(batch)
        on_val_batch_end()
    end
    on_val_epoch_end()
end
```

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
