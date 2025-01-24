```@meta
CollapsedDocStrings = true
```

# Callbacks

Callbacks are functions that are called at certain points in the training process. They are useful for logging, early stopping, and other tasks. 

Callbacks are passed to the [`Trainer`](@ref) constructor: 

```julia
callback1 = Checkpointer(...)
trainer = Trainer(..., callbacks = [callback1, ...])
```

Callback implement their functionalities thanks to the hooks described in the [Hooks](@ref) section of the documentation.

## Available Callbacks

A few callbacks are provided by Tsunami.

### Checkpoints 

Callbacks for saving and loading the model and optimizer state.

```@docs
Tsunami.Checkpointer
Tsunami.load_checkpoint
```

## Writing Custom Callbacks

Users can write their own callbacks by defining customs types and implementing the hooks they need. For example

```julia
struct MyCallback end

function Tsunami.on_train_epoch_end(cb::MyCallback, model, trainer)
    fit_state = trainer.fit_state # contains info about the training status
    # do something
end

trainer = Trainer(..., callbacks = [MyCallback()])
```

See the implementation of [`Checkpointer`](@ref) and the 
[Hooks](@ref) section of the documentation for more information
on how to write custom callbacks. Also, the [examples](https://github.com/CarloLucibello/Tsunami.jl/tree/main/examples) folder contains some examples of custom callbacks. 
