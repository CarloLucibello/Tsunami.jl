# Callbacks

Callbacks are functions that are called at certain points in the training process. They are useful for logging, early stopping, and other tasks. 

Callbacks are passed to the [`Tsunami.fit!`](@ref) function: 

```julia
callback1 = Checkpointer(...)
fit(..., callbacks = [callback1, ...])
```

Callback implement their functionalities thanks to the hooks described in the [Hooks](@ref) section of the documentation.

## Available Callbacks

### Checkpoints 

Callbacks for saving and loading the model and optimizer state.

```@docs
Tsunami.Checkpointer
Tsunami.load_checkpoint
```

## Writing Custom Callbacks

See the implementation of [`Checkpointer`](@ref) and the 
[Hooks](@ref) section of the documentation for more information
on how to write custom callbacks.
