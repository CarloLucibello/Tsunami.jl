# Callbacks

Callbacks are functions that are called at certain points in the training process. They are useful for logging, early stopping, and other tasks. Callbacks are passed to the [`fit!`](@ref) function as a list,
and implement their functionality by implementing a subset of the [Hooks](@ref).

## Available Callbacks

### Checkpoints 

```@docs
Tsunami.Checkpointer
Tsunami.load_checkpoint
```

## Writing Custom Callbacks

See the implementation of [`Checkpointer`](@ref) and the 
[Hooks](@ref) section of the documentation for more information
on how to write custom callbacks.
