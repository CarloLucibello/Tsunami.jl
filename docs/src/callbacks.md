# Callbacks

Callbacks are functions that are called at certain points in the training process. They are useful for logging, early stopping, and other tasks. Callbacks are passed to the [`fit!`](@ref) function as a list of callbacks.


## Available Callbacks

### Checkpoints 

```@docs
Tsunami.Checkpointer
Tsunami.load_checkpoint
```

## Writing Custom Callbacks

See the 
