```@meta
CollapsedDocStrings = true
```

# Trainer

The [`Trainer`](@ref) struct is the main entry point for training a model. It is responsible for managing the training loop, logging, and checkpointing. It is also responsible for managing the [`FitState`](@ref Tsunami.FitState) struct, which contains the state of the training loop. 

Pass a model (a [`FluxModule`](@ref)) and a trainer to the function [`Tsunami.fit!`](@ref) to train the model.
After training, you can use the [`Tsunami.test`](@ref) function to test the model on a test dataset.

## References

```@docs
Tsunami.Trainer
Tsunami.fit!
Tsunami.FitState
Tsunami.test
Tsunami.validate
```

