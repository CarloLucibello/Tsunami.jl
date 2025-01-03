```@meta
CollapsedDocStrings = true
```

# [FluxModule](@id sec_fluxmodule)

The [`FluxModule`](@ref) abstract type is the entry point for defining custom models in Tsunami.
Subtypes of `FluxModule` are designed to be used with the [`Tsunami.fit!`](@ref) method, but can also be used independently.

```@docs
FluxModule
```
## Required methods

The following methods must be implemented for a subtype of `FluxModule` to be used with Tsunami.

```@docs
Tsunami.configure_optimisers
Tsunami.train_step
```

## Optional methods

The following methods have default implementations that can be overridden if necessary.
See also the [Hooks](@ref) section of the documentation for other methods that can be overridden.

```@docs
Tsunami.val_step
Tsunami.test_step
```
