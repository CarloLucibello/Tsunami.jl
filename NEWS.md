# Tsunami Release Notes

## v0.3.1 

- Implement caching strategy from [GPUArrays.jl](https://juliagpu.github.io/GPUArrays.jl/dev/interface/#Caching-Allocator) to reduce memory allocations on GPU during training, validation, and testing.

## v0.3.0 

**Breaking changes:**
- `fit!` returns `nothing` instead of a `FitState` object. The `FitState` object can be accessed via `trainer.fit_state`.
- `on_before_pullback` has been removed. Use `on_train_batch_start` instead.
- `on_*_batch_start` now receives the batch on device.
- Some of the hooks now take more inputs.

**Highlights:**

- Now Tsunami uses `MLDataDevices.DeviceIterator` to wrap dataloaders for more efficient device memory management.

- `training_step`, `validation_step`, and `test_step` can now return a named tuple
  for flebility. One of the fields of the named tuple should be `loss` which is used to compute the loss value.

## v0.2.0 - 2025-01-03

**Breaking changes:**
- Devices indexing now starts from 1, as in `MLDataDevices.gpu_device`.

**Highlights:**
- Updated to Flux v0.16.
- Models (i.e. subtypes of `FluxModule`) are now not required to be mutable.
- `Tsunami.fit` is deprecated in favor of `Tsunami.fit!`.
- Added Trainer option to use `Enzyme` for automatic differentiation.
- Improved test infrastructure.
- Improved documentation.
