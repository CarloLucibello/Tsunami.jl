# Tsunami Release Notes

## v0.2 - 2025-01-03

**Breaking changes:**
- Devices indexing now starts from 1, as in `MLDataDevices.gpu_device`.

**Highlights:**
- Updated to Flux v0.16.
- Models (i.e. subtypes of `FluxModule`) are now not required to be mutable.
- `Tsunami.fit` is deprecated in favor of `Tsunami.fit!`.
- Added Trainer option to use `Enzyme` for automatic differentiation.
- Improved test infrastructure.
- Improved documentation.
