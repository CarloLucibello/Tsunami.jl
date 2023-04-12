# Guides 

## Selecting a backend

Tsunami supports both CPU and GPU devices. To select a device, use the `accelerator` and `devices` keyword arguments in the `Trainer` constructor.

```julia
trainer = Trainer(accelerator = :auto) # default, selects CPU or GPU depending on availability
trainer = Trainer(accelerator = :cuda) # selects cuda GPU
```

By default, Tsunami will use the first of the available GPUs and the CPU if no GPUs are present. 
To select a specific GPU, use the `devices` keyword argument:

```julia
trainer = Trainer(devices = [0])
```

## Gradient accumulation

TODO

## Gradient clipping

TODO

## Mixed precision training

TODO
