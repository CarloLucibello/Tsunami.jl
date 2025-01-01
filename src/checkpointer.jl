
"""
    Checkpointer(folder = nothing) <: AbstractCallback

An helper class for saving a [`FluxModule`](@ref) and the fit state.
The checkpoint is saved as a BSON file with the name `ckpt_epoch=X_step=Y.bson`.
A symbolic link to the last checkpoint is also created as `ckpt_last.bson`.

A `Checkpointer` is automatically created when `checkpointer = true` is passed to [`fit!`](@ref).

If `folder` is not specified, the checkpoints are saved in a folder named `checkpoints` in the run directory.

See also: [`load_checkpoint`](@ref).

# Examples
```julia
checkpointer = Checkpointer()
Tsunami.fit!(..., callbacks = [checkpointer])
```
"""
mutable struct Checkpointer <: AbstractCallback
    folder::Union{Nothing, String}
    last_ckpt::Union{Nothing, String}

    function Checkpointer(folder = nothing)
        return new(folder, nothing)
    end
end

function on_train_epoch_end(cp::Checkpointer, model::FluxModule, trainer)
    @unpack fit_state = trainer
    @unpack step, epoch, run_dir = fit_state
   if cp.folder !== nothing
        folder = cp.folder
    else
        folder = joinpath(run_dir, "checkpoints")
    end
    mkpath(folder)
    filename = "ckpt_epoch=$(epoch)_step=$(step).bson"
    filepath = joinpath(folder, filename)

    ckpt = (model_state = cpu(Flux.state(model)), 
            fit_state = fit_state,
            lr_schedulers = trainer.lr_schedulers,
            optimisers = cpu(trainer.optimisers))

    BSON.@save filepath ckpt=ckpt

    if cp.last_ckpt !== nothing
        rm(cp.last_ckpt)
    end
    cp.last_ckpt = filepath

    linklast = joinpath(folder, "ckpt_last.bson") 
    rm(linklast, force=true)
    symlink(filepath, linklast)
    
    return filepath
end

"""
    load_checkpoint(path)

Loads a checkpoint that was saved to `path`. 
Returns a namedtuple containing the model state, the fit state, the lr schedulers and the optimisers.

See also: [`Checkpointer`](@ref).

# Examples

```julia
ckpt = load_checkpoint("checkpoints/ckpt_last.bson")
model = MyModel(...)
Flux.loadmodel!(model, ckpt.model_state)
``` 
"""
function load_checkpoint(path)
    BSON.@load path ckpt
    return ckpt
end

