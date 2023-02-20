
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
Tsunami.fit!(model, trainer, callbacks = [checkpointer])
```
"""
mutable struct Checkpointer <: AbstractCallback
    folder::Union{Nothing, String}
    last_ckpt::Union{Nothing, String}

    function Checkpointer(folder = nothing)
        return new(folder, nothing)
    end
end

function on_training_epoch_end(cp::Checkpointer, model::FluxModule, trainer::Trainer)
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
    BSON.@save filepath ckpt=(; model=cpu(model), fit_state=cpu(fit_state))

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
Returns a namedtuple of the model and the optimizer.

See also: [`Checkpointer`](@ref).
"""
function load_checkpoint(path)
    BSON.@load path ckpt
    return ckpt
end

