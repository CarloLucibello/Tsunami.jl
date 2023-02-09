
"""
    Checkpointer(folder)

Saves a [`FluxModule`](@ref) to `folder` after every training epoch.
"""
mutable struct Checkpointer
    folder::String
    last_ckpt::Union{Nothing, String}

    function Checkpointer(folder::String)
        mkpath(folder)
        return new(folder, nothing)
    end
end

function (cp::Checkpointer)(model::FluxModule, opt; epoch, step, kws...)
    filename = "ckpt_epoch=$(epoch)_step=$(step).bson"
    filepath = joinpath(cp.folder, filename)
    BSON.@save filepath ckpt=(; model=cpu(model), opt=cpu(opt), epoch, step, kws...)
    
    if cp.last_ckpt !== nothing
        rm(cp.last_ckpt)
    end
    cp.last_ckpt = filepath
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

