
"""
    Checkpointer(folder)

Saves a [`FluxModule`](@ref) to `folder` after every training epoch.
"""
struct Checkpointer
    folder
    function Checkpointer(folder)
        mkpath(folder)
        return new(folder)
    end
end

function (cp::Checkpointer)(model::FluxModule, opt; epoch)
    filename = "checkpoint_epoch=$(lpad(string(epoch), 4, '0')).bson"
    filepath = joinpath(cp.folder, filename)
    BSON.@save filepath model=cpu(model) opt=cpu(opt)
    
    # delete old checkpoints
    for path in glob("checkpoint_epoch=*.bson", cp.folder)
        path != filepath && rm(path)
    end
    return filepath
end

"""
    load_checkpoint(path)

Loads a checkpoint that was saved to `path`. 
Returns a namedtuple of the model and the optimizer.

See also: [`Checkpointer`](@ref).
"""
function load_checkpoint(path)
    BSON.@load path model opt
    return (; model, opt)
end
