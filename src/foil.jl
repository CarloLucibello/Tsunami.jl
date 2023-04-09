FOIL_CONSTRUCTOR_ARGS ="""
- **accelerator**: Supports passing different accelerator types `(:cpu, :gpu,  :auto)`.
`:auto` will automatically select a gpu if available.
See also the `devices` option.
    Default: `:auto`.

- **devices**: Pass an integer `n` to train on `n` devices, 
or a list of devices ids to train on specific devices.
If `nothing`, will use all available devices. 
Default: `nothing`.

- **precision**: Supports passing different precision types `(:f32, :f64)`.
    Default: `:f32`.
"""

"""
    Foil(; kws...)

A type that takes care of the acceleration of the training process.

# Constructor Arguments

$FOIL_CONSTRUCTOR_ARGS
"""
mutable struct Foil
    device
    precision
end

function Foil(;
        accelerator::Symbol = :auto,
        precision::Symbol = :f32,
        devices::Union{Int, Nothing} = nothing
    )
    
    device = select_device(accelerator, devices)
    Foil(device, precision) 
end


function select_device(accelerator::Symbol, devices)
    if accelerator == :auto
        if CUDA.functional()
            return select_cuda_device(devices)
        else
            return cpu
        end
    elseif accelerator == :cpu
        return cpu
    elseif accelerator == :gpu
        if !CUDA.functional()
            @warn "CUDA is not available"
            return cpu
        else
            return select_cuda_device(devices)
        end
    else
        throw(ArgumentError("accelerator must be one of :auto, :cpu, :gpu"))
    end
end

select_cuda_device(devices::Nothing) = gpu

function select_cuda_device(devices::Int)
    @assert devices == 1 "Only one device is supported"
    return gpu
end

function select_cuda_device(devices::Union{Vector{Int}, Tuple})
    @assert length(devices) == 1 "Only one device is supported"
    CUDA.device!(devices[1])
    return gpu
end
 
function is_using_cuda(foil::Foil)
    cuda_available = CUDA.functional()
    return cuda_available && foil.device === gpu
end

function to_device(foil::Foil, x)
    return x |> foil.device
end

function setup(foil::Foil, model::FluxModule, optimisers)
    model = to_device(foil, model)
    optimisers = to_device(foil, optimisers)
    return model, optimisers
end
