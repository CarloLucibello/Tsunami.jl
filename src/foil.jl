FOIL_CONSTRUCTOR_ARGS ="""
- **accelerator**: Supports passing different accelerator types `(:cpu, :gpu,  :auto)`.
    Use `:auto` to automatically select a gpu if available.
    See also the `devices` option.
    Default: `:auto`.

- **devices**: Pass an integer `n` to train on `n` devices (only `1` supported at the moment),
    or a list of devices ids to train on specific devices (e.g. `[2]` to train on gpu with idx 2).
    If `nothing`, will use all available devices. 
    Default: `nothing`.

- **precision**: Supports passing different precision types `(:f16, :f32, :f64)`.
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
    fprec
    precision::Symbol
end

function Foil(;
        accelerator::Symbol = :auto,
        precision::Symbol = :f32,
        devices = nothing
    )
    
    device = select_device(accelerator, devices)

    fprec = if precision == :f16
                f16
            elseif precision == :f32
                f32
            elseif precision == :f64
                f64
            else
                throw(ArgumentError("precision must be one of :f16, :f32, :f64"))
            end

    Foil(device, fprec, precision) 
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

to_device(foil::Foil) =  foil.device
to_device(foil::Foil, x) =  x |> foil.device
to_precision(foil::Foil) = foil.fprec
to_precision(foil::Foil, x) = x |> foil.fprec

function setup(foil::Foil, model, optimisers)
    model = model |> to_precision(foil) |> to_device(foil)
    optimisers = optimisers |> to_precision(foil) |> to_device(foil)
    return model, optimisers
end

function setup_batch(foil::Foil, batch)
    return batch |> to_precision(foil) |> to_device(foil)
end

function Zygote.gradient(f, x, foil::Foil)
   return Zygote.gradient(f, x)[1] 
end

function Zygote.withgradient(f, x, foil::Foil)
    fx, gs = Zygote.withgradient(f, x)
    return fx, gs[1] 
 end

 function Zygote.pullback(f, x, foil::Foil)
    fx, pb = Zygote.pullback(f, x)
    return fx, () -> unref(pb(one(fx))[1]) # zygote returns a Ref with immutable, so we need to unref it
 end
 
unref(x::Ref) = x[]
unref(x) = x
