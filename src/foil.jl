

FOIL_CONSTRUCTOR_ARGS ="""
- **accelerator**: Supports passing different accelerator types:
    - `:auto` (default): Automatically select a gpu if available, otherwise fallback on cpu.
    - `:gpu`: Like `:auto`, but will throw an error if no gpu is available.
       In order for a gpu to be available, the corresponding package must be loaded (e.g. with `using CUDA`).
       The trigger packages are `CUDA.jl` for Nvidia GPUs, `AMDGPU.jl` for AMD GPUs, and `Metal.jl` for Apple Silicon.
    - `:cpu`: Force using the cpu.
    See also the `devices` option.

- **devices**: Pass an integer `n` to train on `n` devices (only `1` supported at the moment),
    or a list of devices ids to train on specific devices (e.g. `[2]` to train on gpu with idx 2).
    Ids indexing starts from `1`. If `nothing`, will use the default device 
    (see `MLDataDevices.gpu_device` documentation). 
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
struct Foil{D,F}
    device::D
    fprec::F
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

    return Foil(device, fprec, precision) 
end

function select_device(accelerator::Symbol, idx_devices)
    if accelerator == :cpu
        device = cpu_device()
    elseif accelerator ∈ (:gpu, :cuda, :amdgpu, :amd, :metal)
        if accelerator ∈ (:cuda, :amdgpu, :amd, :metal)
            @warn "The accelerator arguments :cuda, :amdgpu, :metal are deprecated. Use :gpu instead."
        end
        idx = flux_device_idx(idx_devices)
        device = gpu_device(idx, force=true)
    elseif accelerator == :auto
        idx = flux_device_idx(idx_devices)
        device = gpu_device(idx)
    else
        throw(ArgumentError("accelerator must be one of :cpu, :gpu, :auto"))
    end
    return device
end

flux_device_idx(idx_devices::Nothing) = nothing

function flux_device_idx(idx_devices::Int)
    @assert idx_devices == 1 "Only one device is supported"
    return 1
end

function flux_device_idx(idx_devices::Union{Vector{Int}, Tuple})
    @assert length(idx_devices) == 1 "Only one device is supported"
    return idx_devices[1]
end

to_device(foil::Foil) =  foil.device
to_device(foil::Foil, x) =  x |> foil.device
to_precision(foil::Foil) = foil.fprec
to_precision(foil::Foil, x) = x |> foil.fprec

is_using_gpu(foil::Foil) = !(foil.device isa CPUDevice)

function setup(foil::Foil, model)
    return model |> to_precision(foil) |> to_device(foil)
end

"""
    setup(foil::Foil, model, optimisers)

Setup the model and optimisers for training sending them to the device and setting the precision.
This function is called internally by [`Tsunami.fit!`](@ref).

See also [`Foil`](@ref).
"""
function setup(foil::Foil, model, optimisers)
    model = setup(foil, model)
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
 
# TODO remove when Optimisers.jl is able to handle gradients with (nested) Refs
unref(x::Ref) = x[]
unref(x) = x
