

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

- **precision**: Supports passing different precision types `(:bf16, :f16, :f32, :f64)`, 
    where `:bf16` is BFloat16, `:f16` is Float16, `:f32` is Float32, and `:f64` is Float64.
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

    # These functions convert floating point arrays but preserve integer arrays
    fprec = if precision == :f16
                f16
            elseif precision == :bf16
                bf16
            elseif precision == :f32
                f32
            elseif precision == :f64
                f64
            else
                throw(ArgumentError("precision must be one of :bf16, :f16, :f32, :f64"))
            end

    return Foil(device, fprec, precision) 
end

function Base.show(io::IO, foil::Foil)
    print(io, "Foil($(foil.device), $(foil.precision))")
end

# TODO: remove this when https://github.com/FluxML/Flux.jl/issues/2573
bf16(x) = Flux._paramtype(BFloat16, x)

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

"""
    setup(foil::Foil, model, [optimisers])
    
Setup the model and optimisers for training sending them to the device and setting the precision.
This function is called internally by [`Tsunami.fit!`](@ref).

See also [`Foil`](@ref).
"""
function setup(foil::Foil, model, optimisers)
    model = setup(foil, model)
    if !(optimisers isa Optimisers.AbstractRule)
        # Assume it is an opt_state. This can happen for two reasons:
        # 1. In previous version of Tsunami, configure_optimisers returned an opt_state
        # 2. When loadding a checkpoint, the optimisers state is restored directly
        opt_state = optimisers
    else
        opt_state = Optimisers.setup(optimisers, model)
    end
    opt_state = opt_state |> to_precision(foil) |> to_device(foil)
    return model, opt_state
end

function setup(foil::Foil, model)
    return model |> to_precision(foil) |> to_device(foil)
end

function setup_iterator(foil::Foil, iterator)
    iterator = Iterators.map(to_precision(foil), iterator)
    return MLDataDevices.DeviceIterator(foil.device, iterator)
end

setup_iterator(::Foil, iterator::Nothing) = nothing
