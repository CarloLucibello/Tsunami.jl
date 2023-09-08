

FOIL_CONSTRUCTOR_ARGS ="""
- **accelerator**: Supports passing different accelerator types:
    - `:auto` (default): Automatically select a gpu if available, otherwise fallback on cpu.
    - `:gpu`: Like `:auto`, but will throw an error if no gpu is available.
    - `:cpu`: Force using the cpu.
    - `:cuda`: Train on Nvidia gpus using CUDA.jl.
    - `:amdgpu`: Train on AMD gpus using AMDGPU.jl.
    - `:metal`: Train on Apple Silicon hardware using Metal.jl.
    See also the `devices` option.

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

const accelerator_symbol_to_string = Dict(
    :cpu => "CPU",
    :cuda => "CUDA",
    :amdgpu => "AMDGPU",
    :amd => "AMDGPU",
    :metal => "Metal"
)

function select_device(accelerator::Symbol, idx_devices)
    if accelerator ∈ (:cpu, :cuda, :amdgpu, :amd, :metal)
        idx = flux_device_idx(idx_devices)
        device = Flux.get_device(accelerator_symbol_to_string[accelerator], idx)
    elseif accelerator ∈ (:gpu, :auto)
        device = Flux.get_device()
        if device === :gpu && device isa Flux.FluxCPUDevice
            @error "No GPU device found. Try importing a gpu package (`using CUDA`, `using AMDGPU`, or `using Metal`).
                    As an alternative, select `:auto` or `:cpu` as accelerator."
        end
    else
        throw(ArgumentError("accelerator must be one of :cpu, :gpu, :auto, :cuda, :amdgpu, :metal"))
    end
    return device
end

flux_device_idx(idx_devices::Nothing) = 0

function flux_device_idx(idx_devices::Int)
    @assert idx_devices == 1 "Only one device is supported"
    return 0
end

function flux_device_idx(idx_devices::Union{Vector{Int}, Tuple})
    @assert length(idx_devices) == 1 "Only one device is supported"
    return idx_devices[1]
end

to_device(foil::Foil) =  foil.device
to_device(foil::Foil, x) =  x |> foil.device
to_precision(foil::Foil) = foil.fprec
to_precision(foil::Foil, x) = x |> foil.fprec

is_using_gpu(foil::Foil) = !(foil.device isa Flux.FluxCPUDevice)

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
