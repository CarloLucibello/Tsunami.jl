abstract type AbstractFoilDevice <: Function end
abstract type AbstractFoilGPUDevice <: AbstractFoilDevice end

struct FoilCPUDevice <: AbstractFoilDevice end

Base.@kwdef struct FoilCUDADevice <: AbstractFoilGPUDevice 
    name::String = "CUDA"
    pkgid::PkgId = PkgId(UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
end

Base.@kwdef struct FoilAMDGPUDevice <: AbstractFoilGPUDevice 
    name::String = "AMDGPU"
    pkgid::PkgId = PkgId(UUID("21141c5a-9bdb-4563-92ae-f87d6854732e"), "AMDGPU")
end

Base.@kwdef struct FoilMetalDevice <: AbstractFoilGPUDevice 
    name::String = "Metal"
    pkgid::PkgId = PkgId(UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal")
end

(::FoilCPUDevice)(x) = Flux.cpu(x)
(::AbstractFoilGPUDevice)(x) = Flux.gpu(x)

function is_cuda_available()
    device = FoilCUDADevice()
    if !haskey(Base.loaded_modules, device.pkgid)
        return false
    else
        CUDA = Base.loaded_modules[device.pkgid]
        return CUDA.functional()
    end
end

function is_amdgpu_available()
    device = FoilAMDGPUDevice()
    if !haskey(Base.loaded_modules, device.pkgid)
        return false
    else
        AMDGPU = Base.loaded_modules[device.pkgid]
        return AMDGPU.functional()
    end
end

function is_metal_available()
    device = FoilMetalDevice()
    if !haskey(Base.loaded_modules, device.pkgid)
        return false
    else
        Metal = Base.loaded_modules[device.pkgid]
        return Metal.functional()
    end
end

get_cuda_module() = Base.loaded_modules[FoilCUDADevice().pkgid]
get_amdgpu_module() = Base.loaded_modules[FoilAMDGPUDevice().pkgid]
get_metal_module() = Base.loaded_modules[FoilMetalDevice().pkgid]
