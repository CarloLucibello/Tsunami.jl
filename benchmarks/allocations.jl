using Pkg
Pkg.activate(@__DIR__)
using Flux
using Statistics, Random
using GPUArrays

using CUDA

function train_mlp()
    d_in = 1280
    d_out = 1280
    batch_size = 1280
    num_iters = 100
    device_id = 2

    CUDA.device!(device_id-1)
    device = gpu_device(device_id; force=true)
    # device = cpu_device() # make sure it works on CPU too
    cache = GPUArrays.AllocCache()

    
    model = Dense(d_in => d_out) |> device
    x = randn(Float32, d_in, batch_size) |> device
    for iter in 1:num_iters
        GPUArrays.@cached cache begin
            ŷ = model(x)
        end
        @info iter
        CUDA.pool_status()
    end
    
    @info("Freeing cache...")
    CUDA.pool_status()
    GPUArrays.unsafe_free!(cache)
    CUDA.pool_status()
end

train_mlp()

@info("Reclaiming unused memory...")
CUDA.pool_status()
@time CUDA.reclaim()
CUDA.pool_status()