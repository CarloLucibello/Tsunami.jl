@testset "GPU Tsunami.seed!" begin
    using Random
    using MLDataDevices
    dev = gpu_device(force=true)
    rng = default_device_rng(dev)
    Tsunami.seed!(17)
    x = rand(rng, 10) # this is broken on Metal (https://github.com/JuliaGPU/GPUArrays.jl/issues/578)
    Tsunami.seed!(17)
    y = rand(rng, 10)
    @test x ≈ y
end

@testset "GPU loadmodel! can load state on gpu" begin
    using Flux
    model_orig = Chain(Dense(10, 5, relu), Dense(5, 2))
    dev = gpu_device(force=true)
    model = model_orig |> dev
    model[1].weight .= 1f0
    Flux.loadmodel!(model_orig, Flux.state(model))
    @test model_orig[1].weight ≈ collect(model[1].weight)
end
