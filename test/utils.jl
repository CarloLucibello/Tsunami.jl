@testitem "Tsunami.seed!" begin
    using Random
    Tsunami.seed!(17)
    x = rand(10)
    Random.seed!(17)
    y = rand(10)
    @test x ≈ y
end

@testitem "Tsunami.seed! GPU" tags=[:gpu] begin
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

