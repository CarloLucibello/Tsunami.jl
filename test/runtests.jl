using Test
using Tsunami, Flux, Functors, Zygote, Optimisers

ENV["DATADEPS_ALWAYS_ACCEPT"] = true # for MLDatasets in examples

include("test_utils.jl")
include("test_modules.jl")

@testset "FluxModule" begin
   include("fluxmodule.jl") 
end

@testset "Trainer" begin
   include("trainer.jl") 
end

@testset "Examples" begin
   include(joinpath(@__DIR__, "..", "examples/mlp_mnist.jl"))
end