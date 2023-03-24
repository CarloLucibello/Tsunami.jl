using Test
using Tsunami, Flux
using MLUtils, Functors, Zygote, Optimisers
using DataFrames

ENV["DATADEPS_ALWAYS_ACCEPT"] = true # for MLDatasets in examples

include("test_utils.jl")
include("test_modules.jl")

SilentTrainer = (args...; kws...) -> Trainer(args...; kws..., logger=false, checkpointer=false, progress_bar=false)

@testset "Stats" begin
   include("stats.jl")
end

@testset "FluxModule" begin
   include("fluxmodule.jl") 
end

@testset "Logging" begin
   include("logging.jl") 
end

@testset "Trainer" begin
   include("trainer.jl") 
end

@testset "Linear Regression" begin
   include("linear_regression.jl") 
end

@testset "Examples" begin
   include(joinpath(@__DIR__, "..", "examples/mlp_mnist.jl"))
end