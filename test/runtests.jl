using Test
using Tsunami, Flux, Functors, Zygote, Optimisers

include("test_utils.jl")
include("test_modules.jl")

@testset "FluxModule" begin
   include("fluxmodule.jl") 
end

@testset "Trainer" begin
   include("trainer.jl") 
end