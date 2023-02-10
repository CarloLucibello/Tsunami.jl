using Test
using Tsunami, Flux, Functors, Zygote

include("test_modules.jl")

@testset "FluxModule" begin
   include("fluxmodule.jl") 
end

@testset "Trainer" begin
   include("trainer.jl") 
end