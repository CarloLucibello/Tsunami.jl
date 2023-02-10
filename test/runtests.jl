using Test
using Tsunami, Flux, Functors, Zygote

@testset "FluxModule" begin
   include("fluxmodule.jl") 
end

@testset "Trainer" begin
   include("trainer.jl") 
end