@testset "Foil constructor" begin
    foil = Foil(accelerator=:cpu, precision=:f32, devices=nothing)
    @test foil isa Foil
end

@testset "Tsunami.setup" begin
    foil = Foil(accelerator=:cpu, precision=:f32, devices=nothing)
    model = Chain(Dense(28^2 => 512, relu), Dense(512 => 10))
    opt_state = Flux.setup(AdamW(1e-3), model)
    model, opt_state = Tsunami.setup(foil, model, opt_state)
    @test model isa Chain
    @test opt_state.layers[1].weight isa Optimisers.Leaf{<:AdamW}
end
