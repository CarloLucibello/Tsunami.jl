@testitem "Foil constructor" begin
    using MLDataDevices
    foil = Foil(accelerator=:cpu, precision=:f32, devices=nothing)
    @test foil.device isa CPUDevice
    @test foil isa Foil
end

@testitem "Tsunami.setup" setup=[TsunamiTest] begin
    using .TsunamiTest
    foil = Foil(accelerator=:cpu, precision=:f32, devices=nothing)
    model = Chain(Dense(28^2 => 512, relu), Dense(512 => 10))
    opt_state = Flux.setup(AdamW(1e-3), model)
    model, opt_state = Tsunami.setup(foil, model, opt_state)
    @test model isa Chain
    @test opt_state.layers[1].weight isa Optimisers.Leaf{<:AdamW}
end
