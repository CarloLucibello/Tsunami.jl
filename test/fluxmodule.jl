@testitem "abstract type FluxModule" begin
    @test isabstracttype(FluxModule)
end

@testitem "FluxModule functor" setup=[TsunamiTest] begin
    using .TsunamiTest
    using Functors: Functors
    m = TestModule1()
    @test Functors.children(m) == (; net = m.net, tuple_field = m.tuple_field)

    @testset "parametric modules" begin
        struct ParModule{A,B} <: FluxModule; a::A; b::B; end
        m = ParModule(rand(2), rand(2))
        m2 = m |> Flux.f32 
        @test m2.a isa Vector{Float32}
    end
end
