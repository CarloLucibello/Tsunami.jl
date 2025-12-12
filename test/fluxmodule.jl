@testset "abstract type FluxModule" begin
    using Tsunami
    @test isabstracttype(FluxModule)
end

@testset "FluxModule functor" begin
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
