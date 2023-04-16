@testset "abstract type FluxModule" begin
    @test isabstracttype(FluxModule)
end

@testset "functor" begin
    m = TestModule1()
    @test Functors.children(m) == (; net = m.net, tuple_field = m.tuple_field)

    @testset "parametric modules" begin
        struct ParModule{A,B} <: FluxModule; a::A; b::B; end
        m = ParModule(rand(2), rand(2))
        m2 = m |> Flux.f32 
        @test m2.net isa AbstractArray{Float32}
    end

end

@testset "check_fluxmodule" begin
    struct Immut <: FluxModule end
    @test_throws AssertionError Tsunami.check_fluxmodule(Immut())
    Tsunami.check_fluxmodule(TestModule1())
end

