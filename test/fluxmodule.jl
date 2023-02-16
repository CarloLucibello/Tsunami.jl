@testset "abstract type FluxModule" begin
    @test isabstracttype(FluxModule)
end

@testset "functor" begin
    m = TestModule1()
    @test Functors.children(m) == (; net = m.net, tuple_field = m.tuple_field)
end

@testset "check_fluxmodule" begin
    struct Immut <: FluxModule end
    @test_throws AssertionError Tsunami.check_fluxmodule(Immut())
    Tsunami.check_fluxmodule(TestModule1())
end

