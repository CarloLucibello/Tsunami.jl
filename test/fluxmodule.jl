@testset "abstract type FluxModule" begin
    @test isabstracttype(FluxModule)
end

@testset "functor" begin
    m = TestModule1()
    @test Functors.children(m) == (; net = m.net)
end

@testset "check_fluxmodule" begin
    struct Immut <: FluxModule end
    @test_throws AssertionError Tsunami.check_fluxmodule(Immut())
end

