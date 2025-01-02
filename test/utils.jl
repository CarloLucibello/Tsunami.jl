@testitem "Tsunami.seed!" begin
    using Random
    Tsunami.seed!(17)
    x = rand(10)
    Random.seed!(17)
    y = rand(10)
    @test x ≈ y
end

@testitem "Tsunami.seed! GPU" setup=[TsunamiTest] tags=[:gpu] begin
    using .TsunamiTest
    using Random
    Tsunami.seed!(17)
    x = rand(10)
    Random.seed!(17)
    y = rand(10)
    @test x ≈ y
end

