@testset "Tsunami.seed!" begin
    using Tsunami, Random
    Tsunami.seed!(17)
    x = rand(2)
    Random.seed!(17)
    y = rand(2)
    @test x ≈ y
end
