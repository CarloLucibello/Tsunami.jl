@testitem "Tsunami.seed!" begin
    using Random
    Tsunami.seed!(17)
    x = rand(10)
    Random.seed!(17)
    y = rand(10)
    @test x ≈ y
    
    # TODO
    # if CUDA.functional()
    #     Tsunami.seed!(17)
    #     x = CUDA.rand(10)
    #     CUDA.seed!(17)
    #     y = CUDA.rand(10)
    #     @test x ≈ y
    # end
end

