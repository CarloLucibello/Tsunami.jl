@testitem "GradScaler" begin
    gs = Tsunami.GradScaler()
    # test default values
    @test gs.scale == 2.0f0^16
    @test gs.growth_factor == 2.0f0
    @test gs.backoff_factor == 0.5f0
    @test gs.growth_interval == 2000
    @test gs.min_scale == 1.0f0
    @test gs.growth_tracker == 0
    @test gs.found_inf == false

    # @testset "scale loss" begin
    loss = Float32(2)
    @test Tsunami.scale(gs, loss) == loss * gs.scale

    loss = Float16(2)
    @test_warn r"^┌ Warning: Scaling Float16 loss" Tsunami.scale(gs, loss) 
    @test Tsunami.scale(gs, loss) == loss * gs.scale
end