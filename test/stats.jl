@testset "uneven nobs" begin
    s = Tsunami.Stats()
    OnlineStats.fit!(s, Dict("a" => 1, "b" => 2))
    @test OnlineStats.nobs(s) == 1
    OnlineStats.fit!(s, Dict("a" => 1, "c" => 2, "d" => 3))
    @test OnlineStats.nobs(s) == Dict("c" => 1, "b" => 1, "a" => 2, "d" => 1)
end
