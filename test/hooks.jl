@testset "on_before_update" begin
    out = []

    struct OnBeforeUpdateCbk end
    
    function Tsunami.on_before_update(::OnBeforeUpdateCbk, model, trainer, grad)
        push!(out, grad)
    end
    trainer = SilentTrainer(max_epochs=1, callbacks=[OnBeforeUpdateCbk()])
    model = TestModule1()
    train_dataloader = make_dataloader(io_sizes(model)..., 10, 5)
    Tsunami.fit!(model, trainer, train_dataloader)
    @test length(out) == 2
    @test out[1] isa NamedTuple
    @test out[2] isa NamedTuple
    @test size(out[1].net.layers[1].weight) == (3, 4)
end