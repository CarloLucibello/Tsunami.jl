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

@testset "on_train_epoch_start and on_train_epoch_end" begin
    out = []

    struct TrainEpochCbk end
    
    function Tsunami.on_train_epoch_start(::TrainEpochCbk, model, trainer)
        push!(out, 1)
    end

    function Tsunami.on_train_epoch_end(::TrainEpochCbk, model, trainer)
        push!(out, 2)
    end
    trainer = SilentTrainer(max_epochs=2, callbacks=[TrainEpochCbk()])
    model = TestModule1()
    train_dataloader = make_dataloader(io_sizes(model)..., 10, 5)
    Tsunami.fit!(model, trainer, train_dataloader)
    @test out == [1, 2, 1, 2]
end


@testset "on_val_epoch_start and on_val_epoch_end" begin
    out = []

    struct ValEpochCbk end
    
    function Tsunami.on_val_epoch_start(::ValEpochCbk, model, trainer)
        push!(out, 1)
    end

    function Tsunami.on_val_epoch_end(::ValEpochCbk, model, trainer)
        push!(out, 2)
    end
    trainer = SilentTrainer(max_epochs=2, callbacks=[ValEpochCbk()])
    model = TestModule1()
    train_dataloader = make_dataloader(io_sizes(model)..., 10, 5)
    Tsunami.fit!(model, trainer, train_dataloader, train_dataloader)
    @test out == [1, 2, 1, 2, 1, 2]

    Tsunami.validate(model, trainer, train_dataloader)
    @test out == [1, 2, 1, 2, 1, 2, 1, 2]
end

@testset "on_test_epoch_start and on_test_epoch_end" begin
    out = []

    struct TestEpochCbk end
    
    function Tsunami.on_test_epoch_start(::TestEpochCbk, model, trainer)
        push!(out, 1)
    end

    function Tsunami.on_test_epoch_end(::TestEpochCbk, model, trainer)
        push!(out, 2)
    end
    trainer = SilentTrainer(max_epochs=2, callbacks=[TestEpochCbk()])
    model = TestModule1()
    train_dataloader = make_dataloader(io_sizes(model)..., 10, 5)
    Tsunami.test(model, trainer, train_dataloader)
    @test out == [1, 2]
end

@testset "on_before_backward" begin
    out = []

    struct BeforePullbackCbk end
    
    function Tsunami.on_before_backward(::BeforePullbackCbk, model, trainer, loss)
        push!(out, loss)
    end
    function Tsunami.on_before_update(::BeforePullbackCbk, model, trainer, grad)
        @show grad
    end

    trainer = SilentTrainer(max_epochs=2, callbacks=[BeforePullbackCbk()])
    model = TestModule1()
    train_dataloader = make_dataloader(io_sizes(model)..., 10, 5)
    Tsunami.fit!(model, trainer, train_dataloader)
    @test out == [1, 1]
end
