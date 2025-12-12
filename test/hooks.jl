@testset "on_before_update" begin
    using .TsunamiTest
    
    Base.@kwdef struct OnBeforeUpdateCbk 
        res = []
    end
    
    function Tsunami.on_before_update(cb::OnBeforeUpdateCbk, model, trainer, out, grad)
        push!(cb.res, grad)
    end

    cb = OnBeforeUpdateCbk()
    trainer = SilentTrainer(max_epochs=1, callbacks=[cb])
    model = TestModule1()
    train_dataloader = make_dataloader(io_sizes(model)..., n=10, bs=5)
    Tsunami.fit!(model, trainer, train_dataloader)
    @test length(cb.res) == 2
    @test cb.res[1] isa NamedTuple
    @test cb.res[2] isa NamedTuple
    @test size(cb.res[1].net.layers[1].weight) == (3, 4)
end

@testset "on_train_epoch_start and on_train_epoch_end" begin
    using .TsunamiTest
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
    train_dataloader = make_dataloader(io_sizes(model)..., n=10, bs=5)
    Tsunami.fit!(model, trainer, train_dataloader)
    @test out == [1, 2, 1, 2]
end


@testset "on_val_epoch_start and on_val_epoch_end" begin
    using .TsunamiTest
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
    train_dataloader = make_dataloader(io_sizes(model)..., n=10, bs=5)
    Tsunami.fit!(model, trainer, train_dataloader, train_dataloader)
    @test out == [1, 2, 1, 2, 1, 2]

    Tsunami.validate(model, trainer, train_dataloader)
    @test out == [1, 2, 1, 2, 1, 2, 1, 2]
end

@testset "on_test_epoch_start and on_test_epoch_end" begin
    using .TsunamiTest

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
    train_dataloader = make_dataloader(io_sizes(model)..., n=10, bs=5)
    Tsunami.test(model, trainer, train_dataloader)
    @test out == [1, 2]
end

@testset "on_train_batch_start and on_train_batch_end" begin
    using .TsunamiTest
    
    Base.@kwdef struct OnBatchCbk 
        res = []
    end
    
    function Tsunami.on_train_batch_start(cb::OnBatchCbk, model, trainer, batch, batch_idx)
        push!(cb.res, batch_idx)
    end
    function Tsunami.on_train_batch_end(cb::OnBatchCbk, model, trainer, out, batch, batch_idx)
        @test out isa Number
        push!(cb.res, batch_idx)
    end

    cb = OnBatchCbk()
    trainer = SilentTrainer(max_epochs=2, callbacks=[cb])
    model = TestModule1()
    train_dataloader = make_dataloader(io_sizes(model)..., n=10, bs=5)
    Tsunami.fit!(model, trainer, train_dataloader)
    @test cb.res == [1, 1, 2, 2, 1, 1, 2, 2]
end
