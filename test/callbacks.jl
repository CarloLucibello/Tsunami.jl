@testset "custom callbacks" begin
    using .TsunamiTest

    Base.@kwdef mutable struct CounterCbk
        end_counter::Int = 0
        start_counter::Int = 0
    end

    function Tsunami.on_train_epoch_start(cb::CounterCbk, model, trainer::Trainer)
        cb.start_counter += 2
    end

    function Tsunami.on_train_epoch_end(cb::CounterCbk, model, trainer::Trainer)
        cb.end_counter += 1
    end

    model = TestModule1()
    nx, ny = io_sizes(model)
    cb = CounterCbk()
    train_dataloader = [(rand(Float32, nx, 2), rand(Float32, ny, 2))]
    val_dataloader = [(rand(Float32, nx, 2), rand(Float32, ny, 2))]
    trainer = SilentTrainer(max_epochs = 4, callbacks=[cb])
    Tsunami.fit!(model, trainer, train_dataloader, val_dataloader)
    @test cb.start_counter == 8
    @test cb.end_counter == 4
end
