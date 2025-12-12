@testset "Trainer Constructor" begin
    using MLDataDevices, Tsunami
    trainer = Trainer()
    @test trainer.max_epochs == 1000
    @test trainer.max_steps == -1

    trainer = Trainer(max_steps=10)
    @test trainer.max_epochs == typemax(Int)
    @test trainer.max_steps == 10

    trainer = Trainer(max_epochs=10)
    @test trainer.max_epochs == 10
    @test trainer.max_steps == -1

    trainer = Trainer(max_epochs=10, max_steps=20)
    @test trainer.max_epochs == 10
    @test trainer.max_steps == 20

    trainer = Trainer(accelerator=:cpu)
    @test trainer.foil.device isa CPUDevice
end

@testset "no val loader" begin
    using .TsunamiTest
    model = TestModule1()
    nx, ny = io_sizes(model)
    train_dataloader = make_dataloader(nx, ny)
    trainer = Trainer(max_epochs=2, logger=false, checkpointer=false, progress_bar=false)
    Tsunami.fit!(model, trainer, train_dataloader)
end

@testset "fit! mutates" begin
    using .TsunamiTest
    model = TestModule1()
    # model0 = deepcopy(model)
    nx, ny = io_sizes(model)
    @test all(==(0), model.net[1].bias)
    train_dataloader = make_dataloader(nx, ny)
    trainer = SilentTrainer(max_epochs=2, precision=:f64)
    res = Tsunami.fit!(model, trainer, train_dataloader)
    @test all(!=(0), model.net[1].bias)
    @test res === nothing

    @testset "also copy state" begin
        model = TestModule1(Chain(Dense(4, 3, relu), BatchNorm(3), Dense(3, 2)))
        Tsunami.fit!(model, trainer, train_dataloader)
        @test all(!=(0), model.net[2].β)
    end
end

@testset "checkpoint" begin
    using .TsunamiTest
    model = TestModule1()

    nx, ny = io_sizes(model)
    train_dataloader = make_dataloader(nx, ny)
    x, y = first(train_dataloader)
    ŷ0 = model(x)
    loss0 = Flux.Losses.mse(ŷ0, y)

    trainer = Trainer(max_epochs=2, logger=false, checkpointer=true, progress_bar=true, default_root_dir=@__DIR__)
    Tsunami.fit!(model, trainer, train_dataloader)
    runpath1 = trainer.fit_state.run_dir
    ckptpath1 = joinpath(runpath1, "checkpoints", "ckpt_epoch=2_step=4.jld2")
    @test isfile(ckptpath1)
    Tsunami.fit!(model, trainer, train_dataloader)
    runpath2 = trainer.fit_state.run_dir
    ckptpath2 = joinpath(runpath2, "checkpoints", "ckpt_epoch=2_step=4.jld2")
    @test isfile(ckptpath2)

    ŷ = model(x)
    loss = Flux.Losses.mse(ŷ, y)
    @test loss < loss0

    ckpt = Tsunami.load_checkpoint(ckptpath2)
    @test ckpt.fit_state.epoch == 2
    @test ckpt.fit_state.step == 4

    model2 = TestModule1()
    Flux.loadmodel!(model2, ckpt.model_state)
    @test model2(x) ≈ ŷ

    rm(runpath1, recursive=true)
    rm(runpath2, recursive=true)
end

@testset "fast_dev_run" begin
    using .TsunamiTest
    model = TestModule1()
    nx, ny = io_sizes(model)
    train_dataloader = make_dataloader(nx, ny)
    trainer = SilentTrainer(max_epochs=2, fast_dev_run=true)
    Tsunami.fit!(model, trainer, train_dataloader)
    @test trainer.fit_state.epoch == 0
    @test trainer.fit_state.step == 0
end

@testset "val_every_n_epochs" begin
    using .TsunamiTest
    # TODO test properly
    model = TestModule1()
    nx, ny = io_sizes(model)
    train_dataloader = make_dataloader(nx, ny)
    trainer = SilentTrainer(max_epochs=2, val_every_n_epochs=2)
    Tsunami.fit!(model, trainer, train_dataloader)
    @test trainer.fit_state.epoch == 2 
end

@testset "generic dataloader" begin
    using .TsunamiTest
    model = TestModule1()
    nx, ny = io_sizes(model)
    train_dataloader = [(rand(Float32, nx, 2), rand(Float32, ny, 2))]
    val_dataloader = [(rand(Float32, nx, 2), rand(Float32, ny, 2))]
    trainer = SilentTrainer(max_epochs = 2)
    Tsunami.fit!(model, trainer, train_dataloader, val_dataloader)
    @test trainer.fit_state.epoch == 2
end

@testset "Tsunami.test" begin
    using .TsunamiTest
    struct TestModuleTest <: FluxModule; dummy; end 
    function Tsunami.test_step(::TestModuleTest, trainer, batch, batch_idx)
        Tsunami.log(trainer, "a", 1)
        Tsunami.log(trainer, "b", batch_idx)
    end
    test_dataloader = [rand(2) for i=1:3]
    trainer = Trainer()
    model = TestModuleTest(zeros(2))
    res = Tsunami.test(model, trainer, test_dataloader)
    @test res["a"] == 1
    @test res["b"] == 2.0
end

@testset "Tsunami.validate" begin
    using .TsunamiTest
    struct TestModuleVal <: FluxModule; dummy; end 

    function Tsunami.val_step(::TestModuleVal, trainer, batch, batch_idx)
        
        Tsunami.log(trainer, "a", 1)
        Tsunami.log(trainer, "b", batch_idx)
    end
    
    val_dataloader = [rand(2) for i=1:3]
    trainer = Trainer()
    model = TestModuleVal(zeros(2))
    res = Tsunami.validate(model, trainer, val_dataloader)
    @test res["a"] == 1
    @test res["b"] == 2.0
end

using Sys
if !Sys.isapple() # sometimes hangs on mac
@testset "precision bf16" begin
    using .TsunamiTest
    model = TestModule1()
    nx, ny = io_sizes(model)
    train_dataloader = make_dataloader(nx, ny)
    trainer = Trainer(max_epochs=2, precision=:bf16, accelerator=:cpu)
    Tsunami.fit!(model, trainer, train_dataloader)
    # TODO test properly
end
end
