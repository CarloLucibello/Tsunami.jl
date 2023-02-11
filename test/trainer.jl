
@testset "no val loader" begin
    model = TestModule1()
    nx, ny = io_sizes(model)
    train_dataloader = make_dataloader(nx, ny)
    trainer = Trainer(max_epochs=2, logger=false, checkpointer=false, progress_bar=false)
    Tsunami.fit!(model, trainer; train_dataloader, val_dataloader=train_dataloader)
end

@testset "fit! mutates" begin
    model = TestModule1()
    # model0 = deepcopy(model)
    nx, ny = io_sizes(model)
    @test all(==(0), model.net[1].bias)
    train_dataloader = make_dataloader(nx, ny)
    trainer = Trainer(max_epochs=2, logger=false, checkpointer=false, progress_bar=false)
    Tsunami.fit!(model, trainer; train_dataloader)
    @test all(!=(0), model.net[1].bias)
end

@testset "checkpoint" begin
    model = TestModule1()

    nx, ny = io_sizes(model)
    train_dataloader = make_dataloader(nx, ny)
    x, y = first(train_dataloader)
    ŷ0 = model(x)
    loss0 = Flux.Losses.mse(ŷ0, y)

    trainer = Trainer(max_epochs=2, logger=false, checkpointer=true, progress_bar=true, default_root_dir=@__DIR__)
    Tsunami.fit!(model, trainer; train_dataloader)
    runpath1 = joinpath(@__DIR__, "tsunami_logs", "run_1")
    bsonpath1 = joinpath(runpath1, "checkpoints", "ckpt_epoch=2_step=4.bson")
    @test isfile(bsonpath1)
    Tsunami.fit!(model, trainer; train_dataloader)
    runpath2 = joinpath(@__DIR__, "tsunami_logs", "run_2")
    bsonpath2 = joinpath(runpath2, "checkpoints", "ckpt_epoch=2_step=4.bson")
    @test isfile(bsonpath2)

    ŷ = model(x)
    loss = Flux.Losses.mse(ŷ, y)
    @test loss < loss0

    ckpt = Tsunami.load_checkpoint(bsonpath2)
    @test ckpt.model isa TestModule1
    @test ckpt.epoch == 2
    @test ckpt.step == 4
    @test ckpt.model(x) ≈ ŷ

    rm(joinpath(@__DIR__, "tsunami_logs/"), recursive=true)
end

@testset "fast_dev_run" begin
    model = TestModule1()
    nx, ny = io_sizes(model)
    train_dataloader = make_dataloader(nx, ny)
    trainer = Trainer(max_epochs=2, logger=false, checkpointer=false, progress_bar=false, fast_dev_run=true)
    fit_state = Tsunami.fit!(model, trainer; train_dataloader)
    @test fit_state[:epoch] == 1
    @test fit_state[:step] == 1
end