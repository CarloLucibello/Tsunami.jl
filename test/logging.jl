@testset "TensorBoard logging" begin
    batch_sizes = [3, 2]
    max_epochs = 4

    tot_steps = length(batch_sizes) * max_epochs
    steps_per_epoch = length(batch_sizes)
    end_epoch_steps = length(batch_sizes) .* (1:max_epochs)

    model = TBLoggingModule(log_on_train_step=true, log_on_train_epoch=true)
    trainer = Trainer(max_epochs=4, log_every_n_steps=1)
    train_dataloader = make_regression_dataset(io_sizes(model)..., batch_sizes)
    fit_state = Tsunami.fit!(model, trainer; train_dataloader)

    events = Tsunami.read_tensorboard_logs(fit_state.run_dir)
    @test events isa Vector{Tuple{String, Int64, <:Any}} 
    @test count(x -> x[1] == "train/loss_step", events) == 8

    df = read_tensorboard_logs_asdf(fit_state.run_dir)
    @test "train/loss_step" ∈ names(df)
    @test "train/loss_epoch" ∈ names(df)
    @test "train/loss" ∉ names(df)
    @test size(df, 1) == tot_steps

    @test all(df[:,"train/loss_epoch"][(1:tot_steps) .∉ Ref(end_epoch_steps)] .=== missing)
    @test all(df[:,"train/loss_epoch"][(1:tot_steps) .∈ Ref(end_epoch_steps)] .!== missing)
    @test all(df[:,"train/loss_step"] .!== missing)

    @testset "mean weighted with batchsize" begin
        for step in end_epoch_steps
            @test df[step, "train/batch_idx_epoch"] ≈ sum(batch_sizes .* (1:length(batch_sizes))) / sum(batch_sizes)
        end
    end
end
