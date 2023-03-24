function make_regression_dataset(nx, ny, n=10)
    if n isa Integer
        x = randn(Float32, nx..., n)
        y = randn(Float32, ny..., n)
        return x, y
    else
        x = [randn(Float32, nx..., ni) for ni in n]
        y = [randn(Float32, ny..., ni) for ni in n]
        return zip(x, y)
    end
end

function make_dataloader(nx, ny, n=10, bs=5)
    x, y = make_regression_dataset(nx, ny, n)
    return Flux.DataLoader((x, y), batchsize=bs)
end

function read_tensorboard_logs_asdf(logdir)
    events = Tsunami.read_tensorboard_logs(logdir)
    df = DataFrame([(; name, step, value) for (name, step, value) in events])
    df = DataFrames.unstack(df, :step, :name, :value)
    return df
end