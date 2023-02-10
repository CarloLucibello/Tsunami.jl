function make_dataset(nx, ny, n=10)
    x = randn(Float32, nx..., n)
    y = randn(Float32, ny..., n)
    return x, y
end

function make_dataloader(nx, ny, n=10, bs=5)
    x, y = make_dataset(nx, ny, n)
    return Flux.DataLoader((x, y), batchsize=bs)
end