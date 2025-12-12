module TsunamiTest

using Pkg

ENV["DATADEPS_ALWAYS_ACCEPT"] = true # for MLDatasets in examples

## Uncomment below to change the default test settings
# ENV["TSUNAMI_TEST_CUDA"] = "true"
# ENV["TSUNAMI_TEST_AMDGPU"] = "true"
# ENV["TSUNAMI_TEST_Metal"] = "true"

to_test(backend) = get(ENV, "TSUNAMI_TEST_$(backend)", "false") == "true"
has_dependecies(pkgs) = all(pkg -> haskey(Pkg.project().dependencies, pkg), pkgs)
deps_dict = Dict(:CUDA => ["CUDA", "cuDNN"], :AMDGPU => ["AMDGPU"], :Metal => ["Metal"])

for (backend, deps) in deps_dict
    if to_test(backend)
        if !has_dependecies(deps)
            Pkg.add(deps)
        end
        @eval using $backend
        if backend == :CUDA
            @eval using cuDNN
        end
        @eval $backend.allowscalar(false)
    end
end

using Reexport: @reexport
using DataFrames: DataFrames, DataFrame
using Enzyme: Enzyme
using MLUtils: MLUtils
@reexport using Test
@reexport using Tsunami
@reexport using Flux
@reexport using Optimisers

export SilentTrainer, 
      NotModule, TestModule1, LinearModel, TBLoggingModule, MLP,
      io_sizes, make_regression_dataset, make_dataloader, read_tensorboard_logs_asdf,
      DataFrames, DataFrame, Enzyme, MLUtils


const SilentTrainer = (args...; kws...) -> Trainer(args...; kws..., logger=false, checkpointer=false, progress_bar=false)

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

function make_dataloader(nx, ny; n=10, bs=5)
    x, y = make_regression_dataset(nx, ny, n)
    return Flux.DataLoader((x, y), batchsize=bs)
end

function read_tensorboard_logs_asdf(logdir)
    events = Tsunami.read_tensorboard_logs(logdir)
    df = DataFrame([(; name, step, value) for (name, step, value) in events])
    df = DataFrames.unstack(df, :step, :name, :value)
    return df
end

struct NotModule
    net
end

############ TestModule1 ############
struct TestModule1 <: FluxModule
    net
    tuple_field::Tuple{Int, Int}
end

TestModule1() = TestModule1(Flux.Chain(Flux.Dense(4, 3, relu), Flux.Dense(3, 2)), (1, 2))
TestModule1(net) = TestModule1(net, (1, 2))

(m::TestModule1)(x) = m.net(x)

function Tsunami.train_step(m::TestModule1, trainer, batch, batch_idx)
    x, y = batch
    y_hat = m(x)
    loss = Flux.mse(y_hat, y)
    return loss
end

function Tsunami.configure_optimisers(m::TestModule1, trainer)
    return Optimisers.Adam(1f-3)
end

# utility returning input and output sizes
io_sizes(m::TestModule1) = 4, 2

############ LinearModel ############

struct LinearModel{Tw, Tm, F} <: FluxModule
    W::Tw
    mask::Tm
    λ::F  # L2 regularization
end

function LinearModel(N::Int; λ = 0.0f0)
    W = randn(Float32, 1, N) ./ Float32(sqrt(N))
    mask = fill(true, size(W))
    return LinearModel(W, mask, λ)
end

function (m::LinearModel)(x::AbstractMatrix)
    return (m.W .* m.mask) * x
end

function Tsunami.train_step(model::LinearModel, trainer, batch, batch_idx)
    x, y = batch
    ŷ = model(x)
    loss_data = Flux.mse(ŷ, y)
    loss_reg = model.λ * sum(abs2.(model.W)) 
    # Zygote.ignore_derivatives() do
    #     @show loss_data loss_reg
    # end
    return (; loss = loss_data + loss_reg, loss_data, loss_reg)
end

function Tsunami.configure_optimisers(model::LinearModel, trainer)
    return Optimisers.Adam(1f-1)
end

###### MLP ######

struct MLP{T} <: FluxModule
    net::T
    mode::Symbol
end

function MLP(din, dout, mode)
    @assert mode in (:classification, :regression)
    net = Chain(Dense(din => 128, relu), Dense(128 => dout))
    return MLP(net, mode)
end

(model::MLP)(x) = model.net(MLUtils.flatten(x))

function loss_and_accuracy(model::MLP, batch)
    x, y = batch
    ŷ = model(x)
    if model.mode == :regression
        loss = Flux.mse(ŷ, y)
        return loss, loss
    else
        loss, acc = Flux.logitcrossentropy(ŷ, y), Tsunami.accuracy(ŷ, y)
        return loss, acc
    end
end

function Tsunami.train_step(model::MLP, trainer, batch)
    loss, acc = loss_and_accuracy(model, batch)
    Tsunami.log(trainer, "train/loss", loss, prog_bar=true)
    Tsunami.log(trainer, "train/accuracy", acc, prog_bar=true)
    return loss
end

Tsunami.configure_optimisers(model::MLP, trainer) = Optimisers.AdamW(1e-3)

###### TBLoggingModuel ######

Base.@kwdef struct TBLoggingModule <: FluxModule
    net = Chain(Dense(4, 3, relu), Dense(3, 2))
    log_on_train_epoch::Bool = true
    log_on_train_step::Bool = true
    log_on_val_epoch::Bool = true
    log_on_val_step::Bool = true
end

(m::TBLoggingModule)(x) = m.net(x)

io_sizes(m::TBLoggingModule) = 4, 2

function Tsunami.train_step(m::TBLoggingModule, trainer, batch, batch_idx)
    x, y = batch
    y_hat = m(x)
    loss = Flux.mse(y_hat, y)
    on_step = m.log_on_train_step
    on_epoch = m.log_on_train_epoch
    Tsunami.log(trainer, "train/loss", loss; on_step, on_epoch, prog_bar=true)
    Tsunami.log(trainer, "train/batch_idx", batch_idx; on_step, on_epoch, prog_bar=true)
    return loss
end

function Tsunami.val_step(m::TBLoggingModule, trainer, batch, batch_idx)
    x, y = batch
    y_hat = m(x)
    loss = Flux.mse(y_hat, y)
    on_step = m.log_on_val_step
    on_epoch = m.log_on_val_epoch
    Tsunami.log(trainer, "val/loss", loss; on_step, on_epoch, prog_bar=true)
    Tsunami.log(trainer, "val/batch_idx", batch_idx; on_step, on_epoch, prog_bar=true)
    return loss
end

function Tsunami.configure_optimisers(m::TBLoggingModule, trainer)
    return Optimisers.Adam(1f-3)
end

end # TsunamiTest