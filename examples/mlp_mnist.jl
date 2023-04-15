using Flux, Optimisers, Tsunami, MLDatasets
using MLUtils: MLUtils, DataLoader, flatten, mapobs, splitobs
import ParameterSchedulers

mutable struct MLP <: FluxModule
    net
end

function MLP()
    net = Chain(
            flatten,
            Dense(28^2 => 256, relu), 
            Dense(256 => 10))

    return MLP(net)
end

function (m::MLP)(x)
    m.net(x)
end

function Tsunami.train_step(m::MLP, trainer, batch)
    x, y = batch
    ŷ = m(x)
    loss = Flux.Losses.logitcrossentropy(ŷ, y)
    Tsunami.log(trainer, "loss/train", loss)
    Tsunami.log(trainer, "accuracy/train", Tsunami.accuracy(ŷ, y))
    return loss
end

function Tsunami.val_step(m::MLP, trainer, batch)
    x, y = batch
    ŷ = m(x)
    loss = Flux.logitcrossentropy(ŷ, y)
    Tsunami.log(trainer, "loss/val", loss)
    Tsunami.log(trainer, "accuracy/val", Tsunami.accuracy(ŷ, y))
end

function Tsunami.test_step(m::MLP, trainer, batch)
    x, y = batch
    ŷ = m(x)
    loss = Flux.logitcrossentropy(ŷ, y)
    Tsunami.log(trainer, "loss/test", loss)
    Tsunami.log(trainer, "accuracy/test", Tsunami.accuracy(ŷ, y))
end

function Tsunami.configure_optimisers(m::MLP, trainer)
    # initial lr, decay factor, and decay intervals (corresponding to epochs 2 and 4)
    lr_scheduler = ParameterSchedulers.Step(1e-2, 1/10, [2, 2])
    opt = Optimisers.setup(Optimisers.Adam(1e-5), m)
    return opt, lr_scheduler
end

train_data = mapobs(batch -> (batch[1], Flux.onehotbatch(batch[2], 0:9)), MNIST(:train))
train_data, val_data = splitobs(train_data, at = 0.9)
test_data = mapobs(batch -> (batch[1], Flux.onehotbatch(batch[2], 0:9)), MNIST(:test))
 
train_loader = DataLoader(train_data, batchsize=128, shuffle=true)
val_loader = DataLoader(val_data, batchsize=128, shuffle=true)
test_loader = DataLoader(test_data, batchsize=128)

# CREATE MODEL

model = MLP()

# DRY RUN FOR DEBUGGING

trainer = Trainer(fast_dev_run=true, accelerator=:cpu)
Tsunami.fit!(model, trainer, train_loader, val_loader)

# TRAIN FROM SCRATCH

Tsunami.seed!(17)
model = MLP()
trainer = Trainer(max_epochs = 3,
                 max_steps = -1,
                 default_root_dir = @__DIR__,
                 accelerator = :cpu)

fit_state = Tsunami.fit!(model, trainer, train_loader, val_loader)
@assert fit_state.step == 1266

# RESUME TRAINING
trainer = Trainer(max_epochs = 5,
                 default_root_dir = @__DIR__,
                 accelerator = :cpu,
                 checkpointer = true,
                 logger = true,
                 )

ckpt_path = joinpath(fit_state.run_dir, "checkpoints", "ckpt_last.bson")

fit_state = Tsunami.fit!(model, trainer, train_loader, val_loader; ckpt_path)
@assert fit_state.step == 2110

# TEST
test_results = Tsunami.test(model, trainer, test_loader)
