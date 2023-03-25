using Flux, Optimisers, Tsunami, MLDatasets
using MLUtils: MLUtils, DataLoader, flatten
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

function Tsunami.training_step(m::MLP, trainer, batch, batch_idx)
    x, y = batch
    ŷ = m(x)
    y = Flux.onehotbatch(y, 0:9)
    loss = Flux.Losses.logitcrossentropy(ŷ, y)
    Tsunami.log(trainer, "loss/train", loss, prog_bar=true)
    Tsunami.log(trainer, "accuracy/train", Tsunami.accuracy(ŷ, y), prog_bar=true)
    return loss
end

function Tsunami.validation_step(m::MLP, trainer, batch, batch_idx)
    x, y = batch
    ŷ = m(x)
    y = Flux.onehotbatch(y, 0:9)
    loss = Flux.logitcrossentropy(ŷ, y)
    Tsunami.log(trainer, "loss/val", loss)
    Tsunami.log(trainer, "accuracy/val", Tsunami.accuracy(ŷ, y))
end


function Tsunami.configure_optimisers(m::MLP, trainer)
    # initial lr, decay factor, and decay intervals (corresponding to epochs 2 and 4)
    lr_scheduler = ParameterSchedulers.Step(1e-2, 1/10, [2, 2])
    opt = Optimisers.setup(Optimisers.AdamW(), m)
    return opt, lr_scheduler
end

train_loader = DataLoader(MNIST(:train), batchsize=128, shuffle=true)
test_loader = DataLoader(MNIST(:test), batchsize=128)

model = MLP()

# DRY RUN FOR DEBUGGING

trainer = Trainer(fast_dev_run=true, accelerator=:cpu)
Tsunami.fit!(model, trainer, train_loader, test_loader)

# TRAIN FROM SCRATCH

trainer = Trainer(max_epochs = 3, 
                 default_root_dir = @__DIR__,
                 accelerator = :cpu,
                 checkpointer = true,
                 logger = true,
                 )

fit_state = Tsunami.fit!(model, trainer, train_loader, test_loader)
@assert fit_state.step == 1407

# RESUME TRAINING
trainer = Trainer(max_epochs = 5,
                 default_root_dir = @__DIR__,
                 accelerator = :cpu,
                 checkpointer = true,
                 logger = true,
                 )

ckpt_path = joinpath(fit_state.run_dir, "checkpoints", "ckpt_last.bson")

fit_state = Tsunami.fit!(model, trainer, train_loader, test_loader; ckpt_path)
@assert fit_state.step == 2345
