using Flux, Optimisers, Tsunami, MLDatasets
using Flux: DataLoader, flatten
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

function Tsunami.training_step(m::MLP, batch, batch_idx)
    x, y = batch
    ŷ = m(x)
    y = Flux.onehotbatch(y, 0:9)
    loss = Flux.Losses.logitcrossentropy(ŷ, y)
    acc = Tsunami.accuracy(ŷ, y)
    return (; loss, acc)
end

function Tsunami.configure_optimisers(m::MLP)
    # initial lr, decay factor, decay intervals (corresponding to epochs 2 and 4)
    lr_scheduler = ParameterSchedulers.Step(1e-2, 1/10, [2, 2])
    opt = Optimisers.setup(Optimisers.AdamW(), m)
    return lr_scheduler, opt
end

train_loader = DataLoader(MNIST(:train), batchsize=128, shuffle=true)
test_loader = DataLoader(MNIST(:test), batchsize=128)


model = MLP()

# DRY RUN FOR DEBUGGING

trainer = Trainer(fast_dev_run=true, accelerator=:cpu)
Tsunami.fit!(model, trainer; train_dataloader=train_loader, val_dataloader=test_loader)

# TRAIN FROM SCRATCH

trainer = Trainer(max_epochs = 2, 
                 default_root_dir = @__DIR__,
                 accelerator = :cpu,
                 checkpointer = true,
                 logger = true,
                 )

fit_state = Tsunami.fit!(model, trainer; train_dataloader=train_loader, val_dataloader=test_loader)

# RESUME TRAINING

trainer.max_epochs = 5
ckpt_path = joinpath(fit_state[:run_dir], "checkpoints", "ckpt_last.bson")

Tsunami.fit!(model, trainer; train_dataloader=train_loader, val_dataloader=test_loader, ckpt_path)
