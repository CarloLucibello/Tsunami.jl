# An example of plain Flux code where only 
# Foil is used to select the device. 
# Instead of the `Trainer.fit!` function we use a custom training loop.

using Flux, Optimisers, Tsunami, MLDatasets
using MLUtils: MLUtils, DataLoader, flatten, mapobs, splitobs
import ParameterSchedulers
## Uncomment one of the following lines for GPU support
# using CUDA
# using AMDGPU
# using Metal

function MLP()
    return Chain(
            flatten,
            Dense(28^2 => 256, relu), 
            Dense(256 => 10))
end


function train_step(model, batch)
    x, y = batch
    ŷ = model(x)
    loss = Flux.Losses.logitcrossentropy(ŷ, y)
    acc =  Tsunami.accuracy(ŷ, y)
    return loss
end

function val_step(model, batch)
    x, y = batch
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)
    acc = Tsunami.accuracy(ŷ, y)
    return loss, acc
end

function configure_optimisers(model)
    lr_scheduler = ParameterSchedulers.Step(1f-2, 0.1f0, [2, 2])
    opt = Optimisers.setup(Optimisers.Adam(1f-5), model)
    return opt, lr_scheduler
end

train_data = mapobs(batch -> (batch[1], Flux.onehotbatch(batch[2], 0:9)), MNIST(:train))
train_data, val_data = splitobs(train_data, at = 0.9)
test_data = mapobs(batch -> (batch[1], Flux.onehotbatch(batch[2], 0:9)), MNIST(:test))
 
train_loader = DataLoader(train_data, batchsize=128, shuffle=true)
val_loader = DataLoader(val_data, batchsize=128, shuffle=true)
test_loader = DataLoader(test_data, batchsize=128)

model = MLP()
opt, lr_scheduler = configure_optimisers(model)

foil = Foil(accelerator=:auto, precision=:f32)
model, opt = Tsunami.setup(foil, model, opt)

EPOCHS = 10

for epoch in 1:EPOCHS
    for batch in train_loader
        batch = Tsunami.setup_batch(foil, batch)
        grad = gradient(model -> train_step(model, batch), model)
        Optimisers.update!(opt, model, grad[1])
    end

    ntot = 0
    val_loss = 0.
    val_acc = 0.
    for batch in val_loader
        batch = Tsunami.setup_batch(foil, batch)
        n = numobs(batch)
        loss, acc = val_step(model, batch)
        val_loss += loss * n
        val_acc += acc * n
        ntot += n
    end
    val = (; val_loss = val_loss / ntot, val_acc = val_acc / ntot)
    println("Epoch $epoch: $val")

    lr = lr_scheduler(epoch)
    Optimisers.adjust!(opt, lr)
end

