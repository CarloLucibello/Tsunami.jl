"""
    abstract type FluxModule end

An abstract type for Flux models.
A `FluxModule` helps orgainising you code and provides a standard interface for training.

A `FluxModule` comes with the functionality provided by `Flux.@layer` 
(pretty printing, etc...) and the ability to interact with 
[`Trainer`](@ref) and `Optimisers.jl`.

You can change the trainables by implementing `Optimisers.trainables`.

Types subtyping from `FluxModule` have to implement the following methods 
in order to interact with a [`Trainer`](@ref).

# Required methods

- [`configure_optimisers`](@ref)`(model, trainer)`.
- [`train_step`](@ref)`(model, trainer, batch, [batch_idx])`.

# Optional Methods 

- [`val_step`](@ref)`(model, trainer, batch, [batch_idx])`.
- [`test_step`](@ref)`(model, trainer, batch, [batch_idx])`.
- generic [hooks](@ref Hooks).

# Examples

```julia
using Flux, Tsunami, Optimisers

# Define a Multilayer Perceptron implementing the FluxModule interface

struct Model <: FluxModule
    net
end

function Model()
    net = Chain(Dense(4 => 32, relu), Dense(32 => 2))
    return Model(net)
end

(model::Model)(x) = model.net(x)

function Tsunami.train_step(model::Model, trainer, batch)
    x, y = batch
    y_hat = model(x)
    loss = Flux.Losses.mse(y_hat, y)
    return loss
end

function Tsunami.configure_optimisers(model::Model, trainer)
    return Optimisers.setup(Optimisers.Adam(1f-3), model)
end

# Prepare the dataset and the DataLoader
X, Y = rand(4, 100), rand(2, 100)
train_dataloader = Flux.DataLoader((X, Y), batchsize=10)

# Create and Train the model
model = Model()
trainer = Trainer(max_epochs=10)
Tsunami.fit!(model, trainer, train_dataloader)
```
"""
abstract type FluxModule end

Flux.@layer FluxModule

Base.show(io::IO, m::FluxModule) = shortshow(io, m)

not_implemented_error(name) = error("You need to implement the method `$(name)`")

"""
    configure_optimisers(model, trainer)

Return an optimiser's state initialized for the `model`.
It can also return a tuple of `(optimiser, scheduler)`,
where `scheduler` is any callable object that takes 
the current epoch as input and returns a scalar that will be 
set as the learning rate for the next epoch.

# Examples

```julia
using Optimisers, ParameterSchedulers

function Tsunami.configure_optimisers(model::Model, trainer)
    return Optimisers.setup(AdamW(1f-3), model)
end

# Now with a scheduler dropping the learning rate by a factor 10 
# at epochs [50, 100, 200] starting from the initial value of 1e-2
function Tsunami.configure_optimisers(model::Model, trainer)

    function lr_scheduler(epoch)
        if epoch <= 50
            return 1e-2
        elseif epoch <= 100
            return 1e-3
        elseif epoch <= 200
            return 1e-4
        else
            return 1e-5
        end
    end
    
    opt_state = Optimisers.setup(AdamW(), model)
    return opt_state, lr_scheduler
end

# Same as above but using the ParameterSchedulers package.
function Tsunami.configure_optimisers(model::Model, trainer)
    lr_scheduler = ParameterSchedulers.Step(1f-2, 0.1f0, [50, 50, 100])
    opt_state = Optimisers.setup(AdamW(), model)
    return opt_state, lr_scheduler
end
```
"""
function configure_optimisers(model::FluxModule, trainer)
    not_implemented_error("configure_optimisers(model, trainer)")
end

"""
    train_step(model, trainer, batch, [batch_idx])

The method called at each training step during `Tsunami.fit!`.
It should compute the forward pass of the model and return the loss 
(a scalar) corresponding to the minibatch `batch`. 
The optional argument `batch_idx` is the index of the batch in the current epoch.

Any `Model <: FluxModule` should implement either 
`train_step(model::Model, trainer, batch)` or `train_step(model::Model, trainer, batch, batch_idx)`.

The training loop in `Tsunami.fit!` approximately looks like this:
```julia
for epoch in 1:epochs
    for (batch_idx, batch) in enumerate(train_dataloader)
        grads = gradient(model) do m
            loss = train_step(m, trainer, batch, batch_idx)
            return loss
        end
        Optimisers.update!(opt, model, grads[1])
    end
end
```

The output can be either a scalar or a named tuple:

- If a scalar is returned, it is assumed to be the loss.
- If a named tuple is returned, it has to contain the `loss` field. 

The output can be accessed in hooks such as [`on_before_update`](@ref) or [`on_train_batch_end`](@ref).

# Examples

```julia
function Tsunami.train_step(model::Model, trainer, batch)
    x, y = batch
    ŷ = model(x)
    loss = Flux.Losses.logitcrossentropy(ŷ, y)
    Tsunami.log(trainer, "loss/train", loss)
    Tsunami.log(trainer, "accuracy/train", Tsunami.accuracy(ŷ, y))
    return loss
end
```
"""
train_step(model::FluxModule, trainer, batch, batch_idx) = train_step(model, trainer, batch)

function train_step(model::FluxModule, trainer, batch)
    not_implemented_error("train_step(model, trainer, batch)")
end

"""
    val_step(model, trainer, batch, [batch_idx])

The method called at each validation step during `Tsunami.fit!`.
Tipically used for computing metrics and statistics on the validation 
batch `batch`. The optional argument `batch_idx` is the index of the batch in the current 
validation epoch. 

A `Model <: FluxModule` should implement either 
`val_step(model::Model, trainer, batch)` or `val_step(model::Model, trainer, batch, batch_idx)`.

Optionally, the method can return a scalar or a named tuple, to be used in hooks such as 
[`on_val_batch_end`](@ref).

See also [`train_step`](@ref).

# Examples
    
```julia
function Tsunami.val_step(model::Model, trainer, batch)
    x, y = batch
    ŷ = model(x)
    loss = Flux.Losses.logitcrossentropy(ŷ, y)
    accuracy = Tsunami.accuracy(ŷ, y)
    Tsunami.log(trainer, "loss/val", loss, on_step = false, on_epoch = true)
    Tsunami.log(trainer, "loss/accuracy", accuracy, on_step = false, on_epoch = true)
end
```
"""
val_step(model::FluxModule, trainer, batch, batch_idx) = val_step(model, trainer, batch)

function val_step(model::FluxModule, trainer, batch)
    # not_implemented_error("val_step")
    return nothing
end

"""
    test_step(model, trainer, batch, [batch_idx])

Similard to [`val_step`](@ref) but called at each test step.
"""
test_step(model::FluxModule, trainer, batch, batch_idx) = test_step(model, trainer, batch)

function test_step(model::FluxModule, trainer, batch)
    # not_implemented_error("test_step")
    return nothing 
end

check_train_step(m::EnzymeCore.Duplicated, args...) = check_train_step(m.val, args...)

function check_train_step(m::FluxModule, trainer, batch)
    out = train_step(m, trainer, batch, 1)
    @assert out isa Union{Number,NamedTuple} "The output of `train_step` has to be a scalar or named tuple."
    if out isa NamedTuple
        @assert haskey(out, :loss) "A named tuple output of `train_step` has to contain the `loss` field."
    end
end

check_val_step(m::EnzymeCore.Duplicated, args...) = check_val_step(m.val, args...)

function check_val_step(m::FluxModule, trainer, batch)
    val_step(m, trainer, batch, 1)
    @assert true
end

