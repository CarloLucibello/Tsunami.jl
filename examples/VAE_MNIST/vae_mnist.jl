# # Variational Autoencoder on Fashion MNIST dataset
# This example demonstrates how to train a Variational Autoencoder (VAE) on the Fashion MNIST dataset using Tsunami.

# ## Setup

using Optimisers: Optimisers, AdamW
using Tsunami
using MLDatasets
using MLUtils: MLUtils, DataLoader, flatten, mapobs, getobs, splitobs, randn_like
using Statistics, Random, LinearAlgebra
import ParameterSchedulers
using ImageShow, Plots
using Flux
using Flux: logitbinarycrossentropy
using ConcreteStructs: @concrete

# Uncomment one of the following lines for GPU support

## using CUDA
## using AMDGPU
## using Metal

# ## Model Definition

@concrete struct Encoder
    backbone
    mean_head
    var_head
end

Flux.@layer Encoder

function Encoder(; input_dim, hidden_dim, latent_dim)
    backbone = Chain(Dense(input_dim => hidden_dim, relu), 
                     Dense(hidden_dim => hidden_dim, relu))

    mean_head = Dense(hidden_dim => latent_dim)
    var_head = Dense(hidden_dim => latent_dim)
    return Encoder(backbone, mean_head, var_head)
end

function (m::Encoder)(x)
    h = m.backbone(x)
    μ = m.mean_head(h)
    logσ² = m.var_head(h)
    return μ, logσ²
end

@concrete struct Decoder
    layers
end

Flux.@layer Decoder

function Decoder(; latent_dim, hidden_dim, input_dim)
    layers = Chain(Dense(latent_dim => hidden_dim, relu), 
                    Dense(hidden_dim => hidden_dim, relu), 
                    Dense(hidden_dim => input_dim))
    return Decoder(layers)
end

(m::Decoder)(x) = m.layers(x)

struct VAE <: FluxModule
    λ::Float64 # L2 regularization parameter
    η::Float64 # learning rate
    β::Float64 # disentanglement parameter
    latent_dim::Int # dimension of latent space
    input_dim::Int  # dimension of input space
    encoder::Encoder
    decoder::Decoder
end

function VAE(; input_dim, latent_dim, hidden_dim, η = 1e-3, β = 1.0, λ=1e-4)
    encoder = Encoder(; input_dim, hidden_dim, latent_dim)       
    decoder = Decoder(; latent_dim, hidden_dim, input_dim)
    return VAE(λ, η, β, input_dim, latent_dim, encoder, decoder)
end

function (vae::VAE)(x)
    μ, logσ² = vae.encoder(x)
    z = reparametrize(μ, logσ²)
    x̂ = vae.decoder(z)
    return x̂, μ, logσ²
end

function reconstruct(vae::VAE, x)
    x̂, _, _ = vae(x)
    return sigmoid.(x̂)
end

function reparametrize(μ, logσ²)
    ε = randn_like(μ)
    z = μ .+ ε .* exp.(logσ² / 2)
    return z   
end

# Standard ELBO loss with the additional Beta hyperparameter to control disentanglement in latent bottleneck
function elbo_loss(x, x̂, μ, logσ², β=1f0)
    batch_size = numobs(x)
    β = Flux.ofeltype(x, β)
    # recon_loss = Flux.mse(x̂, x, agg=sum) / (2*batch_size)
    recon_loss = logitbinarycrossentropy(x̂, x, agg=sum) / batch_size
    kl = - sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / (2*batch_size)
    loss = recon_loss + β * kl
    return loss, recon_loss, kl
end

function Tsunami.configure_optimisers(model::VAE, trainer)
    lr_scheduler = ParameterSchedulers.Step(model.η, 0.1, fill(trainer.max_epochs ÷ 4, 4))
    opt = Optimisers.setup(Optimisers.AdamW(lambda=model.λ), model)
    return opt, lr_scheduler
end

function Tsunami.train_step(model::VAE, trainer, batch, batch_idx)
    x = batch
    x̂, μ, logσ² = model(x)
    loss, recon_loss, kl_loss = elbo_loss(x, x̂, μ, logσ², model.β)
    Tsunami.log(trainer, "loss/train", loss)
    Tsunami.log(trainer, "recon_loss/train", recon_loss)
    Tsunami.log(trainer, "kl_loss/train", kl_loss)
    return loss
end

function Tsunami.val_step(model::VAE, trainer, batch)
    x = batch
    x̂, μ, logσ² = model(x)
    loss, recon_loss, kl_loss = elbo_loss(x, x̂, μ, logσ², model.β)
    
    Tsunami.log(trainer, "loss/val", loss)
    Tsunami.log(trainer, "recon_loss/val", recon_loss)
    Tsunami.log(trainer, "kl_loss/val", kl_loss)
end

function Tsunami.test_step(model::VAE, trainer, batch)
    x = batch
    x̂, μ, logσ² = model(x)
    loss, recon_loss, kl_loss = elbo_loss(x, x̂, μ, logσ², model.β)
    Tsunami.log(trainer, "loss/test", loss)
    Tsunami.log(trainer, "recon_loss/test", recon_loss)
    Tsunami.log(trainer, "kl_loss/test", kl_loss)
end

# ## Callbacks

struct SaveImgsCallback end

function sample(model::VAE, nimgs::Int)
    z = randn(Float32, latent_dim, nimgs)
    x̂ = sigmoid.(model.decoder(z))
    imgs = reshape(x̂, 28, 28, nimgs)
    return MLDatasets.convert2image(FashionMNIST, imgs)
end

function Tsunami.on_train_epoch_end(::SaveImgsCallback, model::VAE, trainer)
    imgs = sample(model, 10)
    plot([plot(getobs(imgs, i)) for i=1:10]..., layout=(1,10), 
        framestyle=:none, axis=nothing, margin=0Plots.mm,
        size=(1000, 100))
    fit_state = trainer.fit_state
    savefig(joinpath(fit_state.run_dir, "samples_epoch=$(fit_state.epoch).png"))
end


# ## Data Preparation

train_data = mapobs(batch -> flatten(batch[1]), FashionMNIST(:train))
train_data, val_data = splitobs(train_data, at = 0.9)
test_data = mapobs(batch -> flatten(batch[1]), FashionMNIST(:test))

train_loader = DataLoader(train_data, batchsize=128, shuffle=true)
val_loader = DataLoader(val_data, batchsize=128, shuffle=true)
test_loader = DataLoader(test_data, batchsize=128)

# ## Training

latent_dim = 64
hidden_dim = 512
λ = 1e-4                # regularization paramater
model = VAE(; input_dim = 28*28, latent_dim, hidden_dim, η = 1e-3, β = 1.0)

trainer = Trainer(max_epochs = 10,
                 max_steps = -1,
                 default_root_dir = @__DIR__,
                 accelerator = :cpu,
                 callbacks = [SaveImgsCallback()])

Tsunami.fit!(model, trainer, train_loader, val_loader)

# ## Evaluation

# ### Test Likelihood

test_results = Tsunami.test(model, trainer, test_loader)

# ### Sampling

imgs = sample(model, 10)
