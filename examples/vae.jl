using LinearAlgebra, Random, Statistics
using Flux, Tsunami

# Standard ELBO loss with the additional Beta hyperparameter to control disentanglement in latent bottleneck
function elbo_loss(x, x̂, μ, logσ², β=1f0)
    β = Flux.ofeltype(x, β)
    recon_loss = Flux.mse(x̂, x)
    kl = - mean(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / 2
    loss = recon_loss + β * kl
    return loss, recon_loss, kl
end

mutable struct VAE <: FluxModule
    η::Float64
    β::Float64
    latent_dim::Int
    input_dim::Int
    encoder
    decoder
end

function VAE(; input_dim, latent_dim, hidden_dim, η, loss_fn, β)
    
    encoder = Chain(Dense(input_dim => hidden_dim, relu), 
                    Dense(hidden_dim => hidden_dim, relu),
                    Split(
                        Dense(hidden_dim => latent_dim), # mean
                        Dense(hidden_dim => latent_dim) # logvar
                        )
                    )
                                    
    decoder = Chain(Dense(latent_dim => hidden_dim, relu), 
                    Dense(hidden_dim => hidden_dim, relu), 
                    Dense(hidden_dim => input_dim, softplus))
    
    return VAE(η, β, input_dim, latent_dim, encoder, decoder)
end

function encode(m::VAE, x)
    μ, logσ² = m.encoder(x)
    return μ, logσ²
end

function decode(m::VAE, z)
    return m.decoder(x)    
end
    
    
    def reparameterize(self, mean, std):
        epsilon = torch.randn_like(std).to(self.device)      
        z = mean + std*epsilon
        return z

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, torch.exp(0.5 * logvar))
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW( # Adam with weight decay (in PyT this is effectively L2 reg)
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.lr*1e-2
        )

        scheduler = torch.optim.lr_scheduler.StepLR( # Decrease lr by 10% each time 1/4 of the total training passes
            optimizer, 
            step_size=self.trainer.max_epochs//4, 
            gamma=0.1
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x = batch

        x_hat, mean, logvar = self.forward(x)

        if self.loss_fn == 'mmd':
            prior_sample = torch.randn(200, self.latent_dim).to(self.device)
            loss, recon_loss, prior = mmd_loss(x, x_hat, prior_sample, mean)
        else:
            loss, recon_loss, prior = elbo_loss(x, x_hat, mean, logvar, self.beta)
        
        self.log('train_loss', loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_prior_loss', prior)

        # At the end of each epoch, visualize the reconstruction quality and latent mean embedding
        if batch_idx == self.trainer.num_training_batches-1:
            fig = plt.figure()
            plt.scatter(x.detach().cpu().numpy()[0], x_hat.detach().cpu().numpy()[0])
            plt.title('true (x) vs reconstructed (y) values from')
            plt.tight_layout()
            self.logger.experiment.log({"train_reconstruction": wandb.Image(fig)})
            plt.close()

        return loss

    def on_validation_epoch_start(self):
        self.num_val_batches = 0
        self.overall_val_loss = 0
        self.overall_val_recon_loss = 0
        self.overall_val_prior_loss = 0

    def validation_step(self, batch, batch_idx):
        x = batch

        x_hat, mean, logvar = self.forward(x)

        if self.loss_fn == 'mmd':
            prior_sample = torch.randn(200, self.latent_dim).to(self.device)
            loss, recon_loss, prior = mmd_loss(x, x_hat, prior_sample, mean)
        else:
            loss, recon_loss, prior = elbo_loss(x, x_hat, mean, logvar, self.beta)

        self.log('val_loss', loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_prior_loss', prior)

        self.num_val_batches += 1
        self.overall_val_loss += loss.item()
        self.overall_val_recon_loss += recon_loss.item()
        self.overall_val_prior_loss += prior.item()

        # At the end of each epoch, visualize the reconstruction quality and latent mean embedding
        if batch_idx == self.trainer.num_val_batches[0]-1:
            fig = plt.figure()
            plt.scatter(x.detach().cpu().numpy(), x_hat.detach().cpu().numpy())
            plt.title('true (x) vs reconstructed (y) values (val)')
            plt.tight_layout()
            self.logger.experiment.log({"val_reconstruction": wandb.Image(fig)})
            plt.close()

    def on_validation_epoch_end(self):
        # Print average performance on command line
        print('Avg. loss', self.overall_val_loss/self.num_val_batches)
        print('Avg. reconstruction loss', self.overall_val_recon_loss/self.num_val_batches)
        print('Avg. prior loss', self.overall_val_prior_loss/self.num_val_batches)