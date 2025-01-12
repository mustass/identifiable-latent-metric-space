import distrax
import jax.numpy as jnp
from ..models.celeba_vae import z_shape


class NelboLoss:
    def __init__(self, batch_size=32, kl_warmup_factor=1e-4, kl_end= 1.0):
        self.prior = distrax.Normal(
            jnp.zeros(z_shape(batch_size)), jnp.ones(z_shape(batch_size))
        )
        self.kl_warmup_factor = kl_warmup_factor
        self.kl_end = kl_end

    def __call__(self, dec_mean, dec_logstd, enc_mean, enc_logstd, targets, step):

        #MSE =F.mse_loss(recon_x, x.view(-1, image_dim))
        MSE = optax.aljf
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kld_weight = 0.00025
        loss = MSE + kld_weight * KLD  
        return loss

       


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var):
    

#  likelihood = distrax.Normal(
#             dec_mean, jnp.exp(jnp.maximum(dec_logstd, -10.0))
#         ).log_prob(targets)
#         kl = distrax.Normal(
#             enc_mean, jnp.exp(jnp.maximum(enc_logstd, -10.0))
#         ).kl_divergence(self.prior)
#         denominator = jnp.prod(jnp.array(likelihood.shape)).astype(jnp.float32)
#         reconstuction_loss = -jnp.sum(likelihood) / denominator
#         kl_loss = jnp.sum(kl) / denominator  # increasing slowly the kl loss weight
#         neg_elbo = reconstuction_loss + kl_loss * jnp.minimum(
#             step.astype(jnp.float32) * self.kl_warmup_factor, self.kl_end
#         )  # ()
#         return neg_elbo, reconstuction_loss, kl_loss