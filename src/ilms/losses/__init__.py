import jax.numpy as jnp
from ..models.celeba_vae import z_shape
from optax import l2_loss


class NelboLoss:
    def __init__(self, batch_size, kl_warmup_factor=1e-4, kl_end= 1.0):
        self.batch_size = batch_size
        self.kl_warmup_factor = kl_warmup_factor
        self.kl_end = kl_end

    def __call__(self, dec_mean, dec_logstd, enc_mean, enc_logstd, targets, step):
        MSE = jnp.sum(l2_loss(dec_mean, targets))/self.batch_size
        KLD = -0.5 * jnp.sum(1 + enc_logstd - jnp.pow(enc_mean,2) - jnp.exp(enc_logstd))
        loss = MSE + KLD* jnp.minimum(
             step.astype(jnp.float32) * self.kl_warmup_factor, self.kl_end
        )
        return loss, MSE, KLD