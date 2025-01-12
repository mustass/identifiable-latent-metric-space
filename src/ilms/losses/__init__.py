import jax.numpy as jnp
from ..models.celeba_vae import z_shape
from optax import l2_loss


class NelboLoss:
    def __init__(self, kl_warmup_factor=1e-4, kl_end= 1.0):
        self.kl_warmup_factor = kl_warmup_factor
        self.kl_end = kl_end

    def __call__(self, dec_mean, dec_logstd, enc_mean, enc_logstd, targets, step):
        MSE = jnp.sum(jnp.mean(l2_loss(dec_mean, targets), axis=0))
        KLD = -0.5 * jnp.mean(jnp.sum(1 + enc_logstd - jnp.pow(enc_mean,2) - jnp.exp(enc_logstd), axis=1))
        loss = MSE + KLD* jnp.minimum(
             step.astype(jnp.float32) * self.kl_warmup_factor, self.kl_end
        )
        return loss, MSE, KLD