import jax.numpy as jnp
from ..models.celeba_vae_linen import z_shape
from optax import l2_loss
from jax.lax import clamp


class NelboLoss:
    def __init__(self, batch_size, kl_warmup_factor=1e-4, kl_end= 1.0):
        self.batch_size = batch_size
        self.kl_warmup_factor = kl_warmup_factor
        self.kl_end = kl_end

    def __call__(self, dec_mean, enc_mean, enc_logstd, targets, step):
        # clamp the output logstd
        enc_logstd = clamp(-10.0,enc_logstd,10.0)
        rec_loss = l2_loss(dec_mean, targets).sum([-1,-2,-3])
        kl_loss = -0.5 * jnp.sum(1 + enc_logstd - jnp.pow(enc_mean,2) - jnp.exp(enc_logstd), axis=-1)
        loss = rec_loss + kl_loss* jnp.minimum(
             step.astype(jnp.float32) * self.kl_warmup_factor, self.kl_end
        )
        return loss.mean(), rec_loss, kl_loss