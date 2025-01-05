import distrax
import jax.numpy as jnp
from ..models.celeba_vae import z_shape


class NelboLoss:
    def __init__(self, batch_size=32):
        self.prior = distrax.Normal(
            jnp.zeros(z_shape(batch_size)), jnp.ones(z_shape(batch_size))
        )

    def __call__(self, dec_mean, dec_logstd, enc_mean, enc_logstd, targets, step):
        likelihood = distrax.Normal(
            dec_mean, jnp.exp(jnp.maximum(dec_logstd, -10.0))
        ).log_prob(targets)
        kl = distrax.Normal(
            enc_mean, jnp.exp(jnp.maximum(enc_logstd, -10.0))
        ).kl_divergence(self.prior)
        denominator = jnp.prod(jnp.array(likelihood.shape)).astype(jnp.float32)
        reconstuction_loss = -jnp.sum(likelihood) / denominator
        kl_loss = (jnp.sum(kl) / denominator) * jnp.minimum(
            step.astype(jnp.float32) * 1e-4, 1.0
        )  # increasing slowly the kl loss weight
        neg_elbo = reconstuction_loss + kl_loss  # ()
        return neg_elbo, reconstuction_loss, kl_loss
