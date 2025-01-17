from optax import l2_loss
from jax.numpy import sum, exp


class NelboLoss:
    def __init__(self,  beta=1.0):
        self.beta = beta

    def __call__(self, model, batch):
        x_hat, z_mu, z_logvar = model(batch)

        rec_loss = l2_loss(x_hat, batch).sum([-1, -2, -3])
        kl_loss = -0.5 * sum(1.0 + z_logvar - z_mu**2 - exp(z_logvar), axis=-1)

        loss = rec_loss +  self.beta * kl_loss

        stats = {
            "elbo": -loss.mean(),
            "kl_loss": kl_loss.mean(),
            "rec_loss": rec_loss.mean(),
        }

        return loss.mean(), ((x_hat, z_mu, z_logvar), stats)
