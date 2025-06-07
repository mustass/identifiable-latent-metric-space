from optax import l2_loss
from jax.numpy import sum, exp, stack, ones
from jax.random import PRNGKey
from lpips_j.lpips import VGGExtractor
import flax.linen as nn


class NelboLoss:
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, model, batch):
        x_hat, z_mu, z_logvar = model(batch)

        rec_loss = l2_loss(x_hat, batch).sum([-1, -2, -3])
        kl_loss = -0.5 * sum(1.0 + z_logvar - z_mu**2 - exp(z_logvar), axis=-1)

        loss = rec_loss + self.beta * kl_loss

        stats = {
            "elbo": -loss.mean(),
            "kl_loss": kl_loss.mean(),
            "rec_loss": rec_loss.mean(),
        }

        return loss.mean(), ((x_hat, z_mu, z_logvar), stats)


class LPIPSFIX(nn.Module):
    def setup(self):
        self.vgg = VGGExtractor()

    def __call__(self, x, t):
        x = self.vgg(x)
        t = self.vgg(t)

        # conv_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
        #               'conv3_2', 'conv3_3', 'conv3_3', 'conv4_1', 'conv4_2',
        #               'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

        # starting CONV layers are more important for perceptual similarity
        # and produces better results. Including later layers leads to worse results.

        conv_names = [
            "conv1_1",
            "conv1_2",
            "conv2_1",
            "conv2_2",
            "conv3_1",
            "conv3_2",
            "conv3_3",
            "conv3_3",
        ]

        diffs = []
        for f in conv_names:
            diff = (x[f] - t[f]) ** 2
            diff = 0.5 * diff.mean([1, 2, 3])
            diffs.append(diff)

        return stack(diffs, axis=1).sum(axis=1)


class PRCLoss:
    def __init__(self, beta=1.0, image_shape=[64, 64, 3]):
        self.beta = beta
        self.lpips_obj = lpips_obj = LPIPSFIX()
        example = ones(image_shape)
        self.lpips_params = lpips_obj.init(PRNGKey(0), example, example)
        # model.rngs.dinfar(), example, example)

    def __call__(self, model, batch):
        x_hat, z_mu, z_logvar = model(batch)

        kl_loss = -0.5 * sum(1.0 + z_logvar - z_mu**2 - exp(z_logvar), axis=-1)

        rec_loss = l2_loss(x_hat, batch).sum([-1, -2, -3])
        prc_loss = self.lpips_obj.apply(self.lpips_params, batch, x_hat)

        loss = rec_loss + prc_loss + self.beta * kl_loss

        stats = {
            "elbo": -loss.mean(),
            "kl_loss": kl_loss.mean(),
            "rec_loss": rec_loss.mean(),
            "prc_loss": prc_loss.mean(),
            "beta": self.beta,
        }

        return loss.mean(), ((x_hat, z_mu, z_logvar), stats)

        # x_hat, z_mu, z_logvar = model(batch)

        # prc_loss = self.lpips_obj.apply(self.lpips_params, batch, x_hat, breakp=True)

        # stats = {
        #     "prc_loss": prc_loss.mean(),
        # }

        # return prc_loss.mean(), ((x_hat, z_mu, z_logvar), stats)


# def loss_fn(model, batch, current_epoch=512, lpips_obj = None, lpips_params = None):
#     x_hat, z_mu, z_logvar = model(batch)

#     kl_loss = -0.5 * sum(1.0 + z_logvar - z_mu**2 - exp(z_logvar), axis=-1)

#     rec_loss = optax.l2_loss(x_hat, batch).sum([-1, -2, -3]) #array(0.0)
#     prc_loss = lpips_obj.apply(lpips_params, batch, x_hat, breakp=True) # array(0.0)

#     beta = array(1.0) # scaled_sigmoid(current_epoch, model.opts.epochs)

#     # breakpoint()
#     loss = rec_loss + prc_loss + beta * kl_loss

#     stats = {
#         "elbo": -loss.mean(),
#         "kl_loss": kl_loss.mean(),
#         "rec_loss": rec_loss.mean(),
#         "prc_loss": prc_loss.mean(),
#         "beta": beta,
#     }

#     return loss.mean(), ((x_hat, z_mu, z_logvar), stats)
