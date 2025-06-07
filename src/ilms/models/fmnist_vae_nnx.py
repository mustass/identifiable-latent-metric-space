from flax import nnx
import jax
from dataclasses import dataclass
import pickle
from jax.random import permutation, split
import optax
from jax.numpy import exp
from ..utils.stats import Stats


class ResizeAndConv(nnx.Module):
    """
    Resize-Conv Block.

    A simple Nearest-Neighbord upsampling + Conv block, used to upsample images instead of Deconv layers.
    This block is useful to avoid checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, filters, kernel_size, strides, padding, rngs):
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv = nnx.Conv(
            self.in_channels, self.filters, self.kernel_size, self.padding, rngs=rngs
        )

    def __call__(self, x):
        if self.strides != (1, 1):
            x = jax.image.resize(
                x,
                (
                    x.shape[0],
                    x.shape[1] * self.strides[0],
                    x.shape[2] * self.strides[1],
                    x.shape[3],
                ),
                method="nearest",
            )

        x = self.conv(x)
        return x


class VAE(nnx.Module):
    @dataclass
    class DefaultOpts:
        z_dim: int = 128  # latent dimensionality
        num_decoders: int = 8  # number of Decoders

    class Decoder(nnx.Module):
        def __init__(self, opts, rngs):
            self.opts = opts
            self.fc_dec = nnx.Sequential(
                nnx.Linear(opts.z_dim, 256 * 4, rngs=rngs), nnx.elu
            )

            self.convs = nnx.Sequential(
                ResizeAndConv(
                    256, 128, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs
                ),
                nnx.elu,
                ResizeAndConv(
                    128, 64, kernel_size=(3, 3), strides=(1, 1), padding=1, rngs=rngs
                ),
                nnx.elu,
                ResizeAndConv(
                    64, 32, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs
                ),
                nnx.elu,
                ResizeAndConv(
                    32, 16, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs
                ),
                nnx.elu,
                ResizeAndConv(
                    16, 1, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs
                ),  # Upsample to 32x32
                nnx.elu,
            )
            self.final_resize = lambda x: jax.image.resize(
                x,
                (x.shape[0], 28, 28, 1),
                method="nearest",
            )

        def __call__(self, z):
            x_dec = self.fc_dec(z)
            x_dec = x_dec.reshape(x_dec.shape[0], 2, 2, 256)
            x_dec = self.convs(x_dec)
            x_dec = self.final_resize(x_dec)
            return x_dec

    def __init__(self, opts={}, *, rngs: nnx.Rngs):
        self.stats = Stats()
        self.opts = self.DefaultOpts(**opts)
        z_dim = self.opts.z_dim

        self.rngs = rngs
        self.encoder = nnx.Sequential(
            nnx.Conv(
                1, 16, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs
            ),  # 28 → 14
            nnx.elu,
            nnx.Conv(
                16, 32, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs
            ),  # 14 → 7
            nnx.elu,
            nnx.Conv(
                32, 64, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs
            ),  # 7 → 4
            nnx.elu,
            nnx.Conv(
                64, 128, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs
            ),  # 4 → 2
            nnx.elu,
            nnx.Conv(
                128, 256, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs
            ),  # 4 → 2
            nnx.elu,
        )

        self.enc_mu = nnx.Linear(256, z_dim, rngs=rngs)
        self.enc_logvar = nnx.Linear(256, z_dim, rngs=rngs)

        rngss = nnx.vmap(lambda s: nnx.Rngs(s), in_axes=0)(
            split(rngs(), self.opts.num_decoders)
        )
        self.decoder = nnx.vmap(self.Decoder, in_axes=(None, 0))(self.opts, rngss)

    def reparametrize(self, mu, logvar):

        std = jax.random.normal(self.rngs.reparam(), (mu.shape[0], mu.shape[1]))
        return mu + exp(0.5 * logvar) * std

    def encode(self, x, reparam=False):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        z_mu = self.enc_mu(x)
        if not reparam:
            return (
                z_mu,
                z_mu,
                None,
            )

        z_logvar = self.enc_logvar(x)
        z = self.reparametrize(z_mu, z_logvar) if reparam else z_mu
        return z, z_mu, z_logvar

    def __call__(self, x, reparam=True):
        z, z_mu, z_logvar = self.encode(x, reparam)
        x_dec = self.decode(z)

        return x_dec, z_mu, z_logvar

    def decode(self, z):
        # split z into num_decoders parts and decode each part
        z = z.reshape(
            self.opts.num_decoders, z.shape[0] // self.opts.num_decoders, z.shape[1]
        )
        decoded = nnx.vmap(lambda z, d: d(z), in_axes=(0, 0))(z, self.decoder)
        decoded = decoded.reshape(-1, *decoded.shape[2:])
        return decoded

    def decode_ensemble(self, z):
        decoded = nnx.vmap(lambda z, d: d(z), in_axes=(None, 0))(z, self.decoder)
        num_decoders = decoded.shape[0]
        batch_size = decoded.shape[1]
        return decoded.transpose(1, 0, 2, 3, 4).reshape(
            batch_size, num_decoders, -1
        )  # returning flattened image

    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump(
                {"opts": self.opts, "stats": self.stats, "state": nnx.state(self)}, file
            )

    def load(self, path):
        with open(path, "rb") as file:
            model_dict = pickle.load(file)
        self.opts = model_dict["opts"]
        self.stats = model_dict["stats"]
        nnx.update(self, model_dict["state"])
