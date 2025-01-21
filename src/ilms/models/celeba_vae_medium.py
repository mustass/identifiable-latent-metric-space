from flax import nnx
import jax
from dataclasses import dataclass
import pickle
from jax.random import permutation, split
import optax
from jax.numpy import exp
from ..utils.stats import Stats

class ResidualBlock(nnx.Module):
    """Residual block with two convolutions and skip connection"""
    def __init__(self, channels, rngs):
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        # For conv layers, num_features is the number of channels
        self.norm1 = nnx.LayerNorm(channels,rngs=rngs)
        self.norm2 = nnx.LayerNorm(channels,rngs=rngs)
        
    def __call__(self, x):
        identity = x
        x = self.conv1(x)
        # Need to reshape for LayerNorm to treat channels as features
        x = self.norm1(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        x = jax.nn.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return jax.nn.gelu(x + identity)

class ResizeAndConv(nnx.Module):
    """Enhanced Resize-Conv Block with residual connections"""
    def __init__(self, in_channels, filters, kernel_size, strides, padding, rngs):
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv = nnx.Conv(
            self.in_channels, self.filters, self.kernel_size, self.padding, rngs=rngs
        )
        self.norm = nnx.LayerNorm(filters, rngs=rngs)  # num_features is output channels
        self.use_residual = in_channels == filters and strides == (1, 1)

    def __call__(self, x):
        identity = x
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
        # Reshape for LayerNorm
        x = self.norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        if self.use_residual:
            x = x + identity
        return jax.nn.gelu(x)

class VAE(nnx.Module):
    @dataclass
    class DefaultOpts:
        z_dim: int = 128
        num_decoders: int = 8

    class Decoder(nnx.Module):
        def __init__(self, opts, rngs):
            self.opts = opts
            # Reduced initial projection size
            self.fc_dec = nnx.Sequential(
                nnx.Linear(opts.z_dim, 512 * 4, rngs=rngs), 
                jax.nn.gelu
            )

            self.convs = nnx.Sequential(
                ResizeAndConv(
                    512, 256, kernel_size=(3, 3), strides=(1, 1), padding=1, rngs=rngs
                ),
                ResidualBlock(256, rngs=rngs),
                ResizeAndConv(
                    256, 192, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs
                ),
                ResidualBlock(192, rngs=rngs),
                ResizeAndConv(
                    192, 128, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs
                ),
                ResidualBlock(128, rngs=rngs),
                ResizeAndConv(
                    128, 96, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs
                ),
                ResidualBlock(96, rngs=rngs),
                ResizeAndConv(
                    96, 64, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs
                ),
                ResidualBlock(64, rngs=rngs),
                ResizeAndConv(
                    64, 32, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs
                ),
                ResidualBlock(32, rngs=rngs),
                # Final layer to RGB
                nnx.Conv(32, 3, kernel_size=(5, 5), padding=2, rngs=rngs),
            )

        def __call__(self, z):
            x_dec = self.fc_dec(z)
            x_dec = x_dec.reshape(x_dec.shape[0], 2, 2, 512)
            x_dec = self.convs(x_dec)
            return x_dec

    def __init__(self, opts={}, *, rngs: nnx.Rngs):
        self.stats = Stats()
        self.opts = self.DefaultOpts(**opts)
        z_dim = self.opts.z_dim

        self.rngs = rngs
        # Optimized encoder architecture
        self.encoder = nnx.Sequential(
            nnx.Conv(3, 32, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs),
            ResidualBlock(32, rngs=rngs),
            nnx.Conv(32, 64, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs),
            ResidualBlock(64, rngs=rngs),
            nnx.Conv(64, 128, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs),
            ResidualBlock(128, rngs=rngs),
            nnx.Conv(128, 256, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs),
            ResidualBlock(256, rngs=rngs),
            nnx.Conv(256, 512, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs),
            ResidualBlock(512, rngs=rngs),
        )

        self.enc_mu = nnx.Linear(2 * 2 * 512, z_dim, rngs=rngs)
        self.enc_logvar = nnx.Linear(2 * 2 * 512, z_dim, rngs=rngs)

        rngss = nnx.vmap(lambda s: nnx.Rngs(s), in_axes=0)(
            split(rngs(), self.opts.num_decoders)
        )
        self.decoder = nnx.vmap(self.Decoder, in_axes=(None, 0))(self.opts, rngss)

    # Rest of the methods remain unchanged
    def reparametrize(self, mu, logvar):
        std = jax.random.normal(self.rngs.reparam(), (mu.shape[0], mu.shape[1]))
        return mu + exp(0.5 * logvar) * std

    def encode(self, x, reparam=False):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        z_mu = self.enc_mu(x)
        if not reparam:
            return z_mu, z_mu, None
        
        z_logvar = self.enc_logvar(x)
        z = self.reparametrize(z_mu, z_logvar) if reparam else z_mu
        return z, z_mu, z_logvar

    def __call__(self, x, reparam=True):
        z, z_mu, z_logvar = self.encode(x, reparam)
        x_dec = self.decode(z)
        return x_dec, z_mu, z_logvar

    def decode(self, z):
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
        return decoded.transpose(1, 0, 2, 3, 4).reshape(batch_size, num_decoders, -1)

    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump(
                {"opts": self.opts, "stats": self.stats, "state": nnx.state(self)}, file
            )
    
    def load(self, path):
        with open(path, 'rb') as file:
            model_dict = pickle.load(file)
        self.opts = model_dict['opts']
        self.stats = model_dict['stats']
        nnx.update(self, model_dict['state'])