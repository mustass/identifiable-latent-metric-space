import jax, optax, sys, time, pickle, builtins, os
import pandas as pd
import numpy as np
from jax.numpy import *
from jax.random import permutation, split
from flax import nnx
from dataclasses import dataclass, fields
from ..utils.stats import Stats
from functools import partial

class ResizeAndConv(nnx.Module):
    """
    Resize-Conv Block.

    A simple Nearest-Neighbord upsampling + Conv block, used to upsample images instead of Deconv layers.
    This block is useful to avoid checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, filters, kernel_size, strides, rngs):
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv = nnx.Conv(self.in_channels, self.filters, self.kernel_size, (1,1), rngs=rngs)

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
        z_dim: int = 64        # latent dimensionality
        num_decoders: int = 8  # number of Decoders
    
    class Decoder(nnx.Module):
        def __init__(self, opts, rngs):
            self.opts = opts
            self.fc_dec = nnx.Sequential(
                nnx.Linear(opts.z_dim, 256, rngs=rngs),    nnx.elu,
                nnx.Linear(256, 64 * 8 * 8, rngs=rngs), nnx.elu
            )

            self.convs = nnx.Sequential(
                ResizeAndConv(64, 64, kernel_size=(4, 4), strides=(1, 1), rngs=rngs), nnx.elu,
                ResizeAndConv(64, 64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs), nnx.elu,
                ResizeAndConv(64, 64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs), nnx.elu,
                ResizeAndConv(64, 32, kernel_size=(4, 4), strides=(2, 2), rngs=rngs), nnx.elu,
                ResizeAndConv(32, 3,  kernel_size=(4, 4), strides=(1, 1), rngs=rngs),
            )

        def __call__(self, z):
            x_dec = self.fc_dec(z)
            x_dec = x_dec.reshape(x_dec.shape[0], 8, 8, 64)
            x_dec = self.convs(x_dec)
            return x_dec

    def __init__(self, opts={}, *, rngs: nnx.Rngs):
        self.stats = Stats()
        self.opts = self.DefaultOpts(**opts)
        z_dim = self.opts.z_dim

        self.rngs = rngs
        self.encoder = nnx.Sequential(
            nnx.Conv(3, 128,    kernel_size=(4, 4), strides=(1,1), rngs=rngs), nnx.elu,
            nnx.Conv(128, 128,  kernel_size=(4, 4), strides=(2,2), rngs=rngs), nnx.elu,
            nnx.Conv(128, 256,  kernel_size=(4, 4), strides=(2,2), rngs=rngs), nnx.elu,
            nnx.Conv(256, 256,  kernel_size=(4, 4), strides=(2,2), rngs=rngs), nnx.elu,
            nnx.Conv(256, 256,  kernel_size=(4, 4), strides=(1,1), rngs=rngs), nnx.elu,
        )
        
        self.enc = nnx.Linear(4*4*1024, 256, rngs=rngs)
        self.enc_mu = nnx.Linear(256, z_dim, rngs=rngs)
        self.enc_logvar = nnx.Linear(256, z_dim, rngs=rngs)
        
        rngss = nnx.vmap(lambda s: nnx.Rngs(s), in_axes=0)(split(rngs(), self.opts.num_decoders))
        self.decoder = nnx.vmap(self.Decoder, in_axes=(None, 0))(self.opts, rngss)
    
    def encode(self, x, reparam=False):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.enc(x)
        x = nnx.elu(x)
        z_mu = self.enc_mu(x)
        if not reparam:
            return z_mu, z_mu, None,
        
        z_logvar = self.enc_logvar(x)
        z = self.reparametrize(z_mu, z_logvar) if reparam else z_mu
        return z, z_mu, z_logvar

    def reparametrize(self, mu, logvar):
        std = jax.random.normal(self.rngs.reparam(), (mu.shape[0], mu.shape[1]))
        return mu + exp(0.5 * logvar) * std

    def __call__(self, x, reparam=True):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.enc(x)
        x = nnx.elu(x)
        z_mu = self.enc_mu(x)
        z_logvar = self.enc_logvar(x)

        z = self.reparametrize(z_mu, z_logvar) if reparam else z_mu
        x_dec = self.decode(z)

        return x_dec, z_mu, z_logvar

    def decode(self, z):
        # split z into nD parts and decode each part
        z = z.reshape(self.opts.num_decoders, z.shape[0] // self.opts.num_decoders, z.shape[1])
        decoded = nnx.vmap(lambda z, d: d(z), in_axes=(0, 0))(z, self.decoder)
        decoded = decoded.reshape(-1, *decoded.shape[2:])
        return decoded
    
    def decode_ensemble(self, z):
        decoded = nnx.vmap(lambda z, d: d(z), in_axes=(None, 0))(z, self.decoder)
        num_decoders = decoded.shape[0]
        batch_size = decoded.shape[1]
        # returning flattened image
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