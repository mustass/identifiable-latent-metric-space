import jax, optax, sys, time, pickle, builtins, os
import pandas as pd
import numpy as np
from jax.numpy import *
from flax import nnx

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

if 'train' not in locals():
    train_fname = "/data/celeba/celeba_train_images.npy"
    test_fname = "/data/celeba/celeba_test_images.npy"
    train = np.load(train_fname, mmap_mode='r')
    train = jax.device_put(train)
    # test  = np.load(test_fname, mmap_mode='r')
    # test  = jax.device_put(test)

class ResizeAndConv(nnx.Module):
    """
    Resize-Conv Block.

    A simple Nearest-Neighbord upsampling + Conv block, used to upsample images instead of Deconv layers.
    This block is useful to avoid checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, filters, kernel_size, stride, rngs):
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nnx.Conv(self.in_channels, self.filters, self.kernel_size, (1, 1), rngs=rngs)

    def __call__(self, x):
        if self.stride != (1, 1):
            x = jax.image.resize(
                x,
                (
                    x.shape[0],
                    x.shape[1] * self.stride[0],
                    x.shape[2] * self.stride[1],
                    x.shape[3],
                ),
                method="nearest",
            )
        
        x = self.conv(x)
        return x

class VAE(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        z_dim = 32

        self.rngs = rngs
        self.encoder = nnx.Sequential(
            nnx.Conv(3, 128, kernel_size=(4, 4), strides=(1, 1), rngs=rngs),
            nnx.elu,
            nnx.Conv(128, 128, kernel_size=(4, 4), strides=(2, 2), rngs=rngs),
            nnx.elu,
        )
        
        self.enc_mu = nnx.Linear(32*32*128, z_dim, rngs=rngs)
        self.enc_logvar = nnx.Linear(32*32*128, z_dim, rngs=rngs)

        # self.fc_dec = 
        
        self.fc_dec = nnx.Sequential(
            nnx.Linear(z_dim, 4*4*512, rngs=rngs),
            nnx.elu
        )

        self.decoder = nnx.Sequential(
            ResizeAndConv(128, 128, (4, 4), (1, 1), rngs=rngs),
            nnx.elu,
            ResizeAndConv(128, 128, (4, 4), (2, 2), rngs=rngs),
            nnx.elu,
            ResizeAndConv(128, 128, (4, 4), (2, 2), rngs=rngs),
            nnx.elu,
            ResizeAndConv(128, 128, (4, 4), (2, 2), rngs=rngs),
            nnx.elu,
            ResizeAndConv(128, 3,   (4, 4), (1, 1), rngs=rngs),
        )

    def __call__(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        
        x_dec = self.fc_dec(mu)
        x_dec = x_dec.reshape(x_dec.shape[0], 8, 8, 128)
        x_dec = self.decoder(x_dec)

        # breakpoint()

        return mu, logvar, x_dec
    
vae = VAE(rngs=nnx.Rngs(jax.random.PRNGKey(0)))
z = vae(train[:1])
