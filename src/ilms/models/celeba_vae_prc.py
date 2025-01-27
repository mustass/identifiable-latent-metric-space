import jax, optax, sys, time, pickle, builtins, os
import pandas as pd
import numpy as np
from jax.numpy import *
from jax.random import permutation, split
from flax import nnx
from dataclasses import dataclass, fields
# from eugene.stats import Stats
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
        # epochs: int = 512       # Number of epochs to train for
        # bs: int = 128           # batch size
        # lr: float = 1e-5        # learnig rate
        z_dim: int = 64           # latent dimensionality
        num_decoders: int = 8             # number of Decoders
    
    class Decoder(nnx.Module):
        def __init__(self, opts, rngs):
            self.opts = opts
            self.fc_dec = nnx.Sequential(
                nnx.Linear(opts.z_dim, 256, rngs=rngs),    nnx.relu,
                nnx.Linear(256, 64 * 8 * 8, rngs=rngs), nnx.relu
            )

            self.convs = nnx.Sequential(
                ResizeAndConv(64, 64, kernel_size=(4, 4), strides=(1, 1), rngs=rngs), nnx.relu,
                ResizeAndConv(64, 64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs), nnx.relu,
                ResizeAndConv(64, 64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs), nnx.relu,
                ResizeAndConv(64, 32, kernel_size=(4, 4), strides=(2, 2), rngs=rngs), nnx.relu,
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
            nnx.Conv(3, 128,    kernel_size=(4, 4), strides=(1,1), rngs=rngs), nnx.relu,
            nnx.Conv(128, 128,  kernel_size=(4, 4), strides=(2,2), rngs=rngs), nnx.relu,
            nnx.Conv(128, 256,  kernel_size=(4, 4), strides=(2,2), rngs=rngs), nnx.relu,
            nnx.Conv(256, 256,  kernel_size=(4, 4), strides=(2,2), rngs=rngs), nnx.relu,
            nnx.Conv(256, 256,  kernel_size=(4, 4), strides=(1,1), rngs=rngs), nnx.relu,
        )
        
        self.enc = nnx.Linear(4*4*1024, 256, rngs=rngs)
        self.enc_mu = nnx.Linear(256, z_dim, rngs=rngs)
        self.enc_logvar = nnx.Linear(256, z_dim, rngs=rngs)
        
        rngss = nnx.vmap(lambda s: nnx.Rngs(s), in_axes=0)(split(rngs(), self.opts.num_decoders))
        self.decoder = nnx.vmap(self.Decoder, in_axes=(None, 0))(self.opts, rngss)

    def reparametrize(self, mu, logvar):
        std = jax.random.normal(self.rngs.reparam(), (mu.shape[0], mu.shape[1]))
        return mu + exp(0.5 * logvar) * std

    def __call__(self, x, reparam=True):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.enc(x)
        x = nnx.relu(x)
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


# if __name__ == "__main__":
#     if "train" not in locals():
#         train_fname = "/data/celeba/celeba_train_images.npy"
#         test_fname = "/data/celeba/celeba_test_images.npy"
#         train = np.load(train_fname)# , mmap_mode='r')

#         # train = (train - 0.5) * 2.0
#         # train = train[:2500]
#         train = jax.device_put(train)
#         # test  = np.load(test_fname, mmap_mode='r')
#         # test  = jax.device_put(test)


#     model = VAE(rngs=nnx.Rngs(jax.random.PRNGKey(0)))
#     params = nnx.state(model, nnx.Param, ...)[0]
#     param_count = builtins.sum(x.size for x in jax.tree_util.tree_leaves(params))
#     print(f"JAXVAE #of parameters: {param_count // 1e6}M")

#     # LPIPS INIT    
#     example = train[0]#.transpose(0,2,1)
#     lpips_obj = LPIPSFIX()
#     lpips_params = lpips_obj.init(model.rngs.dinfar(), example, example)

#     n_param = builtins.sum(x.size for x in jax.tree.leaves(lpips_params))
#     print(f"PERCEPT #of parameters: {n_param // 1e6}M")

#     # SMOKETEST
#     loss_fn = partial(loss_fn, lpips_obj=lpips_obj, lpips_params=lpips_params)
#     _loss, (_artfcs, _stats) = loss_fn(model, train[:8])
#     print(f"Loss (smoketest): {_loss}")

#     # lr_schedule = optax.warmup_exponential_decay_schedule(0.0, 1e-4, 1000, 100_000, 0.5)
#     # tx = optax.inject_hyperparams(getattr(optax, model.opts.opt))(lr_schedule)
#     # lr_schedule = optax.schedules.warmup_exponential_decay_schedule(1e-6, 1e-4, 5000, 100000, 0.30)
#     # tx = optax.inject_hyperparams(getattr(optax, model.opts.opt))(lr_schedule)
#     tx = getattr(optax, model.opts.opt)(model.opts.lr)
#     optimizer = nnx.Optimizer(model, tx)

#     for epoch_idx in range(model.opts.epochs):
#         with model.stats.time({"time": {"forward_train_epoch"}}, print=0) as block:
#             model.train()
#             stats = train_epoch(model, optimizer, train)
#             # print(jax.tree.map(lambda x: x.item(), optimizer.opt_state.hyperparams))
#             # breakpoint()
#             model.stats({"train": jax.tree.map(lambda x: x.item(), stats)})
#             # model.stats({"opt":  jax.tree.map(lambda x: x.item(), optimizer.opt_state.hyperparams)})

#         print(
#             *model.stats.latest(
#                 *[
#                     f"VAE {epoch_idx:03d} {model.stats['time']['forward_train_epoch'][-1]:.3f}s",
#                     {"train": "*"}#, {"opt": "*"}, 
#                 ]
#             )
#         )

#         if epoch_idx % 8 == 0:
#             print("Dumping model...")
#             model.dump(f"eugene/latest.pickle")

#     print("Training done. Dumping model...")
#     model.dump(f"eugene/latest.pickle")