import jax, optax, sys, time, pickle, builtins, os
import pandas as pd
import numpy as np
from jax.numpy import *
from jax.random import permutation, split
from flax import nnx
from dataclasses import dataclass, fields
from eugene.stats import Stats
from functools import partial

###
from lpips_j.lpips import VGGExtractor
from lpips_j.lpips import NetLinLayer
import h5py
import flax.linen as nn
from huggingface_hub import hf_hub_download
##

class LPIPSFIX(nn.Module):    
    def setup(self):
        self.vgg = VGGExtractor()
        
    def __call__(self, x, t, breakp=False):
        x = self.vgg(x)
        t = self.vgg(t)
        
        conv_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 
                      'conv3_2', 'conv3_3', 'conv3_3', 'conv4_1', 'conv4_2', 
                      'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
        diffs = []
        for f in conv_names:
            diff = (x[f] - t[f]) ** 2
            diff = 0.5 * diff.mean([1, 2, 3])
            diffs.append(diff)

        return stack(diffs, axis=1).sum(axis=1)
        
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

if "train" not in locals():
    train_fname = "/data/celeba/celeba_train_images.npy"
    test_fname = "/data/celeba/celeba_test_images.npy"
    train = np.load(train_fname)# , mmap_mode='r')

    # train = (train - 0.5) * 2.0
    # train = train[:2500]
    # train = jax.device_put(train)
    # test  = np.load(test_fname, mmap_mode='r')
    # test  = jax.device_put(test)


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
        self.conv = nnx.Conv(self.in_channels, self.filters, self.kernel_size, self.padding, rngs=rngs)

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
        epochs: int = 256       # Number of epochs to train for
        bs: int = 128           # batch size
        lr: float = 1e-5        # learnig rate
        dz: int = 64            # latent dimensionality
        opt: str = 'adam'       # 'adam'
        beta: int = 1.0         # \beta-VAE thing
        nD: int = 8             # number of Decoders
    
    class Decoder(nnx.Module):
        def __init__(self, opts, rngs):
            self.opts = opts
            self.fc_dec = nnx.Sequential(
                nnx.Linear(opts.dz, 2*2*512, rngs=rngs), nnx.elu
                # nnx.Linear(opts.dz, 2*2*256, rngs=rngs), nnx.elu, 
                # nnx.Linear(2*2*256, 2*2*512, rngs=rngs), nnx.elu
            )

            self.convs = nnx.Sequential(
                ResizeAndConv(512, 256, kernel_size=(3, 3), strides=(1, 1), padding=1, rngs=rngs), nnx.elu,
                ResizeAndConv(256, 128, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs), nnx.elu,
                ResizeAndConv(128, 64,  kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs), nnx.elu,
                ResizeAndConv(64,  32,  kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs), nnx.elu,
                ResizeAndConv(32,  16,  kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs), nnx.elu,
                ResizeAndConv(16,  8,   kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs), nnx.elu,
                ResizeAndConv(8,   3,   kernel_size=(3, 3), strides=(1, 1), padding=1, rngs=rngs),
            )

        def __call__(self, z):
            x_dec = self.fc_dec(z)
            x_dec = x_dec.reshape(x_dec.shape[0], 2, 2, 512)
            x_dec = self.convs(x_dec)
            x_dec = nnx.sigmoid(x_dec)
            return x_dec

    def __init__(self, opts={}, *, rngs: nnx.Rngs):
        self.stats = Stats()
        self.opts = self.DefaultOpts(**opts)
        z_dim = self.opts.dz

        self.rngs = rngs
        self.encoder = nnx.Sequential(
            nnx.Conv(3, 32,    kernel_size=(3, 3), strides=2, padding=1, rngs=rngs), nnx.elu,
            nnx.Conv(32, 64,   kernel_size=(3, 3), strides=2, padding=1, rngs=rngs), nnx.elu,
            nnx.Conv(64, 128,  kernel_size=(3, 3), strides=2, padding=1, rngs=rngs), nnx.elu,
            nnx.Conv(128, 256, kernel_size=(3, 3), strides=2, padding=1, rngs=rngs), nnx.elu,
            nnx.Conv(256, 512, kernel_size=(3, 3), strides=1, padding=1, rngs=rngs), nnx.elu,
        )
        
        self.enc_mu = nnx.Linear(4*4*512, z_dim, rngs=rngs)
        self.enc_logvar = nnx.Linear(4*4*512, z_dim, rngs=rngs)
        
        rngss = nnx.vmap(lambda s: nnx.Rngs(s), in_axes=0)(split(rngs(), self.opts.nD))
        self.decoder = nnx.vmap(self.Decoder, in_axes=(None, 0))(self.opts, rngss)

        # self.lpips_obj = None
        # self.lpips_params = None

    def reparametrize(self, mu, logvar):
        # if self.opts.beta == 0.0:
        #     return mu

        std = jax.random.normal(self.rngs.reparam(), (mu.shape[0], mu.shape[1]))
        return mu + exp(0.5 * logvar) * std

    def __call__(self, x, reparam=True):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        z_mu = self.enc_mu(x)
        z_logvar = self.enc_logvar(x)

        z = self.reparametrize(z_mu, z_logvar) if reparam else z_mu
        x_dec = self.decode(z)

        return x_dec, z_mu, z_logvar

    def decode(self, z):
        # split z into nD parts and decode each part
        z = z.reshape(self.opts.nD, z.shape[0] // self.opts.nD, z.shape[1])
        decoded = nnx.vmap(lambda z, d: d(z), in_axes=(0, 0))(z, self.decoder)
        decoded = decoded.reshape(-1, *decoded.shape[2:])
        return decoded

    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump(
                {"opts": self.opts, "stats": self.stats, "state": nnx.state(self)}, file
            )

def loss_fn(model, batch, lpips_obj = None, lpips_params = None):
    x_hat, z_mu, z_logvar = model(batch)
    
    kl_loss = -0.5 * sum(1.0 + z_logvar - z_mu**2 - exp(z_logvar), axis=-1)

    if False:
        rec_loss = optax.l2_loss(x_hat, batch).sum([-1, -2, -3])
        prc_loss = array(0.0)
    else:
        rec_loss = array(0.0) # optax.l2_loss(x_hat, batch).sum([-1, -2, -3]) #array(0.0)
        # batch = ((batch - 0.5) * 2.0).transpose(0,2,1,3)
        # x_hat = ((x_hat - 0.5) * 2.0).transpose(0,2,1,3)
        batch = batch.transpose(0,2,1,3)
        x_hat = x_hat.transpose(0,2,1,3)
        prc_loss = lpips_obj.apply(lpips_params, batch, x_hat, breakp=True)
        # prc_loss = prc_loss.sum([1,2,3])
        
    
    # breakpoint()
    loss = rec_loss + prc_loss + model.opts.beta * kl_loss

    stats = {
        "elbo": -loss.mean(),
        "kl_loss": kl_loss.mean(),
        "rec_loss": rec_loss.mean(),
        "prc_loss": prc_loss.mean(),
    }

    return loss.mean(), ((x_hat, z_mu, z_logvar), stats)


@nnx.jit
def train_epoch(model, optimizer, train):
    # t0 = time.time()
    n_full = train.shape[0] // model.opts.bs
    permut = permutation(model.rngs.permut(), n_full * model.opts.bs)
    batches = train[permut].reshape(n_full, model.opts.bs, *train.shape[1:])
    # print(f"train_epoch permut: {time.time() - t0:.3f}s")
    return train_epoch_inner(model, optimizer, batches)


@nnx.jit
def train_epoch_inner(model, optimizer, batches):
    grad_loss_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    def train_step(model_opt, batch):
        model, optimizer = model_opt
        (loss_, (artfcs_, stats)), grads = grad_loss_fn(model, batch)
        optimizer.update(grads)
        return (model, optimizer), stats

    in_axes = (nnx.Carry, 0)
    train_step_scan_fn = nnx.scan(train_step, in_axes=in_axes)
    model_opt = (model, optimizer)
    _, stats_stack = nnx.jit(train_step_scan_fn)(model_opt, batches)
    return jax.tree.map(lambda x: x.mean(), stats_stack)


if __name__ == "__main__":
    model = VAE(rngs=nnx.Rngs(jax.random.PRNGKey(0)))
    params = nnx.state(model, nnx.Param, ...)[0]
    param_count = builtins.sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"JAXVAE #of parameters: {param_count // 1e6}M")

    # LPIPS INIT    
    example = train[0]#.transpose(0,2,1)
    lpips_obj = LPIPSFIX()
    lpips_params = lpips_obj.init(model.rngs.dinfar(), example, example)

    n_param = builtins.sum(x.size for x in jax.tree.leaves(lpips_params))
    print(f"PERCEPT #of parameters: {n_param // 1e6}M")

    # SMOKETEST
    loss_fn = partial(loss_fn, lpips_obj=lpips_obj, lpips_params=lpips_params)
    _loss, (_artfcs, _stats) = loss_fn(model, train[:8])
    print(f"Loss (smoketest): {_loss}")

    # lr_schedule = optax.warmup_exponential_decay_schedule(0.0, 1e-4, 1000, 100_000, 0.5)
    # tx = optax.inject_hyperparams(getattr(optax, model.opts.opt))(lr_schedule)
    # lr_schedule = optax.schedules.warmup_exponential_decay_schedule(1e-6, 1e-4, 5000, 100000, 0.30)
    # tx = optax.inject_hyperparams(getattr(optax, model.opts.opt))(lr_schedule)
    tx = getattr(optax, model.opts.opt)(model.opts.lr)
    optimizer = nnx.Optimizer(model, tx)

    for epoch_idx in range(model.opts.epochs):
        with model.stats.time({"time": {"forward_train_epoch"}}, print=0) as block:
            model.train()
            stats = train_epoch(model, optimizer, train)
            # print(jax.tree.map(lambda x: x.item(), optimizer.opt_state.hyperparams))
            # breakpoint()
            model.stats({"train": jax.tree.map(lambda x: x.item(), stats)})
            # model.stats({"opt":  jax.tree.map(lambda x: x.item(), optimizer.opt_state.hyperparams)})

        print(
            *model.stats.latest(
                *[
                    f"VAE {epoch_idx:03d} {model.stats['time']['forward_train_epoch'][-1]:.3f}s",
                    {"train": "*"}#, {"opt": "*"}, 
                ]
            )
        )

        if epoch_idx % 8 == 0:
            print("Dumping model...")
            model.dump(f"eugene/latest.pickle")

    print("Training done. Dumping model...")
    model.dump(f"eugene/latest.pickle")