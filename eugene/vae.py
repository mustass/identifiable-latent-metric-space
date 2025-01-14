import jax, optax, sys, time, pickle, builtins, os
import pandas as pd
import numpy as np
from jax.numpy import *
from jax.random import permutation
from flax import nnx
from dataclasses import dataclass, fields
from eugene.stats import Stats

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

if 'train' not in locals():
    train_fname = "/data/celeba/celeba_train_images.npy"
    test_fname = "/data/celeba/celeba_test_images.npy"
    train = np.load(train_fname)#, mmap_mode='r')
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
    @dataclass
    class DefaultOpts:
        epochs: int = 256       # Number of epochs to train for
        bs: int = 64            # batch size
        lr: float = 1e-5        # learnig rate
        dz: int = 96            # latent dimensionality
        opt: str = 'adam'       # 'adam'
        beta: int = 1.0         # \beta-VAE thing
        
    def __init__(self, opts = {}, *, rngs: nnx.Rngs):
        self.stats = Stats()
        self.opts = self.DefaultOpts(**opts)
        z_dim = self.opts.dz

        self.rngs = rngs
        self.encoder = nnx.Sequential(
            nnx.Conv(3, 128, kernel_size=(4, 4), strides=(1, 1), rngs=rngs),
            nnx.elu,
            nnx.Conv(128, 128, kernel_size=(4, 4), strides=(2, 2), rngs=rngs),
            nnx.elu,
        )
        
        self.enc_mu = nnx.Linear(32*32*128, z_dim, rngs=rngs)
        self.enc_logvar = nnx.Linear(32*32*128, z_dim, rngs=rngs)
        
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

    def reparametrize(self, mu, logvar):
        # if self.opts.beta == 0.0:
        #     return mu
        
        std = jax.random.normal(self.rngs.reparam(), (mu.shape[0], mu.shape[1]))
        return mu + exp(0.5 * logvar)* std

    def __call__(self, x, reparam = True):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        z_mu = self.enc_mu(x)
        z_logvar = self.enc_logvar(x)

        z = self.reparametrize(z_mu, z_logvar) if reparam else z_mu
        x_dec = self.decode(z)
        
        return x_dec, z_mu, z_logvar
    
    def decode(self, z):
        x_dec = self.fc_dec(z)
        x_dec = x_dec.reshape(x_dec.shape[0], 8, 8, 128)
        x_dec = self.decoder(x_dec)

        return x_dec
        
    
    def dump(self, path):
        with open(path, 'wb') as file:
            pickle.dump({
                'opts':   self.opts, 
                'stats':  self.stats,
                'state':  nnx.state(self)
            }, file)

def loss_fn(model, batch):
    x_hat, z_mu, z_logvar = model(batch)
    
    rec_loss = optax.l2_loss(x_hat, batch).sum([-1,-2,-3])
    kl_loss = -0.5 * sum(1.0 + z_logvar - z_mu**2 - exp(z_logvar), axis=-1)
    
    loss = rec_loss + model.opts.beta * kl_loss
    
    stats = {
        'elbo':    -loss.mean(),
        'kl_loss':  kl_loss.mean(),
        'rec_loss': rec_loss.mean(),
    }
    
    return loss.mean(), ((x_hat, z_mu, z_logvar), stats)

@nnx.jit
def train_epoch(model, optimizer, train):
    # t0 = time.time()
    n_full  = train.shape[0] // model.opts.bs
    permut  = permutation(model.rngs.permut(), n_full * model.opts.bs)
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
    print(f"JAXVAE Number of parameters: {param_count // 1e6}M")

    _loss, (_artfcs, _stats) = loss_fn(model, train[:32])
    print(f"Loss (smoketest): {_loss}")

    tx = getattr(optax, model.opts.opt)(model.opts.lr)
    optimizer = nnx.Optimizer(model, tx)

    for epoch_idx in range(model.opts.epochs):
        with model.stats.time({'time': {'forward_train_epoch'}}, print=0) as block:
            model.train()
            stats = train_epoch(model, optimizer, train)
            # model.eval()
            # stats['oracle'] = oracle_test(model, data)
            model.stats({'train': jax.tree.map(lambda x: x.item(), stats)})
        
        print(*model.stats.latest(*[
            f"VAE {epoch_idx:03d} {model.stats['time']['forward_train_epoch'][-1]:.3f}s",
            {'train': '*'}, 
        ]))

    model.dump(f"eugene/latest.pickle")