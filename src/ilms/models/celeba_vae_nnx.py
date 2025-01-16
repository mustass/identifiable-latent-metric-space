
from flax import nnx
import jax
from dataclasses import dataclass
import pickle
from jax.random import permutation, split
import optax
from jax.numpy import exp

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
        dz: int = 128           # latent dimensionality
        beta: int = 1.0         # \beta-VAE thing
        num_decoders: int = 8             # number of Decoders
    
    class Decoder(nnx.Module):
        def __init__(self, opts, rngs):
            self.opts = opts
            self.fc_dec = nnx.Sequential(
                nnx.Linear(opts.dz, 8*8*32, rngs=rngs), # nnx.Linear(opts.dz, 4*4*512, rngs=rngs),
                nnx.elu
            )

            self.convs = nnx.Sequential(
                ResizeAndConv(32, 128, (4, 4), (1, 1), rngs=rngs), # ResizeAndConv(128, 128, (4, 4), (1, 1), rngs=rngs),
                nnx.elu,
                ResizeAndConv(128, 128, (4, 4), (2, 2), rngs=rngs),
                nnx.elu,
                ResizeAndConv(128, 128, (4, 4), (2, 2), rngs=rngs),
                nnx.elu,
                ResizeAndConv(128, 128, (4, 4), (2, 2), rngs=rngs),
                nnx.elu,
                ResizeAndConv(128, 3,   (4, 4), (1, 1), rngs=rngs),
            )
        
        def __call__(self, z):
            x_dec = self.fc_dec(z)
            x_dec = x_dec.reshape(x_dec.shape[0], 8, 8, 32) #.reshape(x_dec.shape[0], 8, 8, 128)
            x_dec = self.convs(x_dec)
            return x_dec 
        
    def __init__(self, opts = {}, *, rngs: nnx.Rngs):
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
        
        rngss = nnx.vmap(lambda s: nnx.Rngs(s), in_axes=0)(split(rngs(), self.opts.num_decoders))
        self.decoder = nnx.vmap(self.Decoder, in_axes=(None, 0))(self.opts, rngss)
    

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
        # split z into num_decoders parts and decode each part
        z = z.reshape(self.opts.num_decoders, z.shape[0] // self.opts.num_decoders, z.shape[1])
        decoded = nnx.vmap(lambda z, d: d(z), in_axes=(0, 0))(z, self.decoder)
        decoded = decoded.reshape(-1, *decoded.shape[2:])    
        return decoded
    
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
