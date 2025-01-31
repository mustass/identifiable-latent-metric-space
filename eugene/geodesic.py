import os, pickle, jax
os.chdir('..' if os.getcwd().endswith("/eugene") else '.')
from ilms.data import get_celeba_arrays

# jax.config.update('jax_platform_name', 'cpu')
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pandas as pd
# from eugene.vae import VAE

from ilms.models.celeba_vae_prc import VAE

from flax import nnx
import matplotlib.pyplot as plt
from jax.random import PRNGKey
import jax.numpy as jnp

from ilms.training import GeodesicsEval
from omegaconf import DictConfig, OmegaConf, open_dict

from ilms.utils import (
    set_seed,
    load_obj,
    save_useful_info,
    chunks,
    pick_pairs,
)

# import numpy as np
# test_fname = "/data/celeba/celeba_test_images.npy"
# test = np.load(test_fname)# , mmap_mode='r')
# test = test[:32]

with open('eugene/best_model_69.pickle', 'rb') as file:
    model_dict = pickle.load(file)

model = VAE(rngs=nnx.Rngs(0))

model.opts = model_dict['opts']
model.stats = model_dict['stats']
nnx.update(model, model_dict['state'])

# class GeodesicsEval_Extras(GeodesicsEval):
#     def __init__(self, model: nnx.Module, config: dict, wandb_logger):
#         super().__init__(model, config, wandb_logger)

opts = {
    'model': {
        'class_name': 'ilms.models.celeba_vae_prc.VAE',
        'params': {
            'z_dim': 64,
            'num_decoders': 8
        }
    },
    'seed': 0,
    'optimizer': {'class_name': 'adam', 'params': {}},
    'loss': {
        'loss_type': 'prc',
        'class_name': 'ilms.losses.NelboLoss',
        'params': {}
    },
    'datamodule': {
        'batch_size': 32,
    },
    'training': {
        'checkpoint': 'eugene/geo_checkpoints',
        'seed': 0,
        'early_stopping_patience': 10,
        'early_stopping_grace': 0.1,
        'eval_every': 1,
    },
    'inference': {
        'geodesics_params': {
            'n_steps': 1024,
            'n_poly': 16,
            'n_t': 256,
            'n_t_lengths': 256,
            'batch_size': 32,
            'optimizer': {
                'class_name': 'optax.adam', 
                'params': {
                    'b1': 0.9,
                    'b2': 0.999,
                    'eps': 1e-08,
                    'eps_root': 0.0
                }
            },
            'lr': 0.01,
            'mode': 'prc',
            'method': 'rk4',
            'init_mode': 'random',
            'init_scale': 0.1,
            'warmup_steps': 100,
            'early_stopping_n': 10,
            'early_stopping_delta': 0.01
        }
    }
}
opts = OmegaConf.create(opts)

trainer = GeodesicsEval(model, opts, None)
seed = 5

if 'test_images' not in locals():
    _, _, _, _, test_images, test_labels = get_celeba_arrays("/data/celeba")

    point_pairs = pick_pairs(
        test_images,
        test_labels,
        1,
        seed,
    )

key = PRNGKey(0)
input = []
labels = []
norms_ambient = []

for i, pair in enumerate(point_pairs):
    random_key, key = jax.random.split(key, 2)

    input.append(
        jnp.array([test_images[pair[0]], test_images[pair[1]]])
    )
    norms_ambient.append(
        jnp.linalg.norm(
            jnp.ravel(jnp.array(test_images[pair[0]]))
            - jnp.ravel(jnp.array(test_images[pair[1]])),
            ord=2,
        ).item()
    )

    (
        best_sum_of_energies,
        lengths,
        best_single_energies,
        history,
        eucleadian_dists,
        eucleadian_reconstructed_ambient,
        geodesic
    ) = trainer.compute_geodesic(jnp.array(input), random_key, return_geo=True)


t = jnp.linspace(0, 1, opts.inference.geodesics_params.n_t)

geo_evals = geodesic.eval(t)
geo_eval_line = jax.vmap(geodesic._eval_line, (None, 0))(t, geodesic.point_pairs)

diff_sq = ((geo_evals - geo_eval_line)**2).sum()
print("Diff sq:", diff_sq)

pickle.dump({
    'seed': 0,
    'lengths': lengths,
    'eucleadian_dists': eucleadian_dists,
    'geo_evals': geo_evals,
    'geo_eval_line': geo_eval_line,
    'point_pairs': point_pairs,
}, open(f"eugene/geo_evals_seed{seed}.pkl", 'wb'))