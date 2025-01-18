import logging
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
from ilms.utils.utils import load_obj
import numpy as np
import jax.tree_util as jtu
import equinox as eqx
from ilms.geometry import Geodesics
from jax import Array
from jax.random import split
from .trainers import TrainerModule
from flax import nnx

class GeodesicsEval(TrainerModule):
    def __init__(self, model: nnx.Module, config: DictConfig, wandb_logger):
        super().__init__(model, config, wandb_logger)

        self.n_steps = config.inference.geodesics_params.n_steps
        self.n_poly = config.inference.geodesics_params.n_poly
        self.n_t = config.inference.geodesics_params.n_t
        self.n_t_lengths = config.inference.geodesics_params.n_t_lengths
        self.latents = None
        self.labels = None
        self.meshgrid = None
        self.determinant = None

        self.n_ensemble = config.model.get("num_decoders", None)
        self.metric_mode = (
            "ensemble"
            if (self.n_ensemble is not None) and (self.n_ensemble > 1)
            else "single"
        )

        self.geodesics_optimizer = (
            config.inference.geodesics_params.optimizer.class_name
        )
        self.geodesics_optimizer_params = (
            config.inference.geodesics_params.optimizer.params
        )

        self.geodesics_lr = config.inference.geodesics_params.lr
        self.geodesics_mode = config.inference.geodesics_params.mode
        self.geodesics_method = config.inference.geodesics_params.method
        self.geodesics_bs = config.inference.geodesics_params.batch_size
        self.geodesics_init_mode = config.inference.geodesics_params.init_mode
        self.geodesics_init_scale = config.inference.geodesics_params.init_scale
        self.warmup_steps = config.inference.geodesics_params.warmup_steps
        self.early_stopping_n = config.inference.geodesics_params.early_stopping_n
        self.early_stopping_delta = (
            config.inference.geodesics_params.early_stopping_delta
        )

    def create_functions(self):
        def calculate_energy(key, t, diff_model, static_model):
            geodesics = eqx.combine(diff_model, static_model)
            energies = geodesics.calculate_energy(
                t,
                key,
                self.geodesics_mode,
                derivative="delta",
                metric_mode=self.metric_mode,
                n_ensemble=self.n_ensemble,
            )

            return jnp.sum(energies, axis=None), energies

        @eqx.filter_jit
        def geodesic_optim_step(g, t, opt_state, key, filter_spec):
            diff_model, static_model = eqx.partition(g, filter_spec)

            loss = lambda d, s: calculate_energy(key, t, d, s)

            (energy_value, energies), grads = jax.value_and_grad(loss, has_aux=True)(
                diff_model, static_model
            )
            updates, opt_state = self.optimizer.update(grads, opt_state)
            g = eqx.apply_updates(g, updates)

            return g, energy_value, opt_state, energies

        self.geodesic_step = geodesic_optim_step

    def compute_geodesic(self, batch, key):
        key_init, key_encode, key = random.split(key, 3)

        @eqx.filter_jit
        def _encode(batch, key):
            keys = split(key, 2)
            return jax.vmap(self.model.encode)(batch, keys)[0]

        @eqx.filter_jit
        def _decode(batch):
            return jax.vmap(self.model.decode)(batch)

        encode_keys = split(key_encode, batch.shape[0])

        point_pairs = jax.vmap(_encode)(
            batch,
            encode_keys,
        )  ## The double vmap will have result (pair,from/to,dim), so 3 pairs, 4 dims will be (3,2,4)
        # point_pairs_decoded = jax.vmap(_decode)(point_pairs)
        euclidean_distances = jax.vmap(
            lambda x: jnp.linalg.norm(x[0, :] - x[1, :], ord=2)
        )(point_pairs)
        euclidean_in_ambient = jnp.array(
            [0.0]
        )  # jax.vmap(lambda x: jnp.linalg.norm(x[0, :] - x[1, :], ord=2))(point_pairs_decoded)

        geodesic = Geodesics(
            self.model,
            self.n_poly,
            point_pairs,
            key_init,
            self.geodesics_init_mode,
            self.geodesics_init_scale,
        )
        self.optimizer, opt_state = self.init_optimizer(
            eqx.filter(geodesic, eqx.is_array)
        )

        filter_spec = jtu.tree_map(lambda _: False, geodesic)

        filter_spec = eqx.tree_at(
            lambda tree: tree.params,
            filter_spec,
            replace=True,
        )
        t = jnp.linspace(0, 1, self.n_t)
        best_energy = jnp.inf
        best_energy_step = 0

        energy_history = []
        for i in (
            pbar := tqdm(range(self.n_steps), desc="Training geodesic", leave=False)
        ):
            key_geodesic_step, key = random.split(key, 2)
            geodesic, energy, opt_state, energies = self.geodesic_step(
                geodesic, t, opt_state, key_geodesic_step, filter_spec
            )

            energy_history.append(energy.item())

            if best_energy - energy > self.early_stopping_delta:
                best_energy = energy
                best_energies = np.array(energies)
                best_params = geodesic.params
                best_energy_step = i

            pbar.set_postfix(current_energy=energy.item())

            if (i - best_energy_step) > self.early_stopping_n:
                break

        geodesic = eqx.tree_at(lambda g: g.params, geodesic, best_params)

        length_key, key = random.split(key, 2)
        lengths = geodesic.calculate_length(
            jnp.linspace(0, 1, self.n_t_lengths),
            length_key,
            derivative="delta",
            metric_mode=self.metric_mode,
            n_ensemble=self.n_ensemble,
        )

        return (
            best_energy.item(),
            lengths,
            best_energies,
            energy_history,
            euclidean_distances,
            euclidean_in_ambient,
        )

    def latents_data(self, data_set, key: Array):
        logging.info("👉 Computing latents for the dataset...")

        @eqx.filter_jit
        def _encode(batch, key):
            return self.model.encode(batch, key)[0]

        latents = []
        labels = []
        for i, batch in enumerate(data_set):
            _k, key = split(key, 2)
            latents.append(_encode(batch[0], _k))
            labels.append(np.argmax(batch[1]))

        self.latents = np.array(latents)
        self.labels = np.array(labels)

        logging.info("👍 Latents computed successfully")

    def init_optimizer(self, params):
        # grad_transformations = [optax.clip_by_global_norm(1.0)]

        # lr_schedule = optax.warmup_cosine_decay_schedule(
        #    init_value=0.0,
        #    peak_value=self.geodesics_lr,
        #    warmup_steps=self.warmup_steps,
        #    decay_steps=self.n_steps,
        #    end_value=self.geodesics_lr*0.1
        # )

        # grad_transformations.append(
        #    load_obj(self.geodesics_optimizer)(self.geodesics_lr, **self.geodesics_optimizer_params)
        # )

        # self.optimizer = optax.chain(*grad_transformations)
        optimizer = load_obj(self.geodesics_optimizer)(
            self.geodesics_lr, **self.geodesics_optimizer_params
        )

        opt_state = optimizer.init(params)
        return optimizer, opt_state
