import os, logging, jax, optax, sys, time, pickle, builtins, wandb, yaml
from omegaconf import DictConfig, OmegaConf
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
from ilms.utils.utils import compute_num_params, load_obj
import matplotlib.pyplot as plt
from flax import nnx
from jax import jit
from flax.training import train_state
from tqdm import tqdm
from jax.random import permutation, split

class TrainerModule:
    def __init__(self, model: nnx.Module, config: DictConfig, wandb_logger):

        super().__init__()
        self.model = model
        self.optim_config = config["optimizer"]
        self.train_config = config["training"]
        self.loss_config = config["loss"]
        self.batch_size = config["datamodule"]["batch_size"]
        self.grad_clipping_config = config.get("grad_clipping", None)
        self.scheduler_config = config.get("scheduler", None)
        self.config = config


        self.seed = self.train_config["seed"]
        self.early_stopping_patience = self.train_config["early_stopping_patience"]
        self.early_stopping_grace = self.train_config["early_stopping_grace"]
        self.eval_every = self.train_config["eval_every"]

        self.logger = wandb_logger

        self.model_checkpoint_path = self.train_config["checkpoint"]
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model

        self.init_model(model)

        self.loss_func = load_obj(self.loss_config["class_name"])(
            **self.loss_config["params"]
        )

        with open(
                os.path.join(self.model_checkpoint_path, "config.yml"), "w"
            ) as f:
                OmegaConf.save(self.config, f)



    def init_model(self, model):
        params = nnx.state(model, nnx.Param, ...)[0]
        param_count = builtins.sum(x.size for x in jax.tree_util.tree_leaves(params))
        logging.info(f"JAXVAE Number of parameters: {param_count // 1e6}M")

    def init_optimizer(self, num_steps):

        grad_transformations = []

        if self.grad_clipping_config is not None:
            grad_transformations.append(
                load_obj(self.grad_clipping_config["class_name"])(
                    **self.grad_clipping_config["params"]
                )
            )

        if self.scheduler_config["class_name"] == "optax.warmup_cosine_decay_schedule":
            self.scheduler_config["params"]["decay_steps"] = num_steps
            lr_schedule = load_obj(self.scheduler_config["class_name"])(
                **self.scheduler_config["params"]
            )

        elif (
            self.scheduler_config["class_name"]
            == "optax.warmup_exponential_decay_schedule"
        ):
            lr_schedule = load_obj(self.scheduler_config["class_name"])(
                **self.scheduler_config["params"]
            )

        elif self.scheduler_config["class_name"] == "optax.piecewise_constant_schedule":
            assert len(self.scheduler_config["params"]["boundaries"]) == len(
                self.scheduler_config["params"]["scales"]
            ), "LR scheduler must have same number of boundaries and scales"
            boundaries_and_scales = dict(
                [
                    (num_steps * key, value)
                    for key, value in zip(
                        self.scheduler_config["params"]["boundaries"],
                        self.scheduler_config["params"]["scales"],
                    )
                ]
            )
            lr_schedule = load_obj(self.scheduler_config["class_name"])(
                init_value=self.scheduler_config["params"]["init_value"],
                boundaries_and_scales=boundaries_and_scales,
            )
        elif self.scheduler_config["class_name"] == "optax.constant_schedule":
            lr_schedule = load_obj(self.scheduler_config["class_name"])(
                **self.scheduler_config["params"]
            )
        elif self.scheduler_config["class_name"] == "optax.sgdr_schedule":
            n_iterations = (num_steps) // self.scheduler_config["params"][
                "decay_steps"
            ] + 1
            params = [
                dict(self.scheduler_config["params"]) for i in range(n_iterations)
            ]
            lr_schedule = load_obj(self.scheduler_config["class_name"])(params)
        else:
            raise NotImplementedError

        grad_transformations.append(
            optax.inject_hyperparams(load_obj(self.optim_config["class_name"]))(
                lr_schedule, **self.optim_config["params"]
            )
        )

        self.optimizer = nnx.Optimizer(self.model, optax.chain(*grad_transformations))

    def train_model(self, train_array, val_array, num_epochs, logger=None):
        max_steps = (
            self.train_config["num_epochs"] * train_array.shape[0] // self.batch_size
        )
        
        self.init_optimizer(max_steps)

        best_val_elbo = -jnp.inf
        best_val_elbo_epoch=0
        best_checkpoint_name=""
        
        for epoch_idx in range(num_epochs):
            with self.model.stats.time(
                {"time": {"forward_train_epoch"}}, print=0
            ) as block:
                self.model.train()
                stats = self.train_epoch(self.model, self.optimizer, train_array)
                self.model.stats({"train": jax.tree.map(lambda x: x.item(), stats)})
                
                stats["lr"] = (
                    self.optimizer.opt_state[1].hyperparams["learning_rate"]
                    if self.grad_clipping_config is not None
                    else self.optimizer.opt_state[0].hyperparams["learning_rate"]
                )

            print(
                *self.model.stats.latest(
                    *[
                        f"VAE {epoch_idx:03d} {self.model.stats['time']['forward_train_epoch'][-1]:.3f}s",
                        {"train": "*"},
                    ]
                )
            )

            for dict_key, dict_val in stats.items():
                self.logger.log(
                    {"train_" + dict_key + "_epoch": dict_val.item()}, step=epoch_idx
                )

            if epoch_idx % self.eval_every == 0:
                val_elbo = self.eval_model(val_array, epoch_idx)["elbo"]
                if (val_elbo- best_val_elbo) > self.early_stopping_grace:
                    checkpoint_name = f"{self.model_checkpoint_path}/checkpoint_epoch_{epoch_idx}_val_elbo_{val_elbo:.2f}.pickle"
                    self.model.dump(checkpoint_name)
                    if len(best_checkpoint_name)>0:
                        if os.path.exists(best_checkpoint_name):
                            os.remove(best_checkpoint_name)
                    best_checkpoint_name = checkpoint_name
                    best_val_elbo = val_elbo
                    best_val_elbo_epoch = epoch_idx

                    logging.info(f"Saved checkpoint with val_elbo={val_elbo:.2f} at epoch {epoch_idx}.")
                elif (not (val_elbo- best_val_elbo) > self.early_stopping_grace) and (epoch_idx-best_val_elbo_epoch) > self.early_stopping_patience:
                    logging.info(f"Early stopping invoked after {epoch_idx-best_val_elbo_epoch} of no improvement above {self.early_stopping_grace}")
                    break
                else:
                    continue

        self.model.dump(f"{self.model_checkpoint_path}/last_checkpoint.pickle")
        logging.info(f"Model training completed \n Saved the model to {self.model_checkpoint_path}")
        if os.path.exists(best_checkpoint_name):
            os.rename(best_checkpoint_name,f"{self.model_checkpoint_path}/best_model.pickle")
        logging.info(f"Renamed the best model checkpoint to {self.model_checkpoint_path}/best_model.pickle")
        

    def eval_model(self, val_array, step, prefix="val"):
        with self.model.stats.time({"time": {"forward_eval_epoch"}}, print=0) as block:
            self.model.eval()
            stats = self.eval_epoch(self.model, val_array)
            self.model.stats({"eval": jax.tree.map(lambda x: x.item(), stats)})

        print(
            *self.model.stats.latest(
                *[
                    f"VAE {step:03d} {self.model.stats['time']['forward_eval_epoch'][-1]:.3f}s",
                    {"eval": "*"},
                ]
            )
        )
        for dict_key, dict_val in stats.items():
            self.logger.log({f"{prefix}_" + dict_key + "_epoch": dict_val.item()}, step=step)

        self.plot_posterior_samples(
            val_array,
            step,
            prefix,
        )

        self.plot_prior_samples(step)

        return stats 


    def plot_posterior_samples(
        self,
        val_array,
        step,
        prefix,
    ):
        # select random images from the validation set
        orig_images = random.choice(self.model.rngs.pilsner(), val_array, shape=(8,))
        orig_images = jnp.clip(orig_images, a_min=0.0, a_max=1.0)
        # reconstruct the images
        reconstructs, _, _ = self.model(orig_images)
        reconstructs = jnp.clip(reconstructs, a_min=0.0, a_max=1.0)
        fig, axes = plt.subplots(2, 8, figsize=(8, 4))
        for i, ax in enumerate(axes[0]):
            ax.imshow(orig_images[i])
            ax.axis("off")
        for i, ax in enumerate(axes[1]):
            ax.imshow(reconstructs[i])
            ax.axis("off")
        # add labels to rows
        axes[0, 0].set_title("Original")
        axes[1, 0].set_title("Reconstructed")
        plt.tight_layout(pad=-2.0)
        # log to wandb
        # add title to plot
        fig.suptitle(f"Posterior mean at: {step}")
        self.logger.log({f"posterior_samples_{prefix}": wandb.Image(fig)}, step=step)

    def plot_prior_samples(self, step):
        z = jax.random.normal(self.model.rngs.ipa(), (8, self.model.opts.z_dim))
        x = self.model.decode(z).transpose(0, 1, 2, 3)
        x = jnp.clip(x, a_min=0.0, a_max=1.0)
        fig, axes = plt.subplots(1, 8, figsize=(8, 4))
        for i, ax in enumerate(axes):
            ax.imshow(x[i])
            ax.axis("off")
        plt.tight_layout(pad=-2.0)
        fig.suptitle(f"Samples from prior at step: {step}")
        self.logger.log({"prior_samples": wandb.Image(fig)}, step=step)


class Trainer(TrainerModule):
    def create_functions(self):
        # Training function
        @nnx.jit
        def train_epoch(model, optimizer, train):
            n_full = train.shape[0] // self.batch_size
            permut = permutation(model.rngs.permut(), n_full * self.batch_size)
            batches = train[permut].reshape(n_full, self.batch_size, *train.shape[1:])
            return train_epoch_inner(model, optimizer, batches)

        @nnx.jit
        def train_epoch_inner(model, optimizer, batches):
            grad_loss_fn = nnx.value_and_grad(self.loss_func, has_aux=True)

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

        # Eval function
        @nnx.jit
        def eval_epoch(model, valid):
            n_full = valid.shape[0] // self.batch_size
            permut = permutation(model.rngs.permut(), n_full * self.batch_size)
            batches = valid[permut].reshape(n_full, self.batch_size, *valid.shape[1:])
            return eval_epoch_inner(model, batches)

        @nnx.jit
        def eval_epoch_inner(model, batches):

            def val_step(model, batch):
                loss_, (artfcs_, stats) = self.loss_func(model, batch)
                return model, stats

            in_axes = (nnx.Carry, 0)
            val_step_scan_fn = nnx.scan(val_step, in_axes=in_axes)
            _, stats_stack = nnx.jit(val_step_scan_fn)(model, batches)
            return jax.tree.map(lambda x: x.mean(), stats_stack)

        self.train_epoch = train_epoch
        self.eval_epoch = eval_epoch