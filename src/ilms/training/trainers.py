import os, logging, jax, optax, sys, time, pickle, builtins, wandb
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

# class TrainState(train_state.TrainState):
#   counts: nnx.State
#   graphdef: nnx.GraphDef

# class Count(nnx.Variable[nnx.A]):
#   pass


class TrainerModule:
    def __init__(self, model: nnx.Module, config: DictConfig, wandb_logger):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        """

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
        self.early_stopping = self.train_config["early_stopping_patience"]
        self.val_steps = self.train_config["val_steps"]
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

        self.config_saved = False

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
        max_steps = self.train_config["num_epochs"] * train_array.shape[0] // self.batch_size
        self.init_optimizer(max_steps)


        for epoch_idx in range(num_epochs):
            with self.model.stats.time(
                {"time": {"forward_train_epoch"}}, print=0
            ) as block:
                self.model.train()
                stats = self.train_epoch(self.model, self.optimizer, train_array)
                self.model.stats({"train": jax.tree.map(lambda x: x.item(), stats)})

            print(
                *self.model.stats.latest(
                    *[
                        f"VAE {epoch_idx:03d} {self.model.stats['time']['forward_train_epoch'][-1]:.3f}s",
                        {"train": "*"},
                    ]
                )
            )

            # for dict_key, dict_val in self.model.stats.latest.items():
            #     self.logger.log(
            #         {"train_" + dict_key + "_batch": dict_val}, step=epoch_idx
            #     )

        self.model.dump(f"{self.model_checkpoint_path}/dump.pickle")

    def eval_model(self, train_dataset, val_dataset, step, params, key):
        tkey, vkey, plot_key, generation_key = random.split(key, num=4)
        tavg_loss, tavg_rec, tavg_kl, tdec_mean, tdec_logstd, ttargets, _ = (
            self.eval_func(train_dataset, step, params, tkey)
        )
        (
            vavg_loss,
            vavg_rec,
            vavg_kl,
            vdec_mean,
            vdec_logstd,
            vtargets,
            val_metrics_dict,
        ) = self.eval_func(val_dataset, step, params, vkey)

        self.plot_posterior_samples(
            tdec_mean,
            tdec_logstd,
            ttargets,
            vdec_mean,
            vdec_logstd,
            vtargets,
            step,
            plot_key,
        )

        self.plot_prior_samples(params, generation_key, step)

        for dict_key, dict_val in val_metrics_dict.items():
            if not dict_key in ["dec_mean", "dec_logstd"]:
                self.logger.log(
                    {"val_" + dict_key + f"_{self.val_steps}_batches": dict_val},
                    step=step,
                )

        return tavg_loss, tavg_rec, tavg_kl, vavg_loss, vavg_rec, vavg_kl

    # def unnormalize(self, image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    #     image = image * jnp.array(std) + jnp.array(mean)
    #     return image

    def plot_posterior_samples(
        self,
        tdec_mean,
        tdec_logstd,
        ttargets,
        vdec_mean,
        vdec_logstd,
        vtargets,
        step,
        key,
    ):

        tdec_mean = self.unnormalize(tdec_mean)
        ttargets = self.unnormalize(ttargets)
        vdec_mean = self.unnormalize(vdec_mean)
        vtargets = self.unnormalize(vtargets)

        keys = random.split(key, num=4)
        fig, axes = plt.subplots(1, 8, figsize=(8, 4))
        axes[0].imshow(ttargets[0])
        axes[1].imshow(tdec_mean[0])
        axes[2].imshow(ttargets[1])
        axes[3].imshow(tdec_mean[1])
        axes[4].imshow(vtargets[0])
        axes[5].imshow(vdec_mean[0])
        axes[6].imshow(vtargets[1])
        axes[7].imshow(vdec_mean[1])
        plt.tight_layout(pad=-2.0)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # log to wandb
        # add title to plot
        fig.suptitle(f"Posterior mean at: {step}")
        self.logger.log({"posterior_samples": wandb.Image(fig)}, step=step)

    def plot_prior_samples(self, params, key, step):
        generation_key, key = random.split(key)
        pictures = self.model.apply(
            {"params": params}, key, 1.0, 0.3, method=self.model.generate
        )

        # Plot grid of generated pictures
        fig, axes32 = plt.subplots(4, 4, figsize=(10, 10))
        for i, axes8 in enumerate(axes32):
            for j, ax in enumerate(axes8):
                index = i * axes32.shape[1] + j
                ax.imshow(pictures[index])
                ax.axis("off")
        fig.suptitle(f"Samples from prior at step: {step}")
        self.logger.log({"prior_samples": wandb.Image(fig)}, step=step)

    def eval_func(self, dataset, step, params, key):
        # Test model on all images of a data loader and return avg loss
        avg_loss = 0.0
        avg_rec = 0.0
        avg_kl = 0.0
        for v_step, (batch) in zip(range(self.val_steps), dataset):
            key, model_key = random.split(key)
            inputs, targets = batch["image"], batch["image"]
            loss_value, rec, kl, metrics_dict = self.eval_step(
                inputs, targets, step, params, model_key
            )
            avg_loss += loss_value
            avg_rec += rec
            avg_kl += kl

        avg_loss /= v_step + 1
        avg_rec /= v_step + 1
        avg_kl /= v_step + 1

        metrics_dict["avg_loss"] = avg_loss

        metrics_dict["avg_rec"] = avg_rec

        metrics_dict["avg_kl"] = avg_kl

        return (
            avg_loss,
            avg_rec,
            avg_kl,
            metrics_dict["dec_mean"][:2],
            metrics_dict["dec_logstd"][:2],
            targets[:2],
            metrics_dict,
        )


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
        @jit
        def eval_step(inputs, targets, step, params, key):
            # Return the accuracy for a single batch
            enc_mean, enc_logstd, dec_mean, dec_logstd = self.model.apply(
                {"params": params}, key, inputs
            )
            loss_value, rec, kl = self.loss_func(
                dec_mean, dec_logstd, enc_mean, enc_logstd, targets, step
            )
            metrics_dict = {
                "nelbo": loss_value,
                "recons": rec,
                "kl": kl,
                "dec_mean": dec_mean,
                "dec_logstd": dec_logstd,
            }
            return loss_value, rec, kl, metrics_dict

        self.train_epoch = train_epoch
        self.eval_step = eval_step


# class EnsembleTrainer(TrainerModule):

#     def create_functions(self):
#         # Function to calculate the classification loss and accuracy for a model
#         def calculate_loss(
#             model,
#             batch,
#             key,
#         ):
#             imgs, _ = batch["image"], batch["label"]
#             keys = split(key, imgs.shape[0])
#             latents, mus, sigmas = jax.vmap(model.encode)(imgs, keys)
#             latents_reshaped = jnp.reshape(latents, (-1, model.num_decoders, model.latent_dim))
#             mus_x, sigmas_x = jax.vmap(model._decode, in_axes=0)(latents_reshaped)
#             mus_x = jnp.reshape(mus_x, (self.batch_size, -1))
#             sigmas_x = jnp.reshape(sigmas_x, (self.batch_size, -1))
#             sigmas_x = jnp.ones_like(sigmas_x)
#             imgs = jnp.squeeze(imgs.reshape(self.batch_size, -1))
#             log_prob = load_obj(self.loss_config["class_name"])(mus_x, jnp.squeeze(imgs.reshape(self.batch_size, -1)))

#             #log_prob = -1*jnp.mean(jax.vmap(model.log_prob, in_axes=(0, 0, 0))(imgs, mus_x, sigmas_x))

#             kl_loss = lambda mu, sigma: -0.5 * jnp.sum(1 + sigma - mu ** 2 - jnp.exp(sigma), axis=-1)
#             kl = jnp.mean(jax.vmap(kl_loss)(mus,sigmas))

#             loss = log_prob +  model.kl_weight * kl
#             return loss, (kl, log_prob)

#         # Training function
#         @jit
#         def train_step(model: nn.Module, opt_state: PyTree, batch, key:Array):
#             loss_fn = lambda params: calculate_loss(
#                 params,
#                 batch,
#                 key,
#             )
#             # Get loss, gradients for loss, and other outputs of loss function
#             out, grads = jax.value_and_grad(loss_fn, has_aux=True)(model)
#             updates, opt_state = self.optimizer.update(grads, opt_state, model)
#             model = self.optimizer.apply_updates(model, updates)
#             metrics_dict = {
#                 "loss_value": out[0],
#                 "recons": out[1][1],
#                 "kl": out[1][0],
#                 "grads_norm": l2_norm(grads),
#                 "grads_max": max_func(grads),
#                 "updates_norm": l2_norm(updates),
#                 "updates_max": max_func(updates),
#             }
#             return model, opt_state, metrics_dict

#         # Eval function
#         @jit
#         def eval_step(model, opt_state, batch, key: Array):

#             loss, (kl, recons) = calculate_loss(
#                 model,
#                 batch,
#                 key,
#             )
#             return loss

#         @jit
#         def reconstruct(model: nn.Module, batch, mode="mean"):
#             preds = jax.vmap(model, in_axes=0)(batch["image"])
#             reconstructed_mean = jnp.mean(preds, axis=1)
#             reconstructed_std = jnp.std(preds, axis=1)
#             ## reshape back to image
#             original_image_size = batch["image"].shape[1:]

#             loss = load_obj(self.loss_config["class_name"])
#             loss_val = jax.vmap(loss, in_axes=(1, None))(
#                 preds, jnp.squeeze(batch["image"].reshape(self.batch_size, -1))
#             )

#             reconstructed_images = {
#                 "mean": jnp.reshape(reconstructed_mean, (-1,) + original_image_size),
#                 "std": jnp.reshape(reconstructed_std, (-1,) + original_image_size),
#             }[mode]
#             return reconstructed_images, loss_val

#         self.train_step = train_step
#         self.eval_step = eval_step
#         self.reconstruct = reconstruct
