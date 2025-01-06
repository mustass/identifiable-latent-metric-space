import os
import logging
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import optax
import jax.random as random
from tqdm import tqdm
from ilms.utils.utils import compute_num_params, load_obj
import matplotlib.pyplot as plt
import distrax
from jax import Array
from jax.random import split
from flax import linen as nn
from jax import jit
import orbax.checkpoint as ocp
from clu import parameter_overview
from flax.training import train_state, orbax_utils
from tqdm import tqdm
import wandb


class TrainerModule:
    def __init__(self, model: nn.Module, config: DictConfig, wandb_logger):
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
        self.input_shape = config["datamodule"]["input_shape"]
        self.grad_clipping_config = config["grad_clipping"]
        self.scheduler_config = config["scheduler"]
        self.config = config

        self.seed = self.train_config["seed"]
        self.early_stopping = self.train_config["early_stopping_patience"]
        self.max_steps = self.train_config["max_steps"]
        self.val_steps = self.train_config["val_steps"]
        self.eval_every = self.train_config["eval_every"]

        self.logger = wandb_logger

        self.model_checkpoint_path = self.train_config["checkpoint"]
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model

        params = self.init_model(random.PRNGKey(self.seed))

        self.init_optimizer(self.max_steps)

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
        )

        self.loss_func = load_obj(self.loss_config["class_name"])(
            **self.loss_config["params"]
        )

        self.checkpointer = self.create_checkpoint_manager(
            self.model_checkpoint_path, 2
        )

        self.config_saved = False

    def init_model(self, key):
        x_key, init_key = random.split(key, 2)

        x = random.normal(x_key, (self.batch_size, *self.input_shape))
        variables = self.model.init(init_key, init_key, x)

        logging.info(parameter_overview.get_parameter_overview(variables))
        return variables["params"]

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
            load_obj(self.optim_config["class_name"])(
                lr_schedule, **self.optim_config["params"]
            )
        )

        self.optimizer = optax.chain(*grad_transformations)

    def train_model(self, train_loader, val_loader, random_key, logger=None):
        
        initial_step, self.state = self.load_checkpoint_if_exists(
            self.checkpointer, self.state
        )

        for step, batch in (
            pbar := tqdm(
                zip(range(initial_step, self.max_steps), train_loader),
                total=self.max_steps,
                initial=initial_step,
                desc="Training:",
            )
        ):

            random_key, model_key, key2 = random.split(random_key, 3)

            inputs, targets = batch["image"], batch["image"]
            nelbo, rec, kl, self.state, metrics_dict = self.train_step(
                inputs, targets, step, self.state, model_key
            )

            for dict_key, dict_val in metrics_dict.items():
                self.logger.log({"train_" + dict_key + "_batch": dict_val}, step=step)

            if step % self.eval_every == 0 or step == self.max_steps-1:
                tavg_loss, tavg_rec, tavg_kl, vavg_loss, vavg_rec, vavg_kl = (
                    self.eval_model(
                        train_loader, val_loader, step, self.state.params, key2
                    )
                )

                # Save new model checkpoint
                self.save_checkpoint(self.checkpointer, step, self.state)
                logging.info(f"SAVED CHECKPOINT FOR STEP {step}..")
            pbar.set_postfix_str(
                f"train_stats:  nelbo:{nelbo:.4f}  rec:{rec:.4f}  kl:{kl:.4f} | "
                f"val_stats:  nelbo:{vavg_loss:.4f}  rec:{vavg_rec:.4f}  kl:{vavg_kl:.4f}"
            )

    def eval_model(self, train_dataset, val_dataset, step, params, key):
        tkey, vkey, plot_key = random.split(key, num=3)
        tavg_loss, tavg_rec, tavg_kl, tdec_mean, tdec_logstd, ttargets = self.eval_func(
            train_dataset, step, params, tkey
        )
        vavg_loss, vavg_rec, vavg_kl, vdec_mean, vdec_logstd, vtargets = self.eval_func(
            val_dataset, step, params, vkey
        )

        # logging.info(f"\nstep {step}/{self.max_steps}  train_loss:  nelbo:{tavg_loss:.4f}  rec:{tavg_rec:.4f}  kl:{tavg_kl:.4f}      "
        # f"val_loss:  nelbo:{vavg_loss:.4f}  rec:{vavg_rec:.4f}  kl:{vavg_kl:.4f}")
        self.plot_samples(
            tdec_mean,
            tdec_logstd,
            ttargets,
            vdec_mean,
            vdec_logstd,
            vtargets,
            step,
            plot_key,
        )

        return tavg_loss, tavg_rec, tavg_kl, vavg_loss, vavg_rec, vavg_kl

    def plot_samples(
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
        keys = random.split(key, num=4)
        fig, axes = plt.subplots(1, 8, figsize=(8, 4))
        axes[0].imshow(ttargets[0])
        axes[1].imshow(
            distrax.Normal(tdec_mean[0], jnp.exp(tdec_logstd[0])).sample(seed=keys[0])
        )
        axes[2].imshow(ttargets[1])
        axes[3].imshow(
            distrax.Normal(tdec_mean[1], jnp.exp(tdec_logstd[1])).sample(seed=keys[1])
        )
        axes[4].imshow(vtargets[0])
        axes[5].imshow(
            distrax.Normal(vdec_mean[0], jnp.exp(vdec_logstd[0])).sample(seed=keys[2])
        )
        axes[6].imshow(vtargets[1])
        axes[7].imshow(
            distrax.Normal(vdec_mean[1], jnp.exp(vdec_logstd[1])).sample(seed=keys[3])
        )
        plt.tight_layout(pad=-2.0)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # log to wandb
        # add title to plot
        fig.suptitle(f"Reconstructs Step {step}")
        self.logger.log({"plot": wandb.Image(fig)})

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

        return (
            avg_loss,
            avg_rec,
            avg_kl,
            metrics_dict["dec_mean"][:2],
            metrics_dict["dec_logstd"][:2],
            targets[:2],
        )

    def create_checkpoint_manager(self, checkpoint_path, max_allowed_checkpoints=2):
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_allowed_checkpoints, create=True
        )
        return ocp.CheckpointManager(
            os.path.abspath(checkpoint_path),
            ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
            options=options,
        )

    def save_checkpoint(self, checkpoint_manager, step, state):
        ckpt = {"state": state, "step": step}
        save_args = orbax_utils.save_args_from_target(ckpt)

        checkpoint_manager.save(step, ckpt, save_kwargs={"save_args": save_args})

        if self.config_saved is False:
            with open(
                os.path.join(checkpoint_manager._directory, "config.yml"), "w"
            ) as f:
                OmegaConf.save(self.config, f)
            self.config_saved = True

    def load_checkpoint_if_exists(self, checkpoint_manager, state):
        if checkpoint_manager.latest_step() is not None:
            print(
                f"Loading checkpoint state for step {checkpoint_manager.latest_step()} from {checkpoint_manager._directory}"
            )
            step = checkpoint_manager.latest_step()
            target = {"state": state, "step": 0}
            ckpt = checkpoint_manager.restore(step, items=target)
            return ckpt["step"], ckpt["state"]

        print(
            f"Couldn't find state checkpoints to load in {checkpoint_manager._directory}!!"
        )
        return 0, state


class Trainer(TrainerModule):
    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(
            params,
            inputs,
            targets,
            step,
            key,
        ):
            enc_mean, enc_logstd, dec_mean, dec_logstd = self.model.apply(
                {"params": params}, key, inputs
            )
            loss_value, rec, kl = self.loss_func(
                dec_mean, dec_logstd, enc_mean, enc_logstd, targets, step
            )
            return loss_value, (rec, kl)

        # Training function
        @jit
        def train_step(inputs, targets, step, state, key):

            loss_fn = lambda params: calculate_loss(
                params,
                inputs,
                targets,
                step,
                key,
            )
            # Get loss, gradients for loss, and other outputs of loss function
            (nelbo, (rec, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            state = state.apply_gradients(grads=grads)

            metrics_dict = {
                "nelbo": nelbo,
                "recons": rec,
                "kl": kl,
            }
            return nelbo, rec, kl, state, metrics_dict

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

        self.train_step = train_step
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
