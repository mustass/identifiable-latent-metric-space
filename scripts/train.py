import warnings
warnings.filterwarnings("ignore")
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
from jax import config
import yaml
import pathlib as pl
import jax
from flax import nnx

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)
import wandb

from ilms.utils import set_seed, load_obj, save_useful_info
from ilms.data import get_celeba_arrays
from jax import random

import logging


def main(cfg: DictConfig):
    wandb_logger = wandb.init(
        project=cfg["general"]["project_name"],
        name=cfg["general"]["run_name"],
        entity=cfg["general"]["workspace"],
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        handlers=[
            logging.FileHandler(f"{wandb_logger.dir}/pythonlog.txt"),
            logging.StreamHandler(),
        ],
    )

    set_seed(cfg["training"]["seed"])
    random_key = random.PRNGKey(cfg["training"]["seed"])

    train_images, train_labels, val_images, val_labels, test_images, test_labels = (
        get_celeba_arrays(cfg["datamodule"]["dataset_root"])
    )

    train_images = jax.device_put(train_images)
    val_images = jax.device_put(val_images)

    model = load_obj(cfg["model"]["class_name"])(
        opts=cfg["model"]["params"], rngs=nnx.Rngs(random_key)
    )

    if cfg.training.resume:
        model.load(os.path.join(cfg.training.checkpoint, cfg.training.checkpoint_name))

    trainer = load_obj(cfg["training"]["class_name"])(model, cfg, wandb_logger)

    trainer.train_model(train_images, val_images, cfg["training"]["num_epochs"])
    test_stats = trainer.eval_model(test_images, cfg["training"]["num_epochs"]+2)
    logging.info(f"Test results {test_stats}")
    wandb.finish()


@hydra.main(config_path="../configs", config_name="config")
def launch(cfg: DictConfig):
    os.makedirs("logs", exist_ok=True)
    logging.info(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info(os.path.basename(__file__))

    cfg.training.checkpoint = os.path.join(get_original_cwd(), cfg.training.checkpoint)

    checkpoint_path = pl.Path(cfg.training.checkpoint)
    if checkpoint_path.exists() and not cfg.training.resume:
        logging.info(f"Checkpoint {checkpoint_path} already exists")
        i = 1
        while checkpoint_path.with_name(f"{checkpoint_path.name}_{i}").exists():
            i += 1
        checkpoint_path.rename(checkpoint_path.with_name(f"{checkpoint_path.name}_{i}"))
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logging.info(
            f"Created a new checkpoint folder {checkpoint_path} \n old checkpoint moved to {checkpoint_path.with_name(f'{checkpoint_path.name}_{i}')}"
        )
    else:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    if cfg.training.resume:
        logging.info(f"Resuming training from checkpoint {cfg.training.checkpoint}")
        config = OmegaConf.load(cfg.training.checkpoint + "config.yml")
        config.training = cfg.training
        cfg = config
        logging.info(f"Running with new config:")
        logging.info(cfg)

    main(cfg)


if __name__ == "__main__":
    launch()
