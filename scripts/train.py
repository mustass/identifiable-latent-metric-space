import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
from jax import config
import yaml
import pathlib as pl
import tensorflow as tf

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)
import wandb

from ilms.utils import set_seed, load_obj, save_useful_info, init_decoder_ensamble
from ilms.data import get_dataloaders
from jax import random

import logging

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs visible: {gpus}")
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


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

    key, key1, key2, key3, key4, random_key = random.split(random_key, 6)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg["datamodule"],
        19821,
    )

    model = load_obj(cfg["model"]["class_name"])(
        cfg.datamodule.batch_size, cfg.datamodule.input_shape
    )

    trainer = load_obj(cfg["training"]["class_name"])(model, cfg, wandb_logger)

    key, key1, key2, key3, random_key = random.split(random_key, 5)

    trainer.train_model(train_loader, val_loader, random_key=key)
    tavg_loss, tavg_rec, tavg_kl, vavg_loss, vavg_rec, vavg_kl = trainer.eval_model(
        test_loader,
        val_loader,
        cfg.training.max_steps + 1,
        trainer.state.params,
        key1,
    )
    results = {
        "test_loss": tavg_loss,
        "test_rec": tavg_rec,
        "test_kl": tavg_kl,
        "val_loss": vavg_loss,
        "val_rec": vavg_rec,
        "val_kl": vavg_kl,
    }
    logging.info(f"Results: {results}")
    trainer.logger.log(results)
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
