import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
from jax import config
import yaml

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)
import wandb

from ilms.utils import set_seed, load_obj, save_useful_info, init_decoder_ensamble
from ilms.data.dataloaders import get_dataloaders
from jax import random

import logging


def main(cfg: DictConfig, pretrained=False):
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
        trainer.state["params"],
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
    print(results)

    wandb.finish()


@hydra.main(config_path="../configs", config_name="config")
def launch(cfg: DictConfig):
    os.makedirs("logs", exist_ok=True)
    logging.info(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info(os.path.basename(__file__))

    cfg.general.model_checkpoints_path = os.path.join(
        get_original_cwd(), cfg.general.model_checkpoints_path
    )
    pretrained = False

    if cfg.training.checkpoint_path is not None:
        logging.info(f"Resuming training from checkpoint {cfg.training.checkpoint}")
        config = OmegaConf.load(cfg.training.checkpoint + "config.yml")
        config.training = cfg.training
        cfg = config
        pretrained = True
        logging.info(f"Running with new config:")
        logging.info(cfg)

    main(cfg, pretrained)


if __name__ == "__main__":
    launch()
