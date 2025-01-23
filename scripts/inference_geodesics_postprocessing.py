import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from jax import config
import yaml
from tqdm import tqdm
from itertools import chain
import pickle
from flax import nnx
from pathlib import Path

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)

import wandb
from ilms.utils import (
    set_seed,
    load_obj,
    save_useful_info,
    chunks,
    pick_pairs,
)
from ilms.data import get_celeba_arrays
from jax import random
import pandas as pd
import logging
from itertools import combinations
import gc
from jax import clear_caches
import numpy as np
import jax.numpy as jnp

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

    experiments = list(
        zip(
            cfg.inference.runs,
            cfg.inference.checkpoint_names,
        )
    )

    outputs = []
    
    for checkpoint in os.listdir(cfg.general.checkpoint_path):
        look=os.path.join(cfg.general.checkpoint_path,checkpoint)
        with open(look, "rb") as f:
            checkpoint_data = pickle.load(f)
        outputs.append(checkpoint_data)
    
    
    outputs_processed = []

    for i, output in enumerate(outputs):
        processed = [
            (
                experiments[i][0].split("/")[-2],
                f"From index {res[0][0]} with label {res[1][0]} to index {res[0][1]} with label {res[1][1]}",
                res[1][0],
                res[1][1],
                res[2],
                res[3],
                res[4],
                res[5],
                res[6],
                output[-3],
                output[-2],
                output[-1],
            )
            for res in output[:-3]
        ]
        outputs_processed.append(processed)

    df_raw = pd.DataFrame(
        [
            (
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
                item[5],
                item[6],
                item[7],
                item[8],
                item[9],
                item[10],
                item[11],
            )
            for sublist in outputs_processed
            for item in sublist
        ],
        columns=[
            "checkpoint",
            "from-to",
            "from",
            "to",
            "geolength",
            "energy",
            "euclidean_latent",
            "euclidean_ambient",
            "euclids_reconstructed_ambient",
            "latent_dim",
            "seed",
            "n_ensemble",
        ],
    )

    ## write the table to csv
    ## make dir if it does not exist
    os.makedirs(cfg["general"]["checkpoint_path"], exist_ok=True)
    df_raw.to_csv(
        f'{cfg["general"]["checkpoint_path"]}/geodesics_table.csv', index=False
    )

    table_raw = wandb.Table(data=df_raw)
    wandb_logger.log({"geodesics_table_raw": table_raw})

    df_cv = df_raw.copy()
    df_cv = df_cv.groupby(["from-to", "latent_dim", "n_ensemble"]).agg(
        {
            "energy": ["mean", "std"],
            "geolength": ["mean", "std"],
            "euclidean_latent": ["mean", "std"],
            "euclidean_ambient": ["mean", "std"],
            "euclids_reconstructed_ambient": ["mean", "std"],
        }
    )

    df_cv["energy", "cv"] = df_cv["energy", "std"] / df_cv["energy", "mean"]
    df_cv["euclidean_latent", "cv"] = (
        df_cv["euclidean_latent", "std"] / df_cv["euclidean_latent", "mean"]
    )
    df_cv["euclidean_ambient", "cv"] = (
        df_cv["euclidean_ambient", "std"] / df_cv["euclidean_ambient", "mean"]
    )
    df_cv["euclids_reconstructed_ambient", "cv"] = (
        df_cv["euclids_reconstructed_ambient", "std"]
        / df_cv["euclids_reconstructed_ambient", "mean"]
    )
    df_cv["geolength", "cv"] = df_cv["geolength", "std"] / df_cv["geolength", "mean"]
    df_cv = df_cv.pipe(lambda s: s.set_axis(s.columns.map("_".join), axis=1))
    df_cv = df_cv.reset_index()

    df_cv.to_csv(
        f'{cfg["general"]["checkpoint_path"]}/geodesics_table_cv.csv',
        index=False,
    )
    table_cv = wandb.Table(data=df_cv)

    wandb_logger.log({"geodesics_table_CV": table_cv})

    df = df_raw.copy()

    df = df.pivot(
        index="from-to",
        columns="checkpoint",
        values=[
            "energy",
            "euclidean_latent",
            "euclidean_ambient",
            "euclids_reconstructed_ambient",
        ],
    ).pipe(lambda s: s.set_axis(s.columns.map("_".join), axis=1))
    df = df.reset_index()

    df.to_csv(
        f'{cfg["general"]["checkpoint_path"]}/geodesics_table_pivotted.csv',
        index=False,
    )

    from_to_df = wandb.Table(data=df)
    wandb_logger.log({"geodesics_table_pivotted": from_to_df})

    wandb.finish()


@hydra.main(config_path="../configs", config_name="config_geodesic_inference")
def launch(cfg: DictConfig):
    os.makedirs("logs", exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info(os.path.basename(__file__))

    main(cfg)


if __name__ == "__main__":
    launch()
