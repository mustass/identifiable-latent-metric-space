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

def run_experiment(cfg: DictConfig, wandb_logger, test_dataset, pairs):
    set_seed(cfg["training"]["seed"])
    random_key = random.PRNGKey(cfg["training"]["seed"])
    key, key1, key2, key3, key4, random_key = random.split(random_key, 6)

    model = load_obj(cfg["model"]["class_name"])(
        opts=cfg["model"]["params"], rngs=nnx.Rngs(random_key))
    
    model.load(os.path.join(cfg.training.checkpoint, cfg.training.checkpoint_name))

    trainer = load_obj(cfg["inference"]["class_name"])(model, cfg, wandb_logger)

    point_pairs = pairs

    best_energies = []
    distances = []
    histories = []
    labels_pairs = []
    euclids_latent = []
    euclids_ambient = []
    euclids_reconstructed_ambient = []

    for i, batch in enumerate(
        bar := tqdm(
            chunks(point_pairs, cfg["inference"]["geodesics_params"]["batch_size"]),
            desc="Running batches of geodesics",
        )
    ):
        key_geodesics, random_key = random.split(random_key, 2)

        input = []
        labels = []
        norms_ambient = []
        for i, pair in enumerate(batch):
            input.append(
                jnp.array([test_dataset[pair[0]], test_dataset[pair[1]]])
            )
            norms_ambient.append(
                jnp.linalg.norm(
                    jnp.ravel(jnp.array(test_dataset[pair[0]]))
                    - jnp.ravel(jnp.array(test_dataset[pair[1]])),
                    ord=2,
                ).item()
            )
        
        labels = [("1","1")]*(i+1)

        (
            best_sum_of_energies,
            lengths,
            best_single_energies,
            history,
            eucleadian_dists,
            eucleadian_reconstructed_ambient,
        ) = trainer.compute_geodesic(jnp.array(input), key_geodesics)

        distances.append(lengths)
        best_energies.append(best_single_energies.tolist())
        labels_pairs.append(labels)
        histories.append(history)
        euclids_reconstructed_ambient.append(eucleadian_reconstructed_ambient.tolist())
        euclids_latent.append(eucleadian_dists.tolist())
        euclids_ambient.append(norms_ambient)
        if i % 10 == 0:
            clear_caches()

    distances = list(chain.from_iterable(distances))
    best_energies = list(chain.from_iterable(best_energies))
    labels_pairs = list(chain.from_iterable(labels_pairs))
    euclids_latent = list(chain.from_iterable(euclids_latent))
    euclids_ambient = list(chain.from_iterable(euclids_ambient))
    euclids_reconstructed_ambient = list(
        chain.from_iterable(euclids_reconstructed_ambient)
    )
    print(len(distances), len(best_energies), len(labels_pairs), len(euclids_latent), len(euclids_ambient), len(euclids_reconstructed_ambient)   )

    return list(
        zip(
            point_pairs,
            labels_pairs,
            distances,
            best_energies,
            euclids_latent,
            euclids_ambient,
            euclids_reconstructed_ambient,
        )
    )


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

    _config = OmegaConf.load(Path(experiments[0][0]) / "config.yml")
    _, _, _, _, test_images, test_labels = get_celeba_arrays(
        _config["datamodule"]["dataset_root"])

    point_pairs = pick_pairs(
        test_images,
        test_labels,
        cfg.inference.num_pairs,

    )

    logging.info(
        f"Producing following {len(point_pairs)} geodesic pairs: {str(point_pairs)}"
    )

    outputs = []
    for experiment in experiments:
        config = OmegaConf.load(Path(experiment[0]) / "config.yml")
        config.training.checkpoint = experiment[0]
        config.training.checkpoint_name = experiment[1]
        with open_dict(config):
            config.inference = cfg.inference
        output = run_experiment(
            config,
            wandb_logger,
            test_images,
            point_pairs,
        )
        output.append(config.model.params.z_dim)
        output.append(config.training.seed)
        output.append(
            config.model.params.num_decoders
        )
        outputs.append(output)

        file_path = os.path.join(cfg["general"]["checkpoint_path"], os.path.basename(os.path.normpath(experiment[0])))
        filename = str(file_path)+".pickle"
        os.makedirs(cfg["general"]["checkpoint_path"], exist_ok=True)
        with open(filename, "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()

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
