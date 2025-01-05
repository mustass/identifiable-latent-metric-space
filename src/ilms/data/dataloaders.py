from .celeba import CelebAIterator, create_pipeline
from omegaconf import DictConfig
import logging
from ilms.data import CelebAIterator, create_pipeline
from nvidia.dali.plugin.jax import DALIGenericIterator

dataset_sources = {"celeba": CelebAIterator}
dataset_pipelines = {"celeba": create_pipeline}


def get_dataloaders(cfg: DictConfig, seed: int, inference_mode=False):
    dataset = cfg["dataset_name"]
    bs = cfg["batch_size"]

    if dataset not in dataset_sources:
        raise ValueError(f"Dataset {dataset} not recognized")

    train_src = dataset_sources[dataset](bs, "train")
    val_src = dataset_sources[dataset](bs, "val")
    test_src = dataset_sources[dataset](bs, "test")

    train_pipeline = create_pipeline(bs, train_src)
    val_pipeline = create_pipeline(bs, val_src)
    test_pipeline = create_pipeline(bs, test_src)

    train_pipeline.build()
    val_pipeline.build()
    test_pipeline.build()

    train_dataset = DALIGenericIterator([train_pipeline], ["image"])
    val_dataset = DALIGenericIterator([val_pipeline], ["image"])
    test_dataset = DALIGenericIterator([test_pipeline], ["image"])

    return train_dataset, val_dataset, test_dataset
