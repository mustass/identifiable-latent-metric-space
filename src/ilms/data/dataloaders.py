# from .celeba_dali import CelebAIterator, create_pipeline
from omegaconf import DictConfig
import logging

# from nvidia.dali.plugin.jax import DALIGenericIterator
import tensorflow_datasets as tfds
import os
import tensorflow as tf

IMAGENET_MEAN = [0.5,0.5,0.5] 

IMAGENET_STD = [0.5,0.5,0.5]

mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
std = tf.constant(IMAGENET_STD, dtype=tf.float32)

# dataset_sources = {"celeba": CelebAIterator}
# dataset_pipelines = {"celeba": create_pipeline}


# def get_dataloaders_dali(cfg: DictConfig, seed: int, inference_mode=False):
#     dataset = cfg["dataset_name"]
#     bs = cfg["batch_size"]
#     image_dims = cfg["input_shape"]
#     images_dir = cfg["data_path"]

#     if dataset not in dataset_sources:
#         raise ValueError(f"Dataset {dataset} not recognized")

#     train_src = dataset_sources[dataset](bs, images_dir, "train")
#     val_src = dataset_sources[dataset](bs, images_dir, "val")
#     test_src = dataset_sources[dataset](bs, images_dir, "test")

#     train_pipeline = create_pipeline(bs, image_dims, train_src)
#     val_pipeline = create_pipeline(bs, image_dims, val_src)
#     test_pipeline = create_pipeline(bs, image_dims, test_src)

#     train_pipeline.build()
#     val_pipeline.build()
#     test_pipeline.build()

#     train_dataset = DALIGenericIterator([train_pipeline], ["image"])
#     val_dataset = DALIGenericIterator([val_pipeline], ["image"])
#     test_dataset = DALIGenericIterator([test_pipeline], ["image"])

#     return train_dataset, val_dataset, test_dataset


def get_dataloaders_tfds(cfg: DictConfig, seed: int, inference_mode=False):
    dataset_name = cfg["dataset_name"]
    batch_size = cfg["batch_size"]

    dataset = tfds.load(dataset_name)

    def preprocess_val_test(example, image_size=(64, 64)):
        image = example["image"]
        image = tf.image.central_crop(image, central_fraction=140 / 178)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def map_fn(sample):
        img = sample["image"]
        img = tf.image.resize(img, [64, 64], tf.image.ResizeMethod.BILINEAR) / 255.0
        # normalize to imagenet stats
        img = (img - mean) / std
        return {"image": img, "label": sample["labels"]}

    def preprocess_train(sample, image_size=(64, 64), augment=True):
        img = sample["image"]
        # Apply augmentations (if augment is True)
        if augment:
            # Random horizontal flip
            img = tf.image.random_flip_left_right(img)
        
        img = tf.image.resize(img, image_size, tf.image.ResizeMethod.BILINEAR)/ 255.0
        img = (img - mean) / std
        return {"image": img, "label": sample["labels"]}

    train_dataset = (
        dataset["train"]
        .map(preprocess_train)
        .shuffle(
            1000, reshuffle_each_iteration=True
        )  # buffer_size=dataset["train"].cardinality())
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )
    val_dataset = (
        dataset["validation"]
        .map(map_fn)
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )

    test_dataset = (
        dataset["test"]
        .map(map_fn)
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )

    return train_dataset, val_dataset, test_dataset
