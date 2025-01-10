#from .celeba_dali import CelebAIterator, create_pipeline
from omegaconf import DictConfig
import logging
#from nvidia.dali.plugin.jax import DALIGenericIterator
import tensorflow_datasets as tfds
import os
import tensorflow as tf



#dataset_sources = {"celeba": CelebAIterator}
#dataset_pipelines = {"celeba": create_pipeline}


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

def get_dataloaders_tfds(
    cfg: DictConfig, seed: int, inference_mode=False):
    dataset_name = cfg["dataset_name"]
    batch_size = cfg["batch_size"]

    dataset = tfds.load(dataset_name)

    def preprocess_val_test(example, image_size=(64, 64)):
        image = example["image"]
        image = tf.image.central_crop(image, central_fraction=140/178)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    def map_fn(sample):
        img = sample["image"]
        img = tf.image.resize(img,[64,64],tf.image.ResizeMethod.BILINEAR)/255
        return {'image': img, 'label': sample["labels"]}

    def preprocess_train(example, label, image_size=(64, 64), augment=True):
    # Resize image to desired dimensions
        # Normalize pixel values to the range [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Crop 140x140 in the center
        image = tf.image.central_crop(image, central_fraction=140/178)
        # Apply augmentations (if augment is True)
        if augment:
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            
            # Random vertical flip
            image = tf.image.random_flip_up_down(image)

            # Random rotation
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            
            # Random brightness adjustment
            image = tf.image.random_brightness(image, max_delta=0.2)
            
            # Random contrast adjustment
            image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
            
            # Random saturation adjustment
            image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        
        image = tf.image.resize(example, image_size)
        return image
    
    train_dataset = (
        dataset["train"]
        .map(map_fn)
        .shuffle(1000, reshuffle_each_iteration=True) #buffer_size=dataset["train"].cardinality())
        .batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE,drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )
    val_dataset = (
        dataset["validation"]
        .map(map_fn)
        .batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE,drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )


    test_dataset = (
        dataset["test"]
        .map(map_fn)
        .batch(batch_size,num_parallel_calls = tf.data.AUTOTUNE,drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )

    return train_dataset, val_dataset, test_dataset