"""celebahq64 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for celebahq64 dataset."""

    VERSION = tfds.core.Version("1.0.2")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(celebahqhq): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(148, 148, 3)),
                    "labels": tfds.features.FeaturesDict(
                        {
                            "gender": tfds.features.ClassLabel(
                                names=["female", "male"]
                            ),
                            "glasses": tfds.features.ClassLabel(
                                names=["no_glasses", "glasses"]
                            ),
                            "smile": tfds.features.ClassLabel(
                                names=["no_smile", "smile"]
                            ),
                            "young": tfds.features.ClassLabel(names=["old", "young"]),
                            "hat": tfds.features.ClassLabel(names=["no_hat", "hat"]),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # TODO(celebahqhq): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples("train"),
            "validation": self._generate_examples("valid"),
            "test": self._generate_examples("test"),
        }

    def _generate_examples(self, split="train"):
        """Yields examples."""
        # TODO(celebahqhq): Yields (key, example) tuples from the dataset

        from datasets import load_dataset
        import numpy as np

        dataset = load_dataset("flwrlabs/celeba", cache_dir="~/ssdb/hf_cache")
        print(f"DATASET KEYS: {dataset.keys()}")
        dataset = dataset[split].to_tf_dataset()

        def deserialization_fn(data):
            image = data["image"]
            image = tf.image.crop_to_bounding_box(image, 40, 15, 148, 148)
            # image = tf.image.resize(image, (64, 64), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
            return {
                "image": image,
                "gender": data["Male"],
                "glasses": data["Eyeglasses"],
                "smile": data["Smiling"],
                "young": data["Young"],
                "hat": data["Wearing_Hat"],
            }

        dataset = dataset.map(deserialization_fn)
        dataset = tfds.as_numpy(dataset)
        for i, example in enumerate(dataset):
            yield i, {
                "image": example["image"].astype(np.uint8),
                "labels": {
                    "gender": example["gender"],
                    "glasses": example["glasses"],
                    "smile": example["smile"],
                    "young": example["young"],
                    "hat": example["hat"],
                },
            }
