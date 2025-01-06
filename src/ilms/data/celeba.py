import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types


class CelebAIterator(object):
    def __init__(self, batch_size, set_name: str = "train"):
        self.images_dir = "/home/stasy/ssdb/celeba_manual/images/img_align_celeba/"
        self.batch_size = batch_size
        with open(
            "/home/stasy/ssdb/celeba_manual/images/" + "list_eval_partition.txt", "r"
        ) as f:
            self.files = [line.rstrip() for line in f if line != ""]

        if set_name not in ["train", "val", "test"]:
            raise ValueError("set_name must be one of 'train', 'val', 'test'")

        set_int = {"train": 0, "val": 1, "test": 2}[set_name]
        self.files = [
            line.split(" ")[0]
            for line in self.files
            if line.split(" ")[1] == str(set_int)
        ]
        print(f"Found {len(self.files)} images in {set_name} set")

        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    @property
    def num_outputs(self):
        return 1

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            jpeg_filename = self.files[self.i]
            f = open(self.images_dir + jpeg_filename, "rb")
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            self.i = (self.i + 1) % self.n

        return (batch,)


def create_pipeline(batch_size, image_dims, src):
    pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    with pipe:
        jpegs = fn.external_source(source=src, num_outputs=1, dtype=types.UINT8)
        decode = fn.decoders.image(jpegs, device="mixed")/255.0
        decode = fn.crop_mirror_normalize(
            decode,
            dtype=types.FLOAT,
            std=[0.5, 0.5, 0.5],
            mean=[0.5, 0.5, 0.5],
            crop=(148, 148),
        )
        decode = fn.resize(decode, resize_x=image_dims[0], resize_y=image_dims[1])
        decode = fn.transpose(decode, perm=[1, 2, 0])
        pipe.set_outputs(decode)
    return pipe
