import numpy as np
from pathlib import Path

files_celeba = {
    "train_images": "celeba_train_images.npy",
    "train_labels": "celeba_train_labels.npy",
    "val_images": "celeba_val_images.npy",
    "val_labels": "celeba_val_labels.npy",
    "test_images": "celeba_test_images.npy",
    "test_labels": "celeba_test_labels.npy", 
}

files = {
    "train_images": "train_images.npy",
    "train_labels": "train_labels.npy",
    "val_images": "val_images.npy",
    "val_labels": "val_labels.npy",
    "test_images": "test_images.npy",
    "test_labels": "test_labels.npy", 
}

def get_arrays(root:str, files:dict=files):
    
    # check if the files exist
    for key, value in files.items():
        if not Path(root) / value:
            raise FileNotFoundError(f"{value} not found in {root}. Consider running the script to generate the files.")
    
    # load the files
    train_images = np.load(Path(root) / files["train_images"])
    # train_labels = np.load(Path(root) / files["train_labels"])
    val_images = np.load(Path(root) / files["val_images"])
    # val_labels = np.load(Path(root) / files["val_labels"])
    test_images = np.load(Path(root) / files["test_images"])
    # test_labels = np.load(Path(root) / files["test_labels"])

    return train_images, None, val_images, None, test_images, None

