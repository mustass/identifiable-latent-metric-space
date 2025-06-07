import importlib
from typing import Any
import logging
import os
import shutil
from jax import tree_flatten
import jax.numpy as jnp
import hydra
import jax
import numpy as np
import jax.random as random
from itertools import product, combinations, chain
import random as pyrandom


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def compute_num_params(pytree):
    return sum(
        x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(pytree)
    )


def save_useful_info(name) -> None:
    logging.info(hydra.utils.get_original_cwd())
    logging.info(os.getcwd())
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), "src"),
        os.path.join(hydra.utils.get_original_cwd(), f"{os.getcwd()}/code/src"),
    )
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), "data"),
        os.path.join(hydra.utils.get_original_cwd(), f"{os.getcwd()}/code/data"),
    )
    shutil.copy2(
        os.path.join(f"{hydra.utils.get_original_cwd()}/scripts", name),
        os.path.join(hydra.utils.get_original_cwd(), os.getcwd(), "code"),
    )


def l2_norm(tree):
    """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = tree_flatten(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def max_func(tree):
    leaves, _ = tree_flatten(tree)
    return jnp.max(
        jnp.concatenate([jnp.asarray(abs(x)).ravel() for x in leaves], axis=None)
    )


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def pick_pairs(images_array, labels_array, num_pairs, seed):
    # pick random from_images, give indices only
    np.random.seed(seed)
    indices_from = np.random.choice(images_array.shape[0], num_pairs, replace=False)
    indices_to = np.random.choice(images_array.shape[0], num_pairs, replace=False)

    # make sure that  from and to indices are not the same
    for i in range(num_pairs):
        while indices_to[i] == indices_from[i]:
            indices_to[i] = np.random.choice(images_array.shape[0])

    # return zipped indices
    return list(zip(indices_from, indices_to))
