from typing import Any, List
import equinox as eqx
from jax import Array
import jax.numpy as jnp

from flowjax.bijections import (
    AbstractBijection,
    Identity,
)

from jaxmflow.transforms.projections import ProjectionSplit

import equinox as eqx
from jax import Array

from flowjax.bijections import AbstractBijection


class BaseAE(eqx.Module):
    """Base class for diffeomorphic autoencoders.

    Flows are bijective transformations that can be used to transform samples from a
    simple distribution (e.g. a Gaussian) into samples from a more complex distribution.
    """

    outer_transform: AbstractBijection

    def __init__(self, outer_transform: AbstractBijection):
        """Initialise a flow.

        Args:
            bijections: List of bijections.
        """
        self.outer_transform = outer_transform

    def forward(self, x: Array, condition: Array | None = None) -> Array:
        """Forward transformation.

        Args:
            x: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        raise NotImplementedError

    def inverse(self, y: Array, condition: Array | None = None) -> Array:
        """Inverse transformation.

        Args:
            y: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        raise NotImplementedError


class ManifoldAE(BaseAE):
    """Manifold Autoencoder.

    This Autoencoder is designed to learn a low-dimensional representation of the data
    using diffeomorphisms and slice & padding operations.
    """

    outer_transform: AbstractBijection
    projection: eqx.Module
    total_data_dim: int
    total_latent_dim: int

    def __init__(self, data_dim, latent_dim, outer_transform: AbstractBijection):
        """Initialise a flow.

        Args:
            bijections: List of bijections.
        """
        self.outer_transform = outer_transform
        self.total_data_dim = data_dim if isinstance(data_dim, int) else data_dim[0] * data_dim[1] * data_dim[2]
        self.total_latent_dim = (
            latent_dim if isinstance(latent_dim, int) else latent_dim[0] * latent_dim[1] * latent_dim[2]
        )
        self.projection = ProjectionSplit(self.total_data_dim, self.total_latent_dim)

    def __call__(self, x: Array) -> Any:
        """Forward transformation.

        Args:
            x: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        x = jnp.squeeze(x.reshape(-1, self.total_data_dim))
        # u, hm, ho = self._encode(x)
        u = self.encode(x)
        # h = self._decode(u)
        h = self.decode(u)

        return h

    def _encode(self, x: Array, condition: Array | None = None) -> tuple:
        """Forward transformation.

        Args:
            x: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        h = self.outer_transform.transform(x, condition)
        h_manifold, h_orthogonal = self.projection.transform(h)
        u = h_manifold  # something weird going on with the Identity bijection
        return u, h_manifold, h_orthogonal

    def _decode(self, u: Array, u_orthogonal: Array = None, condition: Array | None = None) -> tuple:
        """Inverse transformation.

        Args:
            y: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        h_manifold = jnp.squeeze(u)
        h = self.projection.inverse(h_manifold)
        x = self.outer_transform.inverse(h, condition)
        return x

    def encode(self, x: Array, condition: Array | None = None) -> tuple:
        """Forward transformation.

        Args:
            x: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        x = jnp.squeeze(x.reshape(-1, self.total_data_dim))
        h = self.outer_transform.transform(x, condition)
        h_manifold, h_orthogonal = self.projection.transform(h)
        return h_manifold

    def decode(self, u: Array, u_orthogonal: Array = None, condition: Array | None = None) -> tuple:
        """Inverse transformation.

        Args:
            y: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        h = jnp.squeeze(u)
        h = self.projection.inverse(h)
        x = self.outer_transform.inverse(h, condition)
        return x


class EncoderManifoldAE(BaseAE):
    outer_transform: AbstractBijection
    encoder: eqx.Module
    projection: eqx.Module
    total_data_dim: int
    total_latent_dim: int

    def __init__(self, data_dim, latent_dim, outer_transform: AbstractBijection, encoder: eqx.Module):
        """Initialise a flow.

        Args:
            bijections: List of bijections.
        """
        self.outer_transform = outer_transform
        self.total_data_dim = data_dim if isinstance(data_dim, int) else data_dim[0] * data_dim[1] * data_dim[2]
        self.total_latent_dim = (
            latent_dim if isinstance(latent_dim, int) else latent_dim[0] * latent_dim[1] * latent_dim[2]
        )
        self.projection = ProjectionSplit(self.total_data_dim, self.total_latent_dim)
        self.encoder = encoder

    def __call__(self, x: Array, condition: Array | None = None) -> Any:
        """Forward transformation.

        Args:
            x: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """

        # x = jnp.expand_dims(x,axis=0) # (c, h, w)

        u = self.encode(x)
        h = self.decode(u)

        return h

    def encode(self, x: Array, condition: Array | None = None) -> Any:
        """Forward transformation.

        Args:
            x: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        h = self.encoder(x)

        return h

    def decode(self, u: Array, u_orthogonal: Array = None, condition: Array | None = None) -> Any:
        """Inverse transformation.

        Args:
            y: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        h = jnp.squeeze(u)
        h = self.projection.inverse(h)
        x = self.outer_transform.inverse(h, condition)
        return x


class EnsambleManifoldAE(eqx.Module):
    decoders: eqx.Module
    encoder: eqx.Module
    projection: eqx.Module
    total_data_dim: int
    total_latent_dim: int
    num_decoders: int

    def __init__(self, data_dim, latent_dim, n_ensamble, decoders: eqx.Module, encoder: eqx.Module):
        """Initialise a flow.

        Args:
            bijections: List of bijections.
        """
        self.decoders = decoders
        self.num_decoders = n_ensamble
        self.total_data_dim = data_dim if isinstance(data_dim, int) else data_dim[0] * data_dim[1] * data_dim[2]
        self.total_latent_dim = (
            latent_dim if isinstance(latent_dim, int) else latent_dim[0] * latent_dim[1] * latent_dim[2]
        )
        self.projection = ProjectionSplit(self.total_data_dim, self.total_latent_dim)
        self.encoder = encoder

    def __call__(self, x: Array, condition: Array | None = None) -> Any:
        """Forward transformation.

        Args:
            x: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """

        # x = jnp.expand_dims(x,axis=0) # (c, h, w)
        u = self.encode(x)
        h = self.decode(u)

        return h

    def encode(self, x: Array, condition: Array | None = None) -> Any:
        """Forward transformation.

        Args:
            x: Input.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Transformed input.
        """
        h = self.encoder(x)

        return h

    def _decode(self, u: Array) -> Any:
        """
        For training use as the function is batched
        as a batch for each decoder
        """

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), 0))
        def _decode_per_ensamble(model, x):
            return model.transform(x)

        return _decode_per_ensamble(self.decoders, u)

    def decode(self, u: Array) -> Any:
        """
        Not batched, will evaluate each decoder on the input.
        """
        h = jnp.squeeze(u)
        h = self.projection.inverse(h)
        h = jnp.expand_dims(h, 1)
        u = jnp.repeat(h, self.num_decoders, 1)
        u = jnp.swapaxes(u, 0, 1)

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), 0))
        def _decode_per_ensamble(model, x):
            return model.transform(x)

        return _decode_per_ensamble(self.decoders, u)
