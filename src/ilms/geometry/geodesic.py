from jax import jit, vmap, jacfwd, Array, jacrev
from jax.random import PRNGKey, normal, uniform, choice, split
import jax.numpy as jnp
from typing import List, Literal
from flax import linen as nn
from flax import nnx


class Geodesics(nn.Module):
    bases: Array  # (n_geodesics, basis_dim)
    params: Array  # (n_geodesics, n_params)
    n_poly: int
    n_geodesics: int
    model: nnx.Module
    point_pairs: Array  # (n_geodesics, 2, dim)
    dim: int

    def __init__(
        self,
        model: nnx.Module,
        n_poly: int,
        point_pairs: Array,
        key: PRNGKey,
        init_mode: str = "normal",
        init_scale: float = 1.0,
    ):
        self.model = model
        self.n_poly = n_poly
        self.n_geodesics = len(point_pairs)
        self.bases = self.init_bases()
        self.dim = point_pairs.shape[-1]
        self.params = self.init_params(key, init_mode, init_scale)
        self.point_pairs = point_pairs

    def calculate_energy(
        self,
        t,
        key,
        mode: Literal["bruteforce"],
        derivative="delta",
        metric_mode="single",
        n_ensemble=None,
    ):
        return self._calculate_energy_bruteforce(t, key, derivative, metric_mode, n_ensemble)

    def calculate_length(
        self, t, key, derivative="delta", metric_mode="single", n_ensemble=None
    ):
        x = self.eval(t)  # (n_geodesics, dim, len(t))
        x = x.transpose(0, 2, 1)  # (n_geodesics, len(t), dim)

        decoded = vmap(vmap(self.model.decode, in_axes=0), in_axes=0)(
            x
        )  # (n_geodesics, len(t), n_ensemble, 784)
        if metric_mode == "ensemble":

            def sample(key, decoded):
                idx = choice(key, decoded.shape[0], shape=(1,))
                return jnp.squeeze(decoded[idx])

            split_func = lambda key: split(key, decoded.shape[1])
            keys = vmap(split_func)(split(key, decoded.shape[0]))

            decoded = vmap(vmap(sample))(keys, decoded)
        else:
            decoded = decoded[:, :, 0, :]

        def norm_of_differences(input):
            a, b = input[1:, :], input[:-1, :]
            difference = a - b
            return vmap(lambda x: jnp.linalg.norm(x, ord=2))(difference)

        result = jnp.sum(vmap(norm_of_differences)(decoded), axis=1)

        return result

    def _calculate_energy_bruteforce(
        self, t, key, derivative="delta", metric_mode="single", n_ensemble=None
    ):
        x = self.eval(t)  # (n_geodesics, dim, len(t))
        x = x.transpose(0, 2, 1)  # (n_geodesics, len(t), dim)
        decoded = vmap(vmap(self.model.decode, in_axes=0), in_axes=0)(
            x
        )  # (n_geodesics, len(t), n_ensemble, 784)
        if metric_mode == "ensemble":

            def sample(key, decoded):
                idx = choice(key, decoded.shape[0], shape=(1,))
                return jnp.squeeze(decoded[idx])

            split_func = lambda key: split(key, decoded.shape[1])
            keys = vmap(split_func)(split(key, decoded.shape[0]))

            decoded = vmap(vmap(sample))(keys, decoded)
        else:
            decoded = decoded[:, :, 0, :]

        def squared_norm_of_differences(input):
            a, b = input[1:, :], input[:-1, :]
            difference = a - b
            return vmap(lambda x: jnp.dot(x, x))(difference)

        result = jnp.sum(vmap(squared_norm_of_differences)(decoded), axis=1) * len(t)

        return result

    def delta(self, t, trajectory):
        return (trajectory[:, 1:] - trajectory[:, :-1]) * (len(t) - 1)

    def init_bases(self):
        return vmap(self._basis)(jnp.arange(self.n_geodesics))

    def init_params(self, key, mode="normal", scale=1.0):
        shape = (self.n_geodesics, self.bases.shape[2], self.dim)

        if mode == "normal":
            return scale * normal(key, shape)
        elif mode == "uniform":
            return scale * uniform(key, shape)
        else:
            return jnp.zeros(shape)

    def _basis(self, nonarg):
        # region Boundary conditions
        # Boundary conditions for a third-degree polynomial
        # starting at t=0 and ending at t=1:
        # [[0^0, 0^1, 0^2, 0^3], == [[1, 0, 0, 0],
        #  [1^0, 1^1, 1^2, 1^3]] ==  [1, 1, 1, 1]]
        #
        # ^ note two equations and 4 unknowns
        # boundary = array([[0],[1]])**arange(4)
        # endregion
        np = self.n_poly  # number of polynomials in spline
        tc = jnp.linspace(0, 1, np + 1)[1:-1]  # time cutoffs between polynomials

        boundary = jnp.zeros((2, 4 * np))
        boundary = boundary.at[0, 0].set(1)
        boundary = boundary.at[1, -4:].set(1)

        zeroth, first, second = jnp.zeros((3, np - 1, 4 * np))
        for i in range(np - 1):
            si = 4 * i  # start index
            fill_0 = jnp.array([1.0, tc[i], tc[i] ** 2, tc[i] ** 3])
            zeroth = zeroth.at[i, si : (si + 4)].set(fill_0)
            zeroth = zeroth.at[i, (si + 4) : (si + 4 * 2)].set(-fill_0)

            fill_1 = jnp.array([0.0, 1.0, 2.0 * tc[i], 3.0 * tc[i] ** 2])
            first = first.at[i, si : (si + 4)].set(fill_1)
            first = first.at[i, (si + 4) : (si + 4 * 2)].set(-fill_1)

            # fill_2 = array([0.0, 0.0, 6.0 * tc[i], 2.0])
            fill_2 = jnp.array([0.0, 0.0, 2.0, 6.0 * tc[i]])
            second = second.at[i, si : (si + 4)].set(fill_2)
            second = second.at[i, (si + 4) : (si + 4 * 2)].set(-fill_2)

        constraints = jnp.r_[boundary, zeroth, first, second]

        # region Nullspace comment
        # get the Nullspace of the boundary conditions
        # such that we can vary free parameters but at
        # t=0 and t=1 the polynomial is constrained to
        # be zero. _eval_line() will take care of the
        # linear interpolation between the endpoints and
        # _eval_poly() will take care of the polynomial
        # diviations from the line.
        # endregion
        _, S, VT = jnp.linalg.svd(constraints)

        return VT.T[:, S.size :]

    def eval(self, t):
        # vmap over eval for params and point_pairs
        func = lambda params, bases, points: self._eval(t, params, bases, points)
        return vmap(func, in_axes=(0, 0, 0))(self.params, self.bases, self.point_pairs)

    def _eval(self, t, params, basis, point_pair):
        return self._eval_poly(params, basis, t) + self._eval_line(t, point_pair)

    def _eval_poly(self, params, basis, t):
        coefs = basis @ params
        coefs = coefs.reshape(self.n_poly, 4, self.dim)
        idx = jnp.floor(t * self.n_poly).clip(0, self.n_poly - 1).astype(int)
        coefs_idx = coefs[idx]  # t × n_term × d
        tp = t ** (jnp.arange(4))[:, None]  # n_term × t
        return jnp.einsum("ted,et->td", coefs_idx, tp).T

    def _eval_line(self, t, point_pair):
        p0, p1 = point_pair[0, :], point_pair[1, :]
        a, b = p1 - p0, p0
        return a[:, None] * t + b[:, None]
