import jax.numpy as jnp
import equinox as eqx


class ProjectionSplit(eqx.Module):
    input_dim: int
    output_dim: int
    input_dim_total: int
    output_dim_total: int
    mode_in: str
    mode_out: str

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_total = input_dim if isinstance(input_dim, int) else input_dim[0] * input_dim[1] * input_dim[2]
        self.output_dim_total = (
            output_dim if isinstance(output_dim, int) else output_dim[0] * output_dim[1] * output_dim[2]
        )
        self.mode_in = "vector" if isinstance(input_dim, int) else "image"
        self.mode_out = "vector"

        assert self.input_dim_total >= self.output_dim_total, "Input dimension has to be larger than output dimension"

    def transform(self, inputs, condition=None):
        if self.mode_in == "vector" and self.mode_out == "vector":
            u = inputs[: self.output_dim]
            rest = inputs[self.output_dim :]
        elif self.mode_in == "image" and self.mode_out == "vector":
            (
                c,
                h,
                w,
            ) = self.input_dim
            inputs = inputs.ravel()
            u = inputs[: self.output_dim]
            rest = inputs[self.output_dim :]
        else:
            raise NotImplementedError("Unsupported projection modes {}, {}".format(self.mode_in, self.mode_out))
        return u, rest

    def inverse(self, inputs, condition=None):
        orthogonal_inputs = jnp.zeros(self.input_dim_total - self.output_dim_total)
        if self.mode_in == "vector" and self.mode_out == "vector":
            x = jnp.concatenate((inputs, orthogonal_inputs), axis=0)
        elif self.mode_in == "image" and self.mode_out == "vector":
            c, h, w = self.input_dim
            x = jnp.concatenate((inputs, orthogonal_inputs), axis=0)
            x = x.reshape(c, h, w)
        else:
            raise NotImplementedError("Unsupported projection modes {}, {}".format(self.mode_in, self.mode_out))
        return x


class Projection(eqx.Module):
    input_dim: int
    output_dim: int
    input_dim_total: int
    output_dim_total: int
    mode_in: str
    mode_out: str

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_total = input_dim if isinstance(input_dim, int) else input_dim[0] * input_dim[1] * input_dim[2]
        self.output_dim_total = (
            output_dim if isinstance(output_dim, int) else output_dim[0] * output_dim[1] ** output_dim[2]
        )
        self.mode_in = "vector" if isinstance(input_dim, int) else "image"
        self.mode_out = "vector"

        assert self.input_dim_total >= self.output_dim_total, "Input dimension has to be larger than output dimension"

    def transform(self, inputs, condition=None):
        if self.mode_in == "vector" and self.mode_out == "vector":
            u = inputs[: self.output_dim]
        elif self.mode_in == "image" and self.mode_out == "vector":
            (
                c,
                h,
                w,
            ) = self.input_dim
            inputs = inputs.ravel()
            u = inputs[: self.output_dim]
        else:
            raise NotImplementedError("Unsupported projection modes {}, {}".format(self.mode_in, self.mode_out))
        return u

    def inverse(self, inputs, condition=None):
        if self.mode_in == "vector" and self.mode_out == "vector":
            x = jnp.concatenate((inputs, jnp.zeros(self.input_dim_total - self.output_dim_total)), axis=0)
        elif self.mode_in == "image" and self.mode_out == "vector":
            c, h, w = self.input_dim
            x = jnp.concatenate((inputs, jnp.zeros(self.input_dim_total - self.output_dim_total)), axis=0)
            x = x.reshape(c, h, w)
        else:
            raise NotImplementedError("Unsupported projection modes {}, {}".format(self.mode_in, self.mode_out))
        return x
