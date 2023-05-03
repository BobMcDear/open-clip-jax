"""
General-purpose deep learning layers.
"""


from typing import Callable, Optional, Union, Tuple

from flax import linen as nn
from flax.linen.dtypes import Array, Dtype
from jax import numpy as jnp


def gelu(input: Array, approximate: bool = False) -> Array:
    """
    Transforms the input using the GELU activation function.

    Args:
        input: Input to transform using GELU.
        approximate: Whether to use a tanh-based approximation of GELU. If
            False, GELU's original formulation is used.

    Returns:
        Input transformed using GELU.
    """
    return nn.gelu(input, approximate=approximate)


class MLP(nn.Module):
    """
    Multilayer perceptron with a single hidden layer.

    Attributes:
        out_dim: Number of output features. If None, it is set to the number of
            input features.
        expansion_factor: Expansion factor for the hidden layer.
        act: Activation function applied in the hidden layer.
        bias: Whether the linear layers should contain bias terms.
        dtype: The data type of the computations.
    """
    out_dim: Optional[int] = None
    expansion_factor: float = 4.
    act: Callable = gelu
    bias: bool = True
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input: Array) -> Array:
        in_dim = input.shape[-1]
        hidden_dim = int(self.expansion_factor*in_dim)
        out_dim = self.out_dim or in_dim

        output = nn.Dense(
            features=hidden_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            )(input)
        output = self.act(output)
        output = nn.Dense(
            features=out_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            )(output)

        return output


class MultiHeadAttention(nn.MultiHeadDotProductAttention):
    """
    Drop-in replacement for Flax's attention module that sets the key-value
    input to the query input when the former is None.

    See base class.
    """
    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
        ) -> Array:
        inputs_kv = inputs_kv or inputs_q
        return super().__call__(inputs_q, inputs_kv, mask, deterministic)


def global_avg_pool(
    input: Array,
    axis: Union[int, Tuple[int, ...]] = 1,
    ) -> Array:
    """
    Global average pooling over arbitrary axis.

    Args:
        input: Input to average pool.
        axis: Axis over which global average pooling is performed.

    Returns:
        Globally average-pooled input, with the same data type
        as the input.
    """
    # GAP is calculated in float32.
    output = jnp.asarray(input, dtype=jnp.float32)
    output = jnp.mean(output, axis=axis)
    output = jnp.asarray(output, dtype=input.dtype)
    return output
