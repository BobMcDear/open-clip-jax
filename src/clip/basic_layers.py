"""
General-purpose deep learning layers.
"""


from typing import Any, Callable, Optional, Union, Tuple

from flax import linen as nn
from jax import numpy as jnp
from jax._src.numpy.lax_numpy import _ScalarMeta


class MLP(nn.Module):
    """
    Multilayer perceptron with a single hidden layer.

    Attributes:
        out_dim: Number of output features. If None, it is set to the number of
            input features.
        expansion_factor: Expansion factor for the hidden layer.
        act: Activation function applied in the hidden layer.
        dtype: The data type of the computations.
    """
    out_dim: Optional[int] = None
    expansion_factor: float = 4.
    act: Callable = nn.gelu
    bias: bool = True
    dtype: _ScalarMeta = jnp.float32

    @nn.compact
    def __call__(self, input: Any) -> Any:
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
        inputs_q: Any,
        inputs_kv: Optional[Any] = None,
        mask: Optional[Any] = None,
        deterministic: Optional[bool] = None,
        ) -> Any:
        inputs_kv = inputs_kv or inputs_q
        return super().__call__(inputs_q, inputs_kv, mask, deterministic)


def global_avg_pool(
    input: Any,
    axis: Union[int,Tuple[int, ...]] = 1,
    ) -> Any:
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
