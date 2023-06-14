"""
CLIP model.
"""


from typing import Optional

from flax import linen as nn
from flax.linen.dtypes import Array, Dtype
from jax import numpy as jnp


def l2_norm(input: Array) -> Array:
    """
    L2-normalizes the input.

    Args:
        input: Input to normalize.

    Returns:
        Normalized input, with the same data type as the input.
    """
    # Norms are calculated in float32.
    output = jnp.asarray(input, dtype=jnp.float32)
    output = output / jnp.linalg.norm(output, axis=1, keepdims=True)
    output = jnp.asarray(output, dtype=input.dtype)
    return output


class CLIP(nn.Module):
    """
    CLIP model that extracts feature vectors from image and text data and
    projects them to a multimodal embedding space.

    Attributes:
        image_model: Model used to extract feature vectors from image data.
        text_model: Model used to extract features vectors from text data.
        proj_dim: Dimension to which the image and text features are projected.
        proj_bias: Whether the linear projection layers should contain bias
            terms.
        norm: Whether to L2-normalize the projected vectors.
        temp_init: Initial value for a learnable temperature coefficient the
            projected image vectors are scaled by, with None for no scaling.
        dtype: The data type of the computations.
    """
    image_model: nn.Module
    text_model: nn.Module
    proj_dim: int = 512
    proj_bias: bool = False
    norm: bool = True
    temp_init: Optional[float] = 2.6593
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, image_input: Array, text_input: Array) -> Array:
        image_output = self.image_model(image_input)
        text_output = self.text_model(text_input)

        image_proj = nn.Dense(
            features=self.proj_dim,
            use_bias=self.proj_bias,
            dtype=self.dtype,
            )(image_output)
        text_proj = nn.Dense(
            features=self.proj_dim,
            use_bias=self.proj_bias,
            dtype=self.dtype,
            )(text_output)

        if self.norm:
            image_proj = l2_norm(image_proj)
            text_proj = l2_norm(text_proj)

        if self.temp_init:
            temp = self.param(
                name='temp',
                init_fn=lambda _: jnp.array(self.temp_init, dtype=image_proj.dtype),
                )
            image_proj = jnp.exp(temp) * image_proj

        return image_proj, text_proj
