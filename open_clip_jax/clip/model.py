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
    CLIP model that calculates similarity of image and text inputs.

    Attributes:
        image_model: Model used to extract feature vectors from image data.
        text_model: Model used to extract features vectors from text data.
        proj_dim: Dimension to which the image and text features are projected.
        proj_bias: Whether the linear projection layers should contain bias
            terms.
        norm: Whether to L2-normalize the projected feature vectors prior to
            calculating their dot product.
        softmax_temp: Temperature coefficient the logits are scaled by before
            calculating softmax and returning, with None for no scaling and
            softmax.
        dtype: The data type of the computations.
    """
    image_model: nn.Module
    text_model: nn.Module
    proj_dim: int = 512
    proj_bias: bool = False
    norm: bool = True
    softmax_temp: Optional[float] = None
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, image_input: Array, text_input: Array) -> Array:
        image_output = self.image_model(image_input)
        text_output = self.text_model(text_input)

        image_projection = nn.Dense(
            features=self.proj_dim,
            use_bias=self.proj_bias,
            dtype=self.dtype,
            )(image_output)
        text_projection = nn.Dense(
            features=self.proj_dim,
            use_bias=self.proj_bias,
            dtype=self.dtype,
            )(text_output)

        if self.norm:
            image_projection = l2_norm(image_projection)
            text_projection = l2_norm(text_projection)

        logits_per_image = image_projection @ text_projection.T
        logits_per_text = logits_per_image.T

        if self.softmax_temp:
            scaled = self.softmax_temp*logits_per_image
            logits_per_image = nn.softmax(scaled)
            logits_per_text = nn.softmax(scaled.T)

        return logits_per_image, logits_per_text
