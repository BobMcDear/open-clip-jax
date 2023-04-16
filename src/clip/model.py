"""
CLIP model.
"""


from typing import Any

from flax import linen as nn
from jax import numpy as jnp
from jax._src.numpy.lax_numpy import _ScalarMeta


def l2_norm(input: Any) -> Any:
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
    CLIP model that calculates similarity logits of image and text features.

    Attributes:
        image_model: Model used to extract feature vectors from image data.
        text_model: Model used to extract features vectors from text data.
        proj_dim: Dimension to which the image and text features are projected.
        norm: Whether to take the norm of the feature vectors prior to
            calculating their dot product.
        dtype: The data type of the computations.
    """
    image_model: nn.Module
    text_model: nn.Module
    proj_dim: int = 512
    norm: bool = True
    dtype: _ScalarMeta = jnp.float32

    @nn.compact
    def __call__(self, image_input: Any, text_input: Any) -> Any:
        image_output = self.image_model(image_input)
        text_output = self.text_model(text_input)

        image_projection = nn.Dense(
            features=self.proj_dim,
            dtype=self.dtype,
            )(image_output)
        text_projection = nn.Dense(
            features=self.proj_dim,
            dtype=self.dtype,
            )(text_output)

        if self.norm:
            image_projection = l2_norm(image_projection)
            text_projection = l2_norm(text_projection)

        logits = image_projection @ text_projection.T
        return logits, logits.T
