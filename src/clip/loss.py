"""
CLIP (contrastive) loss.
"""


from typing import Any, Optional

from flax import linen as nn
from jax import numpy as jnp
from optax import softmax_cross_entropy_with_integer_labels

from .model import CLIP


class CLIPLoss(nn.Module):
    """
    CLIP (contrastive) loss. Does not support variable batch sizes.

    Attributes:
        temp_init: Initial value for a learnable temperature coefficient the logits
            are scaled by, with None for no scaling.
    """
    temp_init: Optional[float] = 1.155

    @nn.compact
    def __call__(self, logits_per_image: Any, logits_per_text: Any) -> Any:
        if self.temp_init:
            temp =  self.param(
                name='temp',
                init_fn=lambda _: jnp.array(self.temp_init),
                )
            logits_per_image = jnp.exp(temp)*logits_per_image
            logits_per_text = jnp.exp(temp)*logits_per_text

        labels = self.variable(
            col='labels',
            name='labels',
            init_fn=lambda: jnp.arange(0, len(logits_per_image)),
            ).value
        return jnp.mean(
            softmax_cross_entropy_with_integer_labels(logits_per_image, labels) +
            softmax_cross_entropy_with_integer_labels(logits_per_text, labels)
            ) / 2


class CLIPWithLoss(nn.Module):
    """
    CLIP model and loss refactored into a class returning loss given inputs.
    Does not support variable batch sizes.

    Attributes:
        model: CLIP model.
        temp_init: Initial value for a learnable temperature coefficient the logits
            are scaled by when calculating the loss, with None for no scaling.
    """
    model: CLIP
    temp_init: Optional[float] = 2.6593

    @nn.compact
    def __call__(self, image_input: Any, text_input: Any) -> Any:
        logits_per_image, logits_per_text = self.model(image_input, text_input)
        loss_fn = CLIPLoss(self.temp_init)
        return loss_fn(logits_per_image, logits_per_text)
