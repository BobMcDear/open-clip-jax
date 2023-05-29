"""
Utilities for conducting inference with CLIP models.
"""


import warnings
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import jax
import tensorflow as tf
from flax.linen.dtypes import Array, Dtype
from jax import numpy as jnp

from . import image_transforms
from ..training.train import tf_to_np
from .factory import create_model_with_params
from .tokenizer import tokenize


Image = Any


def preprocess_image(
    image: Union[Image, List[Image]],
    dtype: tf.DType = tf.float32,
    ) -> Array:
    """
    Pre-processes an image or list of images for inference with CLIP models.

    Args:
        image: Image or list of images to pre-process. The image type must be
            consumable by TensorFlow, e.g., PIL image or NumPy array.
        dtype: The data type the image is converted to.

    Returns:
        Image or list of images pre-processed, with an additional batch axis.
    """
    if not isinstance(image, list):
        image = [image]

    item_transforms = image_transforms.Sequential(
        image_transforms.resize_smallest_edge,
        image_transforms.center_crop_with_padding,
        )
    preprocessed = list(map(item_transforms, image))

    # Some transforms are applied over batches for greater efficiency.
    preprocessed = tf.stack(preprocessed, axis=0)
    preprocessed = image_transforms.Sequential(
            image_transforms.normalize,
            partial(tf.image.convert_image_dtype, dtype=dtype),
            partial(tf_to_np, device_axis=False),
            )(preprocessed)

    return preprocessed


class CLIPInference:
    """
    Convenience class for end-to-end CLIP inference with raw image/text inputs.

    Attributes:
        model: The CLIP model.
        apply_fn: The CLIP model's JITted apply function, with variables fixed.
        dtype: The data type of the CLIP model.
    """
    def __init__(
        self,
        model_name: str,
        softmax_temp: Optional[float] = 100.,
        pretrained: Union[str, bool] = True,
        dtype: Dtype = jnp.float32,
        ) -> None:
        """
        Creates the CLIP model and initializes its parameters.

        Args:
            model_name: Name of CLIP model to return. See list_models for available
                options.
            softmax_temp: Temperature coefficient the CLIP model's logits are scaled
                by before calculating softmax and returning, with None for no scaling
                and softmax.
            pretrained: If False, the model's parameters are randomly initialized.
                Otherwise, pretrained is interpreted as the name of pre-trained
                parameters to return, with True for the most performant set of
                parameters. See list_pretrained or list_pretrained_by_model for
                available options.
            dtype: The data type of the CLIP model.
        """
        if pretrained is False:
            warnings.warn('You are performing inference with randomly-initialized weights.')

        self.model, vars = create_model_with_params(
            model_name,
            softmax_temp=softmax_temp,
            pretrained=pretrained,
            dtype=dtype,
            )
        self.apply_fn = partial(jax.jit(self.model.apply), vars)
        self.dtype = dtype

    def __repr__(self) -> str:
        return str(self.model)

    def __call__(
        self,
        image: Union[Image, List[Image]],
        text: Union[str, List[str]],
        ) -> Tuple[Array, Array]:
        """
        Computes the CLIP similarity between an input image or a list of images
        and an input text or a list of texts.

        Args:
            image: Input image or list of images. The image type must be
                consumable by TensorFlow, e.g., PIL image or NumPy array.
            text: Input text or list of texts.

        Returns:
            The CLIP similarity between the input image(s) and text(s), in two
            formats: A per-image view where entry i, j corresponds to the
            similarity between the ith image and the jth text, and its transpose.
        """
        image_input = preprocess_image(image, self.dtype)
        text_input = tokenize(text)._numpy()
        return self.apply_fn(image_input, text_input)
