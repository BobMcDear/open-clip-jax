"""
Utilities for conducting inference with CLIP models.
"""


import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import tensorflow as tf
from flax import linen as nn
from flax.linen.dtypes import Array, Dtype
from jax import numpy as jnp

from .image_transforms import create_image_transforms
from .factory import create_model_with_params
from .model import CLIP
from .tokenizer import tokenize


Image = Any


@partial(jax.jit, static_argnums=(0, 4))
def calculate_similarity(
    model: CLIP,
    vars: Dict,
    image_input: Array,
    text_input: Array,
    softmax_temp: Optional[float] = 100.,
    ) -> Tuple[Array, Array]:
    """
    Calculates CLIP similarities of images and texts.

    Args:
        model: CLIP model returning projected image and text vectors.
        vars: The CLIP model's variables.
        image_input: Input to the image model.
        text_input: Input to the text model.
        softmax_temp: Temperature coefficient the logits are scaled by before
            calculating softmax and returning, with None for no scaling and
            softmax.

    Returns:
        The CLIP similarities between the input image(s) and text(s), in two
        formats: A per-image view where entry i, j corresponds to the
        similarity between the ith image and the jth text, and a per-text
        view where entry i, j corresponds to the similarity between the ith
        text and jth image.
    """
    image_proj, text_proj = model.apply(vars, image_input, text_input)
    logits_per_image = image_proj @ text_proj.T
    logits_per_text = logits_per_image.T

    if softmax_temp:
        scaled = softmax_temp * logits_per_image
        logits_per_image = nn.softmax(scaled)
        logits_per_text = nn.softmax(scaled.T)

    return logits_per_image, logits_per_text


class CLIPInference:
    """
    Convenience class for end-to-end CLIP inference with raw image/text inputs.

    Attributes:
        model: The CLIP model.
        image_transforms: Transforms applied on input images before being fed
            to the CLIP model.
        calculate_similarity: Function returning CLIP similarities given
            pre-processed input images and tokenized texts.
    """
    def __init__(
        self,
        model_name: str,
        softmax_temp: Optional[float] = 100.,
        pretrained: Union[str, bool] = True,
        dtype: Dtype = jnp.float32,
        ) -> None:
        """
        Creates the CLIP model and related modules.

        Args:
            model_name: Name of CLIP model. See list_models for available
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
            temp_init=None,
            pretrained=pretrained,
            dtype=dtype,
            )
        self.image_transforms = create_image_transforms(
            train=False,
            input_format='image',
            do_batch_transforms=False,
            dtype=dtype,
            )
        self.calculate_similarity = partial(
            calculate_similarity,
            model=self.model,
            vars=vars,
            softmax_temp=softmax_temp,
            )

    def __repr__(self) -> str:
        return str(self.model)

    def __call__(
        self,
        image: Union[Image, List[Image]],
        text: Union[str, List[str]],
        ) -> Tuple[Array, Array]:
        """
        Computes CLIP similarities between an input image or a list of images
        and an input text or a list of texts.

        Args:
            image: Input image or list of images. The image type must be
                consumable by TensorFlow, e.g., PIL image or NumPy array.
            text: Input text or list of texts.

        Returns:
            The CLIP similarity between the input image(s) and text(s), in two
            formats: A per-image view where entry i, j corresponds to the
            similarity between the ith image and the jth text, and a per-text
            view where entry i, j corresponds to the similarity between the ith
            text and jth image.
        """
        if not isinstance(image, list):
            image = [image]

        image_input = tf.stack(list(map(self.image_transforms, image)))._numpy()
        text_input = tokenize(text)._numpy()

        return self.calculate_similarity(
            image_input=image_input,
            text_input=text_input,
            )
