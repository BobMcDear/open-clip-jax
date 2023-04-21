"""
Factory for retrieving CLIP models.
"""


import json
from typing import List
from pathlib import Path

from jax import numpy as jnp
from jax._src.numpy.lax_numpy import _ScalarMeta

from .model import CLIP
from .transformer import TextTransformer, VisionTransformer


# Model configurations are registered at MODEL_CONFIGS.
MODEL_CONFIGS = {}
def register_configs():
    """
    Registers model configurations under model_configs/ at MODEL_CONFIGS.
    """
    path_configs = Path(__file__).parent/'model_configs/'
    for path in path_configs.glob('*.json'):
        model_name = path.name.split('.')[0]
        with open(path, mode='r') as file:
            MODEL_CONFIGS[model_name] = json.load(file)
register_configs()


def list_models() -> List[str]:
    """
    Lists the names of available models.

    Returns:
        List of available models.
    """
    return list(MODEL_CONFIGS)


def create_model(
    model_name: str,
    dtype: _ScalarMeta = jnp.float32,
    ) -> CLIP:
    """
    Creates a CLIP model given its name.

    Args:
        model_name: Name of CLIP model to return. See src/clip/model_configs/
            or list_models for available options.
        dtype: The data type of the CLIP model.

    Returns:
        CLIP model.

    Raises:
        ValueError: Model of name model_name was not found.
    """
    if model_name.startswith('vit'):
        configs = MODEL_CONFIGS[model_name]
        image_model = VisionTransformer(**configs['image_model'], dtype=dtype)
        text_model = TextTransformer(**configs['text_model'], dtype=dtype)
        proj_dim = configs['proj_dim']

    else:
        raise ValueError(f'Model {model_name} not recognized. Available models are {list_models()}')

    model = CLIP(
        image_model=image_model,
        text_model=text_model,
        proj_dim=proj_dim,
        dtype=dtype,
        )
    return model
