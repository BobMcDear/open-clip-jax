"""
Factory for retrieving CLIP models and pre-trained parameters.
"""


import json
import pickle
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax
from flax.linen.dtypes import Dtype
from huggingface_hub import hf_hub_download
from jax import numpy as jnp

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


# Maps CLIP architectures to their available pre-trained parameters.
PRETRAINED = {
    'vit-base-patch32': ('laion400m-e31', 'laion400m-e32', 'laion2b-e16', 'laion2b-s34b-b79k'),
    'vit-base-patch16': ('laion400m-e31', 'laion400m-e32', 'laion2b-s34b-b88k'),
    'vit-large-patch14': ('laion400m-e31', 'laion400m-e32', 'laion2b-s32b-b82k'),
    'vit-huge-patch14': ('laion2b-s32b-b79k',),
    }


def check_model_exists(model_name: str) -> None:
    """
    Raises a ValueError exception if model of name model_name does not exist.

    Args:
        model_name: Name of model that is checked.

    Raises:
        ValueError: Model of name model_name does not exist.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f'Model {model_name} not recognized. Available models are {list_models()}')


def check_model_has_pretrained(model_name: str) -> None:
    """
    Raises a ValueError exception if model of name model_name does not have
    pre-trained parameters.

    Args:
        model_name: Name of model that is checked.

    Raises:
        ValueError: Model of name model_name does not have pre-trained parameters.
    """
    if model_name not in PRETRAINED:
        raise ValueError(f'{model_name} does not have pre-trained parameters.')


def list_models() -> List[str]:
    """
    Lists the names of available models.

    Returns:
        List of available models.
    """
    return list(MODEL_CONFIGS)


def list_pretrained() -> Tuple[Tuple[str, str], ...]:
    """
    Lists the pair of available (model, pre-trained parameters) pairs.

    Returns:
        List of available (model, pre-trained parameters) pairs.
    """
    return tuple(
        (model_name, pretrained) for model_name, tags in PRETRAINED.items()
        for pretrained in tags
        )


def list_pretrained_by_model(model_name: str) -> Tuple[str, ...]:
    """
    Lists the pre-trained parameters of a particular model.

    Args:
        model_name: Name of model whose pre-trained parameters are listed.

    Returns:
        List of the pre-trained parameters for model_name.
    """
    check_model_exists(model_name)
    check_model_has_pretrained(model_name)
    return PRETRAINED[model_name]


def create_model(
    model_name: str,
    softmax_temp: Optional[float] = None,
    dtype: Dtype = jnp.float32,
    ) -> CLIP:
    """
    Creates a CLIP model given its name.

    Args:
        model_name: Name of CLIP model to return. See list_models for available
            options.
        softmax_temp: Temperature coefficient the CLIP model's logits are scaled
            by before calculating softmax and returning, with None for no scaling
            and softmax.
        dtype: The data type of the CLIP model.

    Returns:
        CLIP model.
    """
    check_model_exists(model_name)

    if model_name.startswith('vit'):
        configs = MODEL_CONFIGS[model_name]
        image_model = VisionTransformer(**configs['image_model'], dtype=dtype)
        text_model = TextTransformer(**configs['text_model'], dtype=dtype)
        proj_dim = configs['proj_dim']

    model = CLIP(
        image_model=image_model,
        text_model=text_model,
        proj_dim=proj_dim,
        softmax_temp=softmax_temp,
        dtype=dtype,
        )
    return model


def download_pretrained_params(
    model_name: str,
    pretrained: Union[str, bool] = True,
    ) -> Dict:
    """
    Downloads pre-trained CLIP parameters from Hugging Face Hub.

    Args:
        model_name: Name of CLIP model whose parameters are downloaded.
        pretrained: Name of pre-trained parameters to download, with True for
            the most performant set of parameters. See list_pretrained or
            list_pretrained_by_model for available options.

    Returns:
        Downloaded pre-trained CLIP parameters.

    Raises:
        ValueError: Specified pre-trained parameters not found for the provided
            model.
    """
    available_pretrained =  list_pretrained_by_model(model_name)

    if pretrained is True:
        # The last pre-trained tag is the most performant one.
        pretrained = available_pretrained[-1]

    if pretrained not in available_pretrained:
        raise ValueError(
            f'{model_name} has no pre-trained parameters of name {pretrained}. '
            f'Available options are {available_pretrained}.'
            )

    # The pre-trained parameters have been ported from PyTorch for use by
    # this repository and can be found at https://huggingface.co/bobmcdear.
    name = f'{model_name}-{pretrained}'
    path = hf_hub_download(
        repo_id=f'BobMcDear/open-clip-jax-{name}',
        filename=f'{name}.pkl'
        )
    with open(path, mode='rb') as file:
        pretrained_params = pickle.load(file)

    return pretrained_params


def create_model_with_params(
    model_name: str,
    image_size: int = 224,
    context_len: int = 77,
    softmax_temp: Optional[float] = None,
    pretrained: Union[str, bool] = True,
    dtype: Dtype = jnp.float32,
    ) -> Tuple[CLIP, Dict]:
    """
    Creates a CLIP model and initializes its parameters.

    Args:
        model_name: Name of CLIP model to return. See list_models for available
            options.
        image_size: Image size the image model should expect. This argument has
            no effects on the returned parameters if pretrained is not False.
        context_len: Context length of the text model. This argument has no
            effects on the returned parameters if pretrained is not False.
        softmax_temp: Temperature coefficient the CLIP model's logits are scaled
            by before calculating softmax and returning, with None for no scaling
            and softmax.
        pretrained: If False, the model's parameters are randomly initialized.
            Otherwise, pretrained is interpreted as the name of pre-trained
            parameters to return, with True for the most performant set of
            parameters. See list_pretrained or list_pretrained_by_model for
            available options.
        dtype: The data type of the CLIP model.

    Returns:
        CLIP model and its parameters.
    """
    if pretrained is not False:
        pretrained_params = download_pretrained_params(model_name, pretrained)

    model = create_model(
        model_name=model_name,
        softmax_temp=softmax_temp,
        dtype=dtype,
        )
    vars = model.init(
        rngs=jax.random.PRNGKey(0),
        image_input=jnp.empty((1, image_size, image_size, 3), dtype=dtype),
        text_input=jnp.empty((1, context_len), dtype=jnp.int32),
        )

    if pretrained is not False:
        vars = {**vars, **pretrained_params}

    return model, vars
