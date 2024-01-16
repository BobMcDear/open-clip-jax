"""
Factory for retrieving CLIP models and pre-trained parameters.
"""


import json
import pickle
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import jax
from flax.linen.dtypes import Dtype
from huggingface_hub import hf_hub_download
from jax import numpy as jnp

from .convnext import ConvNeXt
from .model import CLIP
from .transformer import TextTransformer, VisionTransformer


# Model configurations are registered at MODEL_CONFIGS.
MODEL_CONFIGS = {}
def register_configs():
    """
    Registers model configurations under model_configs/ at MODEL_CONFIGS.
    """
    path_configs = Path(__file__).parent/'model_configs/'
    for path in sorted(list(path_configs.glob('*.json'))):
        model_name = path.name.split('.')[0]
        with open(path, mode='r') as file:
            MODEL_CONFIGS[model_name] = json.load(file)
register_configs()


# Maps CLIP architectures to their available pre-trained parameters.
PRETRAINED = {
    'convnext-base': ('laion400m-s13b-b51k',),
    'convnext-base-w': ('laion-aesthetic-s13b-b82k', 'laion-aesthetic-s13b-b82k-320',
                        'laion-aesthetic-s13b-b82k-augreg-320', 'laion2b-s13b-b82k',
                        'laion2b-s13b-b82k-augreg'),
    'convnext-large-d': ('laion2b-s26b-b102k-augreg', 'laion2b-s29b-b131k-ft-320',
                         'laion2b-s29b-b131k-ft-soup-320'),
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


def list_models() -> Tuple[str]:
    """
    Lists the names of available models.

    Returns:
        List of available models.
    """
    return tuple(MODEL_CONFIGS)


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

    Raises:
        ValueError: Model of name model_name does not exist or does not have
            pre-trained parameters.
    """
    check_model_exists(model_name)
    check_model_has_pretrained(model_name)
    return PRETRAINED[model_name]


def create_model(
    model_name: str,
    temp_init: Optional[float] = None,
    grad_checkpoint: bool = False,
    dtype: Dtype = jnp.float32,
    ) -> CLIP:
    """
    Creates a CLIP model given its name.

    Args:
        model_name: Name of CLIP model to return. See list_models for available
            options.
        temp_init: Initial value for a learnable temperature coefficient CLIP's
            projected image vectors are scaled by, with None for no scaling.
        grad_checkpoint: Whether to perform gradient checkpointing on the
            transformer blocks of the image and text towers. If True, intermediate
            activations are not stored and are recomputed during backpropagation.
        dtype: The data type of the CLIP model.

    Returns:
        CLIP model.

    Raises:
        ValueError: Model of name model_name does not exist.
    """
    check_model_exists(model_name)
    configs = MODEL_CONFIGS[model_name]

    if model_name.startswith('vit'):
        image_model = VisionTransformer(
            **configs['image_model'],
            grad_checkpoint=grad_checkpoint,
            dtype=dtype,
            )

    elif model_name.startswith('convnext'):
        conf = configs['image_model']
        depths = tuple(conf.pop('depths'))
        out_dims = tuple(conf.pop('out_dims'))
        image_model = ConvNeXt(
            depths, out_dims,
            **configs['image_model'],
            grad_checkpoint=grad_checkpoint,
            dtype=dtype,
            )

    text_model = TextTransformer(
        **configs['text_model'],
        grad_checkpoint=grad_checkpoint,
        dtype=dtype)

    return CLIP(
        image_model=image_model,
        text_model=text_model,
        proj_dim=configs['proj_dim'],
        temp_init=temp_init,
        dtype=dtype,
        )


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
    temp_init: Optional[float] = None,
    image_size: int = 224,
    context_len: int = 77,
    pretrained: Union[str, bool] = True,
    grad_checkpoint: bool = False,
    dtype: Dtype = jnp.float32,
    ) -> Tuple[CLIP, Dict]:
    """
    Creates a CLIP model and initializes its parameters.

    Args:
        model_name: Name of CLIP model to return. See list_models for available
            options.
        temp_init: Initial value for a learnable temperature coefficient CLIP's
            projected image vectors are scaled by, with None for no scaling.
        image_size: Image size the image model should expect. This argument has
            no effects on the returned parameters if pretrained is not False.
        context_len: Context length of the text model. This argument has no
            effects on the returned parameters if pretrained is not False.
        pretrained: If False, the model's parameters are randomly initialized.
            Otherwise, pretrained is interpreted as the name of pre-trained
            parameters to return, with True for the most performant set of
            parameters. See list_pretrained or list_pretrained_by_model for
            available options.
        grad_checkpoint: Whether to perform gradient checkpointing on the
            transformer blocks of the image and text towers. If True, intermediate
            activations are not stored and are recomputed during backpropagation.
        dtype: The data type of the CLIP model.

    Returns:
        CLIP model and its parameters.
    """
    if pretrained is not False:
        pretrained_params = download_pretrained_params(model_name, pretrained)

    model = create_model(
        model_name,
        temp_init=temp_init,
        grad_checkpoint=grad_checkpoint,
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
