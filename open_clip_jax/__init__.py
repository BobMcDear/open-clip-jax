"""
Implementation of CLIP and its image/text models in Flax, plus codebase for
training CLIP models.
"""


# Hide GPUs from TensorFlow to ensure it does not allocate GPU memory.
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


from .clip import (
    CLIP,
    CLIPInference,
    create_image_transforms,
    create_model,
    create_model_with_params,
    list_models,
    list_pretrained,
    list_pretrained_by_model,
    tokenize,
    )
