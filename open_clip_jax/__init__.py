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
    create_model,
    create_model_with_params,
    image_transforms,
    list_models,
    list_pretrained,
    list_pretrained_by_model,
    preprocess_image,
    tokenize,
    )
