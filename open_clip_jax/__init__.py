"""
Implementation of CLIP and its image/text models in Flax, plus codebase for
training CLIP models.
"""


# Hide GPUs from TensorFlow to ensure it doesn't allocate GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


from .clip import (
    CLIP,
    CLIPWithLoss,
    create_model,
    create_model_with_params,
    list_models,
    list_pretrained,
    list_pretrained_by_model,
    tokenize,
    )
from .training import (
    IMAGENET_DATASET_MEAN,
    IMAGENET_DATASET_STD,
    INCEPTION_DATASET_MEAN,
    INCEPTION_DATASET_STD,
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    create_csv_dataset,
    create_learning_rate_scheduler,
    create_weight_decay_mask,
    image_transforms,
    train_and_validate,
    )
