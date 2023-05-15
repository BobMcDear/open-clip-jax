"""
Implementation of CLIP and its image/text models in Flax, plus codebase for
training CLIP models.
"""


# Hide GPUs from TensorFlow to ensure it doesn't allocate GPU memory.
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


from .clip import (
    CLIP,
    CLIPInference,
    CLIPWithLoss,
    IMAGENET_DATASET_MEAN,
    IMAGENET_DATASET_STD,
    INCEPTION_DATASET_MEAN,
    INCEPTION_DATASET_STD,
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    create_model,
    create_model_with_params,
    image_transforms,
    list_models,
    list_pretrained,
    list_pretrained_by_model,
    preprocess_image,
    tokenize,
    )
from .training import (
    create_csv_dataset,
    create_learning_rate_scheduler,
    create_weight_decay_mask,
    train_and_validate,
    )
