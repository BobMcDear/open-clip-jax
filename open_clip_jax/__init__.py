"""
Implementation of CLIP and its image/text models in Flax, plus codebase for
training CLIP models.
"""
from .clip import CLIP, CLIPWithLoss, create_model, list_models, tokenize
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
