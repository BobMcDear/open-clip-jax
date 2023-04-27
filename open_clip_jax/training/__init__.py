"""
Codebase for training CLIP models.
"""

from . import image_transforms
from .constants import (
    IMAGENET_DATASET_MEAN,
    IMAGENET_DATASET_STD,
    INCEPTION_DATASET_MEAN,
    INCEPTION_DATASET_STD,
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    )
from .data import create_csv_dataset
from .optimizer import create_weight_decay_mask
from .scheduler import create_learning_rate_scheduler
from .train import train_and_validate
