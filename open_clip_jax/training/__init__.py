"""
Codebase for training CLIP models.
"""


from .data import create_dataset
from .optimizer import create_weight_decay_mask
from .scheduler import create_learning_rate_scheduler
from .train import TrainState, train_and_validate
