"""
Optimizer factory.
"""


from typing import Dict

import jax
from flax.core import FrozenDict


def create_weight_decay_mask(vars: FrozenDict) -> Dict:
    """
    Creates a mask so normalization parameters and biases are not decayed.

    Args:
        vars: Model variables to create a weight decay mask for.

    Returns:
        Mask with True for parameters to decay and False otherwise.
    """
    # Normalization parameters and biases are the only parameter groups having
    # fewer than 2 dimensions.
    return jax.tree_util.tree_map(lambda p: p.ndim > 1, vars)
