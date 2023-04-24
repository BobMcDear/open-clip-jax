"""
Learning rate schedulers.
"""


from typing import Optional

import optax
from optax._src.base import Schedule


def warmup_const_schedule(
    learning_rate: float = 5e-4,
    n_warmup_steps: int = 10000,
    warmup_init: float = 1e-5,
    ) -> Schedule:
    """
    Creates a constant learning rate scheduler with warmup.

    Args:
        learning_rate: Learning rate that is kept constant during the schedule.
        n_warmup_steps: Number of warmup steps.
        warmup_init: Initial learning rate when beginning warmup.

    Returns:
        Constant learning rate scheduler with a learning rate of learning_rate
        and n_warmup_steps warmup steps.
    """
    warmup_fn = optax.linear_schedule(
        init_value=warmup_init,
        end_value=learning_rate,
        transition_steps=n_warmup_steps,
        )
    const_fn = optax.constant_schedule(learning_rate)
    scheduler = optax.join_schedules([warmup_fn, const_fn], boundaries=[n_warmup_steps])
    return scheduler


def create_learning_rate_scheduler(
    scheduler_name: str = 'cosine',
    learning_rate: float = 5e-4,
    n_train_iters: Optional[int] = None,
    n_warmup_steps: int = 10000,
    warmup_init: float = 1e-5,
    ) -> Schedule:
    """
    Creates the desired learning rate scheduler.

    Args:
        scheduler_name: Name of learning rate scheduler to use, with 'cosine'
            for cosine decay and 'const' for a constant scheduler.
        learning_rate: Peak learning rate for the cosine decay scheduler and
            the constant learning rate for the constant scheduler.
        n_train_iters: Number of iterations the model will train for. This
            argument is required for cosine decay and ignored for a constant
            scheduler.
        n_warmup_steps: Number of warmup steps.
        warmup_init: Initial learning rate when beginning warmup.

    Returns:
        Desired learning rate scheduler.

    Raises:
        ValueError: The name of the learning rate scheduler is invalid,
            n_train_iters is not provided for cosine decay or is smaller than
            the number of warmup steps.
    """
    if scheduler_name == 'cosine':
        if n_train_iters is None:
            raise ValueError('n_train_iters must be provided for a cosine decay scheduler.')
        if n_train_iters < n_warmup_steps:
            raise ValueError('Number of iterations cannot be smaller than the number of warmup steps.')

        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=warmup_init,
            peak_value=learning_rate,
            warmup_steps=n_warmup_steps,
            decay_steps=n_train_iters,
            )

    elif scheduler_name == 'const':
        scheduler = warmup_const_schedule(
            learning_rate=learning_rate,
            n_warmup_steps=n_warmup_steps,
            warmup_init=warmup_init,
            )

    else:
        raise ValueError(f'Learning rate scheduler {scheduler_name} not recognized.')

    return scheduler
