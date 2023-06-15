"""
Core code for training CLIP models.
"""


import logging
import time
from typing import Any, Iterable, Optional, Tuple
from functools import partial

import jax
import tensorflow as tf
from flax import jax_utils
from flax.core import frozen_dict
from flax.linen.dtypes import Array
from flax.training import checkpoints, train_state
from flax.training.dynamic_scale import DynamicScale
from jax import lax
from jax import numpy as jnp

from ..clip.loss import clip_loss, generate_labels
from ..clip.image_transforms import tf_to_np


PyTree = Any


class AvgMeter:
    """
    Average meter that keeps track of a series of values and their average.

    Attributes:
        val: The last registered value.
        sum: Sum of all registered values.
        count: Total count of entries registered.
        avg: Average of registered values.
    """
    def __init__(self) -> None:
        """
        Sets up the tracker.
        """
        self.reset()

    def __repr__(self) -> str:
        return (
            f'Last value: {self.val}\n'
            f'Total sum: {self.sum}\n'
            f'Total count: {self.count}\n'
            f'Average: {self.avg}'
            )

    def reset(self) -> None:
        """
        Resets the meter.
        """
        self.val = 0.
        self.sum = 0.
        self.count = 0
        self.avg = 0.

    def update(self, val: Array, count: Optional[int] = None) -> None:
        """
        Updates the tracker.

        Args:
            val: Value to add to the tracker.
            count: Count of entries val accounts for. If None, it is set to 1.
        """
        count = count or 1
        self.val = val
        self.sum += count * val
        self.count += count
        self.avg = self.sum / self.count


class TrainState(train_state.TrainState):
    """
    Flax training state for CLIP models with support for dynamic loss scaling.

    See base class.
    """
    dynamic_scale: Optional[DynamicScale] = None


def save_checkpoint(
    checkpoint_dir: str,
    state: TrainState,
    epoch: int,
    ) -> None:
    """
    Saves a training state checkpoint.

    Args:
        checkpoint_dir: Directory in which checkpoints are saved.
        state: Training state to checkpoint.
        epoch: Current epoch.
    """
    # The leaves have been replicated over all devices,
    # but only one copy is necessary.
    state = jax.tree_util.tree_map(lambda leaf: leaf[0], state)
    checkpoints.save_checkpoint_multiprocess(
        ckpt_dir=checkpoint_dir,
        target=jax.device_get(state),
        step=epoch,
        prefix='checkpoint_epoch_',
        keep=5,
        )


@partial(jax.pmap, axis_name='devices')
def train_iter(
    state: TrainState,
    image_input: Array,
    text_input: Array,
    labels: Array,
    ) -> Tuple[TrainState, Array]:
    """
    Performs one training iteration.

    Args:
        state: Training state whose apply function returns projected image and
            text vectors.
        image_input: Input to the image model.
        text_input: Input to the text model.
        labels: Labels used to calculate the cross-entropy loss of the similarity
            logits.

    Returns:
        Updated training state and loss.
    """
    def loss_fn(vars):
        image_proj, text_proj = state.apply_fn(vars, image_input, text_input)
        return clip_loss(
            image_proj,
            text_proj,
            labels=labels,
            device_axis_name='devices',
            )

    if state.dynamic_scale:
        loss_and_grad_fn = state.dynamic_scale.value_and_grad(
            fun=loss_fn,
            axis_name='devices',
            )
        dynamic_scale, is_finite, loss, grads = loss_and_grad_fn(state.params)
        # Dynamic scale averages gradients across devices automatically.

    else:
        loss_and_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(state.params)
        grads = lax.pmean(grads, axis_name='devices')

    # After update, in case of dynamic scaling for float16,
    # parameters with NaN/infinite gradients are restored.
    updated_state = state.apply_gradients(grads=grads)
    if state.dynamic_scale:
        params_without_nans = jax.tree_util.tree_map(
            partial(jnp.where, is_finite),
            updated_state.params,
            state.params,
            )
        opt_state_without_nans = jax.tree_util.tree_map(
            partial(jnp.where, is_finite),
            updated_state.opt_state,
            state.opt_state,
            )
        updated_state = updated_state.replace(
            params=params_without_nans,
            opt_state=opt_state_without_nans,
            dynamic_scale=dynamic_scale,
            )

    return updated_state, lax.pmean(loss, axis_name='devices')


@partial(jax.pmap, axis_name='devices')
def valid_iter(
    state: TrainState,
    image_input: Array,
    text_input: Array,
    labels: Array,
    ) -> Array:
    """
    Performs one validation iteration.

    Args:
        state: Training state whose apply function returns projected image and
            text vectors.
        image_input: Input to the image model.
        text_input: Input to the text model.
        labels: Labels used to calculate the cross-entropy loss of the similarity
            logits.

    Returns:
        Loss.
    """
    image_proj, text_proj = state.apply_fn(state.params, image_input, text_input)
    loss = clip_loss(
        image_proj,
        text_proj,
        labels=labels,
        device_axis_name='devices',
        )
    return lax.pmean(loss, axis_name='devices')


def tf_dataset_to_np_iter(
    dataset: tf.data.Dataset,
    device_axis: bool = True,
    ) -> Iterable:
    """
    Converts a TensorFlow dataset into an iterator yielding NumPy ararys.

    Args:
        dataset: TensorFlow dataset to convert into an iterator.
        device_axis: Whether to add a leading device axis to the data for
            distributed training.

    Returns:
        Iterator yielding NumPy arrays from dataset, potentially with an
        additional device axis.
    """
    # Data is prefetched to device to speed up training.
    dataset_iter = map(partial(tf_to_np, device_axis=device_axis), dataset)
    return jax_utils.prefetch_to_device(dataset_iter, size=2)


def train_and_validate(
    state: TrainState,
    train_dataset: tf.data.Dataset,
    valid_dataset: tf.data.Dataset,
    n_epochs: int = 32,
    log_freq: int = 100,
    checkpoint_dir: Optional[str] = None,
    checkpoint_freq: int = 5,
    resume_from_checkpoint: Optional[str] = None,
    ) -> None:
    """
    Trains a CLIP model and validates after each epoch.

    Args:
        state: Training state whose apply function returns projected image and
            text vectors.
        train_dataset: Dataset returning (image, text) pairs for training and
            an n_iters_per_epoch attribute denoting the number of iterations
            per epoch.
        valid_dataset: Dataset returning (image, text) pairs for validation and
            an n_iters_per_epoch attribute denoting the number of iterations
            per epoch.
        n_epochs: Number of epochs to train for.
        log_freq: Training and validation information are logged to console
            every log_freq iterations.
        checkpoint_dir: Directory to save checkpoints in. If None, they are
            saved locally in folder 'checkpoint-date/'.
        checkpoint_freq: Checkpoints are saved every checkpoint_freq epochs.
        resume_from_checkpoint: If not None, the checkpoint at path
            resume_from_checkpoint is loaded and training resumed.
    """
    begin_epoch = 1
    if resume_from_checkpoint:
        # Checkpoints end in an '_epoch' suffix denoting the checkpoint epoch.
        begin_epoch = int(resume_from_checkpoint.split('_')[-1]) + 1
        state = checkpoints.restore_checkpoint(
            ckpt_dir=resume_from_checkpoint,
            target=state,
            )

    state = jax_utils.replicate(state)
    train_dataset_iter = tf_dataset_to_np_iter(train_dataset)
    valid_dataset_iter = tf_dataset_to_np_iter(valid_dataset)
    labels = generate_labels(train_dataset.element_spec[0].shape[0])

    loss_meter = AvgMeter()
    checkpoint_dir = checkpoint_dir or time.strftime('checkpoint-%Y-%m-%d-%H-%M', time.gmtime())

    for epoch in range(begin_epoch, n_epochs + 1):
        logging.info(f'Beginning epoch {epoch}...')

        # Train
        loss_meter.reset()
        for iter_ind in range(1, train_dataset.n_iters_per_epoch + 1):
            image_input, text_input = next(train_dataset_iter)
            state, loss = train_iter(state, image_input, text_input, labels)
            loss_meter.update(loss[0])

            # Temperature coefficient is clipped to [0, ln(100)].
            vars = state.params.unfreeze()
            vars['params']['temp'] = jnp.clip(
                a=vars['params']['temp'],
                a_min=0.0,
                a_max=4.6052,
                )
            state = state.replace(params=frozen_dict.freeze(vars))

            if iter_ind % log_freq == 0 or iter_ind == train_dataset.n_iters_per_epoch:
                message = (
                    f'Training epoch: {epoch}/{n_epochs} '
                    f'Iteration: {iter_ind}/{train_dataset.n_iters_per_epoch} '
                    f'Loss (current iteration): {loss_meter.val} '
                    f'Loss (average of current epoch): {loss_meter.avg}'
                    )
                logging.info(message)

        # Validate
        loss_meter.reset()
        for iter_ind in range(1, valid_dataset.n_iters_per_epoch + 1):
            image_input, text_input = next(valid_dataset_iter)
            loss = valid_iter(state, image_input, text_input, labels)
            loss_meter.update(loss[0], len(image_input))

            if iter_ind % log_freq == 0 or iter_ind == valid_dataset.n_iters_per_epoch:
                message = (
                    f'Validation epoch: {epoch}/{n_epochs} '
                    f'Iteration: {iter_ind}/{valid_dataset.n_iters_per_epoch} '
                    f'Loss (current iteration): {loss_meter.val} '
                    f'Loss (average of current epoch): {loss_meter.avg}'
                    )
                logging.info(message)

        # Checkpoint
        if epoch % checkpoint_freq == 0 or epoch == n_epochs:
            save_checkpoint(checkpoint_dir, state, epoch)
