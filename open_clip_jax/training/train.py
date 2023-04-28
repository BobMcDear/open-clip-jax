"""
Core code for training CLIP models.
"""


import logging
import pickle
from typing import Any, Iterable, Tuple
from functools import partial

import jax
import tensorflow as tf
from flax import jax_utils
from flax.core import frozen_dict
from flax.linen.dtypes import Array
from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale
from jax import lax
from jax import numpy as jnp


PyTree = Any


class AvgMeter:
    """
    Average meter that keeps track of a series of values and their average.
    """
    def __init__(self) -> None:
        """
        Sets up the tracker.
        """
        self.reset()

    def reset(self) -> None:
        """
        Resets the meter.
        """
        self.val = 0.
        self.sum = 0.
        self.count = 0
        self.avg = 0.

    def update(self, val: Array, count: int) -> None:
        """
        Updates the tracker.

        Args:
            val: Value to add to the tracker.
            count: Count of entries val accounts for.
        """
        self.val = val
        self.sum += count*val
        self.count += count
        self.avg = self.sum/self.count


class TrainState(train_state.TrainState):
    """
    Flax training state for CLIP models with support for CLIP loss labels and
    dynamic loss scaling.

    See base class.
    """
    labels: Array
    dynamic_scale: DynamicScale


@partial(jax.pmap, axis_name='devices')
def train_iter(
    state: TrainState,
    image_input: Array,
    text_input: Array,
    ) -> Tuple[TrainState, Array]:
    """
    Performs one training iteration.

    Args:
        state: Training state whose apply function returns the CLIP loss.
        image_input: Input to the image model.
        text_input: Input to the text model.

    Returns:
        Updated training state and loss.
    """
    def loss_fn(params):
        vars = {
            'params': params,
            'labels': state.labels,
            }
        return state.apply_fn(vars, image_input, text_input)

    if state.dynamic_scale:
        loss_and_grad_fn = state.dynamic_scale.value_and_grad(
            fun=loss_fn,
            axis_name='devices',
            )
        dynamic_scale, is_finite, loss, grads = loss_and_grad_fn(state.params)
        # Dynamic scale averages gradients across devices automatically
    else:
        loss_and_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(state.params)
        grads = lax.pmean(grads, axis_name='devices')

    # After update, in case of mixed-precision training,
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
    ) -> Array:
    """
    Performs one validation iteration.

    Args:
        state: Training state whose apply function returns the CLIP loss.
        image_input: Input to the image model.
        text_input: Input to the text model.

    Returns:
        Loss.
    """
    vars = {
        'params': state.params,
        'labels': state.labels,
        }
    loss = state.apply_fn(vars, image_input, text_input)
    return lax.pmean(loss, axis_name='devices')


def tf_to_np(pytree: PyTree, device_axis: bool = True) -> PyTree:
    """
    Converts TensorFlow tensors into NumPy arrays.

    Args:
        pytree: PyTree with TensorFlow tensors as leaves.
        device_axis: Whether to add a leading device axis to the data for
            distributed training.

    Returns:
        Input PyTree with its leaves converted into NumPy arrays and
        potentially an additional device axis.
    """
    device_count = jax.local_device_count()
    def _tf_to_jax(leaf):
        leaf = leaf._numpy()
        if device_axis:
            # [global_batch_size, ...] to [device_count, local_batch_size, ...]
            leaf = leaf.reshape((device_count, -1) + leaf.shape[1:])
        return leaf

    return jax.tree_util.tree_map(_tf_to_jax, pytree)


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
    dataset_iter = map(partial(tf_to_np, device_axis=device_axis), dataset)
    # Pre-fetching data to device speeds up GPU training,
    # but is not necessary for CPU/TPU.
    if jax.local_devices()[0].platform == 'gpu':
        dataset_iter = jax_utils.prefetch_to_device(dataset_iter, size=2)
    return dataset_iter


def train_and_validate(
    state: TrainState,
    train_dataset: tf.data.Dataset,
    valid_dataset: tf.data.Dataset,
    n_epochs: int = 32,
    log_freq: int = 100,
    save_freq: int = 5,
    ) -> None:
    """
    Trains a CLIP model and validates after each epoch.

    Args:
        state: Training state whose apply function returns the CLIP loss.
        train_dataset: Dataset returning (image, text) pairs for training and
            an n_iters_per_epoch attribute denoting the number of iterations
            per epoch.
        valid_dataset: Dataset returning (image, text) pairs for validation and
            an n_iters_per_epoch attribute denoting the number of iterations
            per epoch.
        n_epochs: Number of epochs to train for.
        log_freq: Training and validation information are logged to console
            every log_freq iterations.
        save_freq: The CLIP model's parameters are saved every save_freq
            epochs.
    """
    state = jax_utils.replicate(state)
    train_dataset_iter = tf_dataset_to_np_iter(train_dataset)
    valid_dataset_iter = tf_dataset_to_np_iter(valid_dataset)
    loss_meter = AvgMeter()

    for epoch in range(1, n_epochs+1):
        logging.info(f'Beginning epoch {epoch}...')

        # Train
        loss_meter.reset()
        for iter_ind in range(1, train_dataset.n_iters_per_epoch+1):
            image_input, text_input = next(train_dataset_iter)
            state, loss = train_iter(state, image_input, text_input)
            loss_meter.update(loss[0], len(image_input))

            # Temperature coefficient is clipped to [0, ln(100)].
            params = state.params.unfreeze()
            params['CLIPLoss_0']['temp'] = jnp.clip(
                a=params['CLIPLoss_0']['temp'],
                a_min=0.0,
                a_max=4.6052,
                )
            state = state.replace(params=frozen_dict.freeze(params))

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
        for iter_ind in range(1, valid_dataset.n_iters_per_epoch+1):
            image_input, text_input = next(valid_dataset_iter)
            loss = valid_iter(state, image_input, text_input)
            loss_meter.update(loss[0], len(image_input))

            if iter_ind % log_freq == 0 or iter_ind == valid_dataset.n_iters_per_epoch:
                message = (
                    f'Validation epoch: {epoch}/{n_epochs} '
                    f'Iteration: {iter_ind}/{valid_dataset.n_iters_per_epoch} '
                    f'Loss (current iteration): {loss_meter.val} '
                    f'Loss (average of current epoch): {loss_meter.avg}'
                    )
                logging.info(message)

        # Save
        if epoch % save_freq == 0 or epoch == n_epochs:
            logging.info(f'Saving checkpoint for epoch {epoch}')
            with open(f'clip_epoch_{epoch}.pkl', 'wb') as file:
                pickle.dump(
                    obj={'params': state.params['model']},
                    file=file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                    )
