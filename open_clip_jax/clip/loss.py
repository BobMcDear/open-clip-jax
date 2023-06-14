"""
CLIP (contrastive) loss.
"""


from typing import Any, Hashable, Optional

import jax
from flax.linen.dtypes import Array
from jax import numpy as jnp
from optax import softmax_cross_entropy_with_integer_labels


PyTree = Any


def all_gather(pytree: PyTree, device_axis_name: Hashable = 'devices') -> PyTree:
    """
    All-gathers the leaves of a PyTree (assumed to be batches) and concatenates
    them along the batch axis. Copied from https://github.com/google-research/big_vision.

    Args:
        pytree: PyTree to all-gather.
        device_axis_name: Name of the device axis.

    Returns:
        PyTree with its leaves all-gathered and concatenated along the batch axis.
    """
    return jax.tree_util.tree_map(
        lambda leaf: jnp.concatenate(jax.lax.all_gather(leaf, axis_name=device_axis_name)),
        pytree,
        )


def generate_labels(batch_size_per_process: int) -> Array:
    """
    Generates labels that can be used to calculate the cross-entropy loss of the
    similarity logits. This function assumes the loss is being computed
    local-to-globally and adds an offset to the labels accordingly.

    Args:
        batch_size_per_process: Batch size per process.

    Returns:
        Labels for calculating the cross-entropy loss of the similarity logits,
        sharded over local devices.
    """
    local_devices = jax.local_devices()
    n_local_devices = len(local_devices)

    offset = jax.process_index() * batch_size_per_process
    labels = jnp.arange(batch_size_per_process) + offset
    labels = jnp.reshape(labels, (n_local_devices, -1))

    # Labels are sharded so they don't have to be transferred to the appropriate
    # device each time in pmaps.
    shards = [labels[device_ind] for device_ind in range(n_local_devices)]
    sharded = jax.device_put_sharded(shards, devices=local_devices)
    return sharded


def clip_loss(
    image_proj: Array,
    text_proj: Array,
    labels: Array,
    device_axis_name: Optional[Hashable] = None,
    ) -> Array:
    """
    Calculates the CLIP (contrastive) loss given projected image and text vectors.

    Args:
        image_proj: Projected image vectors.
        text_proj: Projected text vectors.
        labels: Labels used to calculate the cross-entropy loss of the similarity
            logits.
        device_axis_name: If None, the similarity logits are calculated locally
            and depend only on image_proj and text_proj. Otherwise, image_proj
            and text_proj are all-gathered, assuming device_axis_name is the
            name of a pmapped axis, and local-to-global similarity logits are
            computed.

    Returns:
        The CLIP (contrastive) loss of the similarity logits.
    """
    if device_axis_name is None:
        logits_per_image = image_proj @ text_proj.T
        logits_per_text = logits_per_image.T

    else:
        all_image_proj, all_text_proj = all_gather((image_proj, text_proj))
        logits_per_image = image_proj @ all_text_proj.T
        logits_per_text = text_proj @ all_image_proj.T

    return jnp.mean(
            softmax_cross_entropy_with_integer_labels(logits_per_image, labels) +
            softmax_cross_entropy_with_integer_labels(logits_per_text, labels)
            ) / 2
