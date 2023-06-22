"""
Data loading with TensorFlow for training CLIP models.
"""


from typing import Optional, Tuple

import jax
import tensorflow as tf

from ..clip.image_transforms import create_image_transforms
from ..clip.tokenizer import tokenize


def create_csv_dataset(
    path_csv: str,
    train: bool,
    col_ind_image: int = 0,
    col_ind_text: int = 1,
    image_size: int = 224,
    context_len: int = 77,
    n_epochs: int = 32,
    global_batch_size: int = 64,
    shuffle_buffer_size: Optional[int] = None,
    dtype: tf.DType = tf.float32,
    ) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset of (image, text) pairs, sharded across all
    processes, given a CSV file of image paths and text captions. See also
    map_item and map_batch.

    Args:
        path_csv: Path to a CSV file containing image paths and text captions.
        train: Whether to shuffle and apply training transforms (True) or
            validation transforms (False) to the images.
        col_ind_image: Index of the column in the CSV file containing image
            paths.
        col_ind_text: Index of the column in the CSV file containing text
            captions.
        image_size: Size to which the images are resized.
        context_len: Context length of the tokenized text. Tokens are padded or
            truncated to ensure the number of tokens is context_len.
        n_epochs: Number of epochs the model will train for.
        global_batch_size: Global batch size across all processes and devices.
        shuffle_buffer_size: Buffer size for shuffling when train is True. If
            None, it is set to 16*global_batch_size (general rule of thumb).
        dtype: The data type the images are converted to.

    Returns:
        Sharded dataset of (image, text) pairs and an n_iters_per_epoch attribute
        denoting the number of iterations per epoch.
    """
    with tf.io.gfile.GFile(path_csv) as file:
        # Minus one for the header.
        n_samples = sum(1 for _ in file) - 1

    dataset = tf.data.experimental.CsvDataset(
        path_csv,
        record_defaults=[tf.string, tf.string],
        header=True,
        select_cols=[col_ind_image, col_ind_text],
        )

    # Number of samples is rounded down to the nearest multiple of the batch size
    # to ensure the number of batches is identical across all processes.
    n_iters_per_epoch = n_samples // global_batch_size
    dataset = dataset.take(global_batch_size * n_iters_per_epoch)
    dataset = dataset.shard(
        num_shards=jax.process_count(),
        index=jax.process_index(),
        )

    # These specific values are used by most Google projects,
    # e.g., Big Vision and Scenic, and generally accelerate data loading.
    # Faster alternatives may exist depending on the hardware though.
    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = 1
    options.threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)

    image_item_transforms, image_batch_transforms = create_image_transforms(
        train=train,
        input_format='path',
        size=image_size,
        dtype=dtype,
        )

    def map_item(
        path_image: tf.Tensor,
        text: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        return image_item_transforms(path_image), text

    def map_batch(
        image: tf.Tensor,
        text: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        return image_batch_transforms(image), tokenize(text, context_len)

    if train:
        shuffle_buffer_size = shuffle_buffer_size or 16 * global_batch_size
        dataset = dataset.shuffle(shuffle_buffer_size)

    batch_size_per_process = global_batch_size // jax.process_count()
    dataset = (dataset
        .repeat(n_epochs)
        .map(map_item, tf.data.AUTOTUNE)
        .batch(batch_size_per_process, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        .map(map_batch, tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        )
    dataset.n_iters_per_epoch = n_samples // global_batch_size

    return dataset
