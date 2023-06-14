"""
Data loading with TensorFlow for training CLIP models.
"""


from typing import Callable, Optional, Tuple
from functools import partial

import jax
import tensorflow as tf

from ..clip import image_transforms
from ..clip.tokenizer import tokenize


def image_item_transforms(
    path_image: tf.Tensor,
    train: bool,
    size: int = 224,
    ) -> tf.Tensor:
    """
    Image transforms applied over individual samples. The training transform
    is random resized cropping, and the validation transforms are resizing and
    center-cropping.

    Args:
        path_image: Path to image to load and transform.
        train: Whether to apply training transforms (True) or validation
            transforms (False).
        size: Size to which the image is resized.

    Returns:
        Image at path_image loaded and transformed using the appropriate
        transforms.
    """
    bytes = tf.io.read_file(path_image)

    if train:
        image = image_transforms.random_resized_crop(bytes, size)

    else:
        image = tf.io.decode_jpeg(bytes, channels=3)
        image = image_transforms.resize_smallest_edge(image, size)
        image = image_transforms.center_crop_with_padding(image, size)

    return image


def image_batch_transforms(
    image: tf.Tensor,
    aug: Optional[Callable] = None,
    dtype: tf.DType = tf.float32,
    ) -> tf.Tensor:
    """
    Image transforms applied over batches of samples for greater efficiency.
    They are optional augmentations, normalization, and data type conversion.

    Args:
        image: Batch of images to transform.
        aug: Optional augmentations. If None, no augmentations are applied.
        dtype: The data type the batch of images is converted to.

    Returns:
        Batch of images normalized, converted to the desired data type, and
        possibly augmented.
    """
    if aug:
        image = aug(image)
    image = image_transforms.normalize(image)
    image = tf.image.convert_image_dtype(image, dtype)
    return image


@tf.function
def map_item(
    path_image: tf.Tensor,
    text: tf.Tensor,
    train: bool,
    image_size: int = 224,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Receives the path to an image and a text, loads and transforms the
    image and returns the text as-is. See also image_item_transforms.

    Args:
        path_image: Path to image to load and transform.
        text: Text.
        train: Whether to apply training transforms (True) or validation
            transforms (False) to the image.
        image_size: Size to which the image is resized.

    Returns:
        Image at path_image loaded and transformed using the appropriate
        transforms and the text as-is.
    """
    return image_item_transforms(path_image, train, image_size), text


@tf.function
def map_batch(
    image: tf.Tensor,
    text: tf.Tensor,
    tokenizer: Callable,
    aug: Optional[Callable] = None,
    dtype: tf.DType = tf.float32,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Receives a batch of images and text, transforms the images and tokenizes
    the text. See also image_batch_transforms.

    Args:
        image: Batch of images to transform.
        text: Batch of text to tokenize.
        tokenizer: Tokenizer used to tokenize the batch of text.
        aug: Optional augmentations applied to the batch of images. If None, no
            augmentations are applied.
        dtype: The data type the batch of images is converted to.

    Returns:
        Batch of images normalized, converted to the desired data type, and
        possibly augmented, and the batch of text tokenized.
    """
    return image_batch_transforms(image, aug, dtype), tokenizer(text)


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

    map_item_with_args = partial(map_item, train=train, image_size=image_size)
    map_batch_with_args = partial(
        map_batch,
        tokenizer=partial(tokenize, context_len=context_len),
        dtype=dtype,
        )

    if train:
        shuffle_buffer_size = shuffle_buffer_size or 16 * global_batch_size
        dataset = dataset.shuffle(shuffle_buffer_size)

    batch_size_per_process = global_batch_size // jax.process_count()
    dataset = (dataset
        .repeat(n_epochs)
        .map(map_item_with_args, tf.data.AUTOTUNE)
        .batch(batch_size_per_process, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        .map(map_batch_with_args, tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        )
    dataset.n_iters_per_epoch = n_samples // global_batch_size

    return dataset
