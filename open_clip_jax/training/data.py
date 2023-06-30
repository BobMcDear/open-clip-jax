"""
Data loading with TensorFlow for training CLIP models.
"""


from typing import Optional, Tuple

import jax
import tensorflow as tf

from ..clip.image_transforms import create_image_transforms
from ..clip.tokenizer import tokenize


def prepare_csv_dataset(
    path_csv: str,
    n_samples: Optional[int] = None,
    image_key: str = 'jpg',
    text_key: str = 'caption',
    ) -> Tuple[tf.data.Dataset, int]:
    """
    Prepares a basic CSV dataset yielding (path to image, text) pairs.

    Args:
        path_csv: Path to a CSV file containing image paths and text captions.
        n_samples: Number of samples from the CSV file to return. If None, all
            the samples are returned.
        image_key: Name of column in the CSV file containing image paths.
        text_key: Name of column in the CSV file containing text captions.

    Returns:
        Dataset yielding (path to image, text) pairs and the number of samples.
    """
    with tf.io.gfile.GFile(path_csv) as file:
        cols = file.readline().strip().split(',')
        col_ind_image, col_ind_text = cols.index(image_key), cols.index(text_key)
        n_samples = n_samples or sum(1 for _ in file)

    dataset = tf.data.experimental.CsvDataset(
        path_csv,
        record_defaults=[tf.string, tf.string],
        header=True,
        select_cols=[col_ind_image, col_ind_text],
        )
    dataset = dataset.take(n_samples)

    # Sharding should happen early in the pipeline to avoid
    # every worker processing all samples.
    dataset = dataset.shard(
        num_shards=jax.process_count(),
        index=jax.process_index(),
        )

    return dataset, n_samples


def prepare_tfrecord_dataset(
    path_tfrecord: str,
    train: bool,
    image_key: str = 'jpg',
    text_key: str = 'caption',
    ) -> tf.data.Dataset:
    """
    Prepares a basic TFRecord dataset yielding (image bytes, text) pairs.

    Args:
        path_tfrecord: Path to a directory of TFRecords.
        train: Whether to shuffle the TFRecord files.
        image_key: Key in each TFRecord example containing image bytes.
        text_key: Key in each TFRecord example containing text captions.

    Returns:
        Dataset yielding (image bytes, text) pairs.
    """
    # path_tfrecord cannot be, e.g., 'path' and must be 'path/'.
    if path_tfrecord[-1] != '/':
        path_tfrecord += '/'

    files = tf.io.gfile.glob(f'{path_tfrecord}*.tfrecord')
    files = files[jax.process_index()::jax.process_count()] # Shard

    # Instead of passing the file names to TFRecordDataset directly,
    # they are wrapped in a Dataset first to be able to shuffle the files
    # before shuffling the examples within them (shuffling a TFRecordDataset
    # achieves the latter).
    files_dataset = tf.data.Dataset.from_tensor_slices(files)
    files_dataset = files_dataset.shuffle(len(files)) if train else files_dataset
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

    def parse_tfrecord(serialized_example: tf.Tensor):
        features = {
            image_key: tf.io.FixedLenFeature([], tf.string),
            text_key: tf.io.FixedLenFeature([], tf.string),
            }
        parsed = tf.io.parse_single_example(serialized_example, features)
        return parsed[image_key], parsed[text_key]

    return dataset.map(parse_tfrecord, tf.data.AUTOTUNE)


def create_dataset(
    path: str,
    train: bool,
    n_samples: Optional[int] = None,
    image_key: str = 'jpg',
    text_key: str = 'caption',
    image_size: int = 224,
    context_len: int = 77,
    global_batch_size: int = 64,
    shuffle_buffer_size: Optional[int] = None,
    dtype: tf.DType = tf.float32,
    ) -> tf.data.Dataset:
    """
    Creates a dataset of batches of (image, text) pairs for image-language
    training from a CSV file or TFRecords.

    Args:
        path: Path to a CSV file containing image paths and text captions or
            path to a directory of TFRecords.
        train: Whether to shuffle and apply training transforms (True) or
            validation transforms (False) to the images.
        n_samples: If path points to a CSV file, n_samples controls how many samples
            are included in the dataset, with None denoting all the samples.
            Otherwise, n_samples should be the number of examples in all the
            TFRecords combined and must be provided.
        image_key: Name of CSV column or TFRecord key containing image paths or
            bytes.
        text_key: Name of CSV column or TFRecord key containing text captions.
        image_size: Size to which the images are resized.
        context_len: Context length of the tokenized text. Tokens are padded or
            truncated to ensure the number of tokens is context_len.
        global_batch_size: Global batch size across all processes and devices.
        shuffle_buffer_size: Buffer size for shuffling when train is True. If
            None, it is set to 16*global_batch_size (general rule of thumb).
        dtype: The data type the images are converted to.

    Returns:
        Dataset yielding batches of (image, text) pairs from the specified source.

    Raises:
        ValueError: n_samples is not provided for a TFRecord dataset.
    """
    if '.csv' in path:
        dataset, n_samples = prepare_csv_dataset(
            path,
            n_samples=n_samples,
            image_key=image_key,
            text_key=text_key,
            )
        image_input_format = 'path'

    else:
        if n_samples is None:
            raise ValueError('n_samples must be provided for TFRecord datasets.')

        dataset = prepare_tfrecord_dataset(
            path,
            train=train,
            image_key=image_key,
            text_key=text_key,
            )
        image_input_format = 'bytes'

    # A threadpool size of 48 is used by most Google projects,
    # e.g., Big Vision and Scenic, and generally accelerate data loading, on TPUs
    # especially. Faster alternatives may exist depending on the hardware though.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.deterministic = False
    dataset = dataset.with_options(options)

    image_item_transforms, image_batch_transforms = create_image_transforms(
        train=train,
        input_format=image_input_format,
        size=image_size,
        dtype=dtype,
        )

    def map_item(image: tf.Tensor, text: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return image_item_transforms(image), text

    def map_batch(image: tf.Tensor, text: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return image_batch_transforms(image), tokenize(text, context_len)

    # The data pipeline is structured as follows:
    # Shuffle -> repeat -> item transforms -> batch -> batch transforms -> prefetch.
    # Repeating before shuffling would slightly improve performance at the cost
    # of blurry epoch boundaries.
    batch_size_per_process = global_batch_size // jax.process_count()
    shuffle_buffer_size = shuffle_buffer_size or 16 * global_batch_size
    dataset = dataset.shuffle(shuffle_buffer_size) if train else dataset
    dataset = (dataset
        .repeat()
        .map(map_item, tf.data.AUTOTUNE)
        .batch(batch_size_per_process, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        .map(map_batch, tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        )
    dataset.n_iters_per_epoch = n_samples // global_batch_size

    return dataset
