"""
Image transforms not provided by TensorFlow.
"""


from typing import Tuple

import tensorflow as tf

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def shape(image: tf.Tensor) -> Tuple[int, int]:
    """
    Gets the height and width of the input image.

    Args:
        image: Input image of shape [..., height, width, channels].

    Returns:
        Heigh and width of the input image.
    """
    shape_ = tf.shape(image)
    return int(shape_[-3]), int(shape_[-2])


def random_resized_crop(
    bytes: tf.Tensor,
    size: int = 224,
    method: str = 'bicubic',
    ratio: Tuple[float, float] = (3/4, 4/3),
    scale: Tuple[float, float] = (0.9, 1.0),
    antialias: bool = True,
    ) -> tf.Tensor:
    """
    Extracts a random crop from the input and resizes it to the desired size.
    Inspired by torchvision's RandomResizedCrop.

    Args:
        bytes: JPEG-encoded bytes of the input image.
        size: Size to which the random crop is resized and returned.
        method: Method to use to resize the random crop. See
            tf.image.ResizeMethod for available options.
        ratio: The lower and upper bounds for the aspect ratio of the
            random crop.
        scale: The lower and upper bounds for the area of the random crop,
            with respect to the area of the input image.
        antialias: Whether to anti-alias when resizing the random crop.

    Returns:
        A random crop of the input image resized to the desired size.
    """
    crop_top_left, crop_size, _ = tf.image.sample_distorted_bounding_box(
        image_size=tf.io.extract_jpeg_shape(bytes),
        bounding_boxes=tf.constant([0., 0., 1., 1.], shape=[1, 1, 4]),
        aspect_ratio_range=ratio,
        area_range=scale,
        max_attempts=5,
        )
    crop_top_left_y, crop_top_left_x, _ = tf.unstack(crop_top_left)
    crop_height, crop_width, _ = tf.unstack(crop_size)
    crop = tf.stack([crop_top_left_y, crop_top_left_x, crop_height, crop_width])

    cropped = tf.io.decode_and_crop_jpeg(bytes, crop, channels=3)
    resized = tf.image.resize(cropped, [size, size], method, antialias=antialias)
    return resized


def resize_smallest_edge(
    image: tf.Tensor,
    size: int = 224,
    method: str = 'bicubic',
    antialias: bool = True,
    ) -> tf.Tensor:
    """
    Resizes an image so the smallest edge is resized to the desired size and
    the aspect ratio is maintained. Inspired by torchvision's Resize.

    Args:
        image: Image to resize.
        size: Size to which the smallest edge of the input image is resized.
        method: Resizing method to use. See tf.image.ResizeMethod for available
            options.
        antialias: Whether to anti-alias when resizing.

    Returns:
        Input image with its smallest edge resized to the desired size and its
        aspect ratio maintained.
    """
    image_w, image_h = shape(image)
    if image_w <= image_h:
        size = (int(size * image_h / image_w), size)
    else:
        size = (size, int(size * image_w / image_h))
    return tf.image.resize(image, size, method, antialias=antialias)


def center_crop_with_padding(
    image: tf.Tensor,
    size: int = 224,
    ) -> tf.Tensor:
    """
    Extracts a central crop of the desired size from the input image. If the
    input image is smaller than the desired size, it is padded first. Inspired
    by torchvision's CenterCrop.

    Args:
        image: Image to center-crop.
        size: Desired size of the center-crop.

    Returns:
        Center-crop of the input image, padded if the input image is smaller
        than the desired size.
    """
    image_h, image_w = shape(image)
    padded = tf.image.pad_to_bounding_box(
        image=image,
        offset_height=tf.maximum(0, (size - image_h) // 2),
        offset_width=tf.maximum(0, (size - image_w) // 2),
        target_height=tf.maximum(size, image_h),
        target_width=tf.maximum(size, image_w),
        )

    padded_h, padded_w = shape(padded)
    cropped = tf.image.crop_to_bounding_box(
        image=padded,
        offset_height=(padded_h - size) // 2,
        offset_width=(padded_w - size) // 2,
        target_height=size,
        target_width=size,
        )

    return cropped


def normalize(
    image: tf.Tensor,
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN,
    std: Tuple[float, ...] = OPENAI_DATASET_STD,
    ) -> tf.Tensor:
    """
    Normalizes the input image at the last axis.

    Args:
        image: Image to normalize.
        mean: Mean per channel used for normalization.
        std: Standard deviation per channel used for normalization.

    Returns:
        Input image normalized using mean and std at the last axis.
    """
    image -= 255*tf.constant(mean, shape=[1, 1, 3], dtype=image.dtype)
    image /= 255*tf.constant(std, shape=[1, 1, 3], dtype=image.dtype)
    return image