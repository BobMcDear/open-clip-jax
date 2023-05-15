"""
Trains CLIP models.
"""


import logging
from argparse import ArgumentParser, Namespace
from functools import partial

import optax
import tensorflow as tf
import jax
from flax.training.dynamic_scale import DynamicScale
from jax import numpy as jnp


from open_clip_jax.clip import CLIPWithLoss, create_model, list_models
from open_clip_jax.training.data import create_csv_dataset
from open_clip_jax.training.optimizer import create_weight_decay_mask
from open_clip_jax.training.scheduler import create_learning_rate_scheduler
from open_clip_jax.training.train import TrainState, train_and_validate


def parse_args() -> Namespace:
    """
    Parses arguments for training CLIP models. See body for all arguments.

    Returns:
        Parsed arguments.
    """
    parser = ArgumentParser()

    # Data loading
    parser.add_argument(
        '--train-path-csv',
        type=str,
        required=True,
        help='Path to a CSV file containing image paths and text captions for training.',
        )
    parser.add_argument(
        '--valid-path-csv',
        type=str,
        required=True,
        help='Path to a CSV file containing image paths and text captions for validation.',
        )
    parser.add_argument(
        '--col-ind-image',
        type=int,
        default=0,
        help='Index of the column in the CSV file containing image paths.',
        )
    parser.add_argument(
        '--col-ind-text',
        type=int,
        default=1,
        help='Index of the column in the CSV file containing text captions.',
        )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Size to which the images are resized.',
        )
    parser.add_argument(
        '--context-len',
        type=int,
        default=77,
        help='Context length of the tokenized text. Tokens are padded or truncated to ensure the number of tokens is context_len.',
        )
    parser.add_argument(
        '--global-batch-size',
        type=int,
        default=64,
        help='Global batch size across all devices.',
        )
    parser.add_argument(
        '--shuffle-buffer-size',
        type=int,
        default=None,
        help='Buffer size for shuffling the training set. If None, it is set to 16*global_batch_size (general rule of thumb).',
        )

    # CLIP model
    parser.add_argument(
        '--model-name',
        type=str,
        choices=list_models(),
        default='vit-base-patch32',
        help='Name of CLIP model to train. See open_clip_jax/clip/model_configs/ for available options.',
        )
    parser.add_argument(
        '--temp-init',
        type=float,
        default=2.6593,
        help='Initial value for a learnable temperature coefficient the logits are scaled by when calculating the loss, with None for no scaling.',
        )

    # Learning rate
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-4,
        help=(
            'Peak learning rate for the cosine decay scheduler and the constant learning rate for the constant scheduler. '
            'See --learning-rate-scheduler to select the scheduler. '
            'Note that the learning rate is not scaled automatically, and the user should adjust it according to the batch size.'
            ),
        )
    parser.add_argument(
        '--learning-rate-scheduler',
        type=str,
        choices=['cosine', 'const'],
        default='cosine',
        help='Name of learning rate scheduler to use, with cosine for cosine decay and const for a constant schedule.',
        )
    parser.add_argument(
        '--n-warmup-steps',
        type=int,
        default=10000,
        help='Number of warmup steps for the learning rate scheduler.',
        )
    parser.add_argument(
        '--warmup-init',
        type=float,
        default=1e-5,
        help='Initial learning rate when beginning warmup.',
        )

    # Optimizer
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        help='Exponential decay rate used to track the running mean of gradients for AdamW.',
        )
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.98,
        help='Exponential decay rate used to track the running mean of the square of gradients for AdamW.',
        )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-6,
        help='Epsilon value for AdamW.',
        )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=2e-1,
        help='Weight decay for AdamW.',
        )

    # Training
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=32,
        help='Number of epochs to train for.',
        )
    parser.add_argument(
        '--dtype',
        type=str,
        choices=['float32', 'float16', 'bfloat16'],
        default='float32',
        help=(
            'The data type training is performed in. '
            'Note that some operations are conducted in float32 regardless of --dtype (i.e., automatic mixed precision).'
            ),
        )

    # Logging
    parser.add_argument(
        '--log-freq',
        type=int,
        default=100,
        help='Training and validation information are logged to console every --log-freq iterations.',
        )
    parser.add_argument(
        '--save-freq',
        type=int,
        default=5,
        help='The parameters of the CLIP model are saved every --save-freq epochs.',
        )

    args = parser.parse_args()
    return args


def setup_logging():
    """
    Sets up Python logger.
    """
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.root.setLevel(logging.INFO)
    logging.root.handlers = [handler]


def main(args: Namespace) -> None:
    """
    Trains CLIP models. See parse_args for arguments.
    """
    setup_logging()

    logging.info('Parsed arguments:')
    for arg_name in args.__dict__:
        logging.info(f'{arg_name}: {getattr(args, arg_name)}')

    logging.info('Creating datasets...')
    create_csv_dataset_with_args = partial(
        create_csv_dataset,
        col_ind_image=args.col_ind_image,
        col_ind_text=args.col_ind_text,
        image_size=args.image_size,
        context_len=args.context_len,
        n_epochs=args.n_epochs,
        global_batch_size=args.global_batch_size,
        dtype=getattr(tf, args.dtype),
        )
    train_dataset = create_csv_dataset_with_args(
        path_csv=args.train_path_csv,
        train=True,
        shuffle_buffer_size=args.shuffle_buffer_size,
        )
    valid_dataset = create_csv_dataset_with_args(
        path_csv=args.valid_path_csv,
        train=False,
        )
    logging.info('Datasets created')

    logging.info('Creating model...')
    dtype = getattr(jnp, args.dtype)
    model = create_model(model_name=args.model_name, dtype=dtype)
    model_with_loss = CLIPWithLoss(model, temp_init=args.temp_init)
    local_batch_size = args.global_batch_size // jax.local_device_count()
    vars = model_with_loss.init(
        rngs=jax.random.PRNGKey(0),
        image_input=jnp.empty((local_batch_size, args.image_size, args.image_size, 3), dtype=dtype),
        text_input=jnp.empty((local_batch_size, args.context_len), dtype=jnp.int32),
        )
    logging.info(f'Model created: {model}')

    learning_rate = create_learning_rate_scheduler(
        scheduler_name=args.learning_rate_scheduler,
        learning_rate=args.learning_rate,
        n_train_iters=args.n_epochs*train_dataset.n_iters_per_epoch,
        n_warmup_steps=args.n_warmup_steps,
        warmup_init=args.warmup_init,
        )
    optim = optax.adamw(
        learning_rate=learning_rate,
        b1=args.beta1,
        b2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        mask=create_weight_decay_mask,
        )

    logging.info(f'Beginning training on device(s) {jax.local_devices()}...')
    state = TrainState.create(
        apply_fn=model_with_loss.apply,
        params=vars['params'],
        tx=optim,
        labels=vars['labels'],
        dynamic_scale=DynamicScale() if dtype is jnp.float16 else None,
        )
    train_and_validate(
        state=state,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        n_epochs=args.n_epochs,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        )
    logging.info('Training finished')


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
