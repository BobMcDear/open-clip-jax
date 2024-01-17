"""
ConvNeXt vision model.
"""


from typing import Optional, Tuple

from flax import linen as nn
from flax.linen.dtypes import Array, Dtype
from jax import numpy as jnp

from .basic_layers import MLP, gelu, global_avg_pool
from .transformer import PatchEmbed


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block.

    Attributes:
        out_dim: Number of output features.
            If None, it is set to the number of input channels.
        layer_scale_init_value: Value for initializing LayerScale.
            If None, no LayerScale is applied.
        eps: Epsilon value passed to the layer normalization modules.
        dtype: The data type of the computations.
    """
    out_dim: Optional[int] = None
    layer_scale_init_value: Optional[float] = 1e-6
    eps: float = 1e-6
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input: Array) -> Array:
        """
        Passes the input through the ConvNeXt block.

        Args:
            input: Input passed through the ConvNeXt block.

        Returns:
            Output of the ConvNeXt block.
        """
        out_dim = self.out_dim or input.shape[-1]
        output = nn.Conv(
            features=out_dim,
            kernel_size=(7, 7),
            padding=2*((3, 3),),
            feature_group_count=out_dim,
            dtype=self.dtype,
            )(input)
        output = nn.LayerNorm(
            epsilon=self.eps,
            dtype=self.dtype,
            )(output)
        output = MLP(
            dtype=self.dtype,
            )(output)
        output = output * self.param(
            name='lambda',
            init_fn=lambda _: self.layer_scale_init_value * jnp.ones(input.shape[-1]),
            ) if self.layer_scale_init_value else output
        return input + output if input.shape == output.shape else output


class ConvNeXtStage(nn.Module):
    """
    ConvNeXt stage.

    Attributes:
        depth: Number of ConvNeXt blocks.
        out_dim: Number of output features.
            If None, it is set to the number of input channels.
        stride: Stride.
        layer_scale_init_value: Value for initializing LayerScale.
            If None, no LayerScale is applied.
        eps: Epsilon value passed to the layer normalization modules.
        grad_checkpoint: Whether to perform gradient checkpointing on the
            ConvNeXt blocks. If True, intermediate activations are not stored
            and are recomputed during backpropagation.
        dtype: The data type of the computations.
    """
    depth: int
    out_dim: int
    stride: int = 1
    layer_scale_init_value: Optional[float] = 1e-6
    eps: float = 1e-6
    grad_checkpoint: bool = False
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input):
        """
        Passes the input through the ConvNeXt stage.

        Args:
            input: Input passed through the ConvNeXt stage.

        Returns:
            Output of the ConvNeXt stage.
        """
        if (self.stride != 1) or (input.shape[-1] != self.out_dim):
            input = PatchEmbed(
                embed_dim=self.out_dim,
                patch_size=self.stride,
                eps=self.eps,
                norm_first=True,
                flatten=False,
                dtype=self.dtype,
                )(input)

        block = nn.remat(ConvNeXtBlock) if self.grad_checkpoint else ConvNeXtBlock
        for _ in range(self.depth):
            input = block(
                layer_scale_init_value=self.layer_scale_init_value,
                eps=self.eps,
                dtype=self.dtype,
                )(input)

        return input


class ConvNeXt(nn.Module):
    """
    ConvNeXt.

    Attributes:
        depths: Number of ConvNeXt blocks per stage.
        out_dims: Number of output features per stage.
        layer_scale_init_value: Value for initializing LayerScale.
            If None, no LayerScale is applied.
        eps: Epsilon value passed to the layer normalization modules.
        grad_checkpoint: Whether to perform gradient checkpointing on the
            ConvNeXt blocks. If True, intermediate activations are not stored
            and are recomputed during backpropagation.
        head_hidden_dim: If not None, the output logits are transformed to
            this dimensionality using a linear transformation.
        dtype: The data type of the computations.
    """
    depths: Tuple[int, ...]
    out_dims: Tuple[int, ...]
    layer_scale_init_value: Optional[float] = 1e-6
    eps: float = 1e-6
    grad_checkpoint: bool = False
    head_hidden_dim: Optional[int] = None
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input: Array) -> Array:
        """
        Passes the input through the ConvNeXt.

        Args:
            input: Input passed through the ConvNeXt.

        Returns:
            Output of the ConvNeXt.
        """
        output = PatchEmbed(
            embed_dim=self.out_dims[0],
            patch_size=4,
            eps=self.eps,
            flatten=False,
            dtype=self.dtype,
            )(input)

        for ind in range(len(self.depths)):
            output = ConvNeXtStage(
                depth=self.depths[ind],
                out_dim=self.out_dims[ind],
                stride=1 if ind == 0 else 2,
                layer_scale_init_value=self.layer_scale_init_value,
                eps=self.eps,
                grad_checkpoint=self.grad_checkpoint,
                dtype=self.dtype,
                )(output)

        pooled = global_avg_pool(output, axis=(-3, -2))
        pooled = nn.LayerNorm(
            epsilon=self.eps,
            dtype=self.dtype,
            )(pooled)

        if self.head_hidden_dim:
            pooled = nn.Dense(
                features=self.head_hidden_dim,
                dtype=self.dtype,
                )(pooled)
            pooled = gelu(pooled)

        return pooled
