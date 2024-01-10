"""
Transformer model, with support for vision and text.
"""


from typing import Callable, Optional

import jax
from flax import linen as nn
from flax.linen.dtypes import Array, Dtype
from jax import numpy as jnp

from .basic_layers import MLP, MultiHeadAttention, gelu, global_avg_pool


class TransformerBlock(nn.Module):
    """
    Transformer block.

    Attributes:
        n_heads: Number of heads for attention.
        expansion_factor: Expansion factor for the hidden layer of the MLP.
        act: Activation function applied in the hidden layer of the MLP.
        attention_bias: Whether the QKV and projection linear layers in the
            attention module should contain bias terms.
        mlp_bias: Whether the linear layers in the MLP should contain bias
            terms.
        eps: Epsilon value passed to the layer normalization modules.
        dtype: The data type of the computations.
    """
    n_heads: int
    expansion_factor: float = 4.
    act: Callable = gelu
    attention_bias: bool = True
    mlp_bias: bool = True
    eps: float = 1e-5
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input: Array, mask: Optional[Array] = None) -> Array:
        """
        Passes the input through the transformer block.

        Args:
            input: Input passed through the transformer block.
            mask: Optional mask for multi-headed self-attention.

        Returns:
            Output of the transformer block.
        """
        residual = input
        output = nn.LayerNorm(
            epsilon=self.eps,
            dtype=self.dtype,
            )(input)
        output = MultiHeadAttention(
            num_heads=self.n_heads,
            use_bias=self.attention_bias,
            dtype=self.dtype,
            )(output, mask=mask)
        output = residual+output

        residual = output
        output = nn.LayerNorm(
            epsilon=self.eps,
            dtype=self.dtype,
            )(output)
        output = MLP(
            expansion_factor=self.expansion_factor,
            act=self.act,
            bias=self.mlp_bias,
            dtype=self.dtype,
            )(output)
        output = residual+output

        return output


class Transformer(nn.Module):
    """
    Transformer.

    Attributes:
        depth: Number of transformer block.
        n_heads: Number of heads for attention.
        expansion_factor: Expansion factor for the hidden layer of the MLPs.
        act: Activation function applied in the hidden layer of the MLPs.
        attention_bias: Whether the QKV and projection linear layers in the
            attention modules should contain bias terms.
        mlp_bias: Whether the linear layers in the MLPs should contain bias
            terms.
        eps: Epsilon value passed to the layer normalization modules.
        grad_checkpoint: Whether to perform gradient checkpointing on the
            transformer blocks. If True, intermediate activations are not stored
            and are recomputed during backpropagation.
        dtype: The data type of the computations.
    """
    depth: int
    n_heads: int
    expansion_factor: float = 4.
    act: Callable = gelu
    attention_bias: bool = True
    mlp_bias: bool = True
    eps: float = 1e-5
    grad_checkpoint: bool = False
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input: Array, mask: Optional[Array] = None) -> Array:
        """
        Passes the input through the transformer.

        Args:
            input: Input passed through the transformer.
            mask: Optional mask for multi-headed self-attention.

        Returns:
            Output of the transformer.
        """
        block = nn.remat(TransformerBlock) if self.grad_checkpoint else TransformerBlock
        for _ in range(self.depth):
            input = block(
                n_heads=self.n_heads,
                expansion_factor=self.expansion_factor,
                act=self.act,
                attention_bias=self.attention_bias,
                mlp_bias=self.mlp_bias,
                eps=self.eps,
                dtype=self.dtype,
                )(input, mask=mask)
        return input


class PatchEmbed(nn.Module):
    """
    Patch embedding for vision transformers.

    Attributes:
        embed_dim: Embedding dimension.
        patch_size: Patch size.
        bias: Whether the projection transformation should contain bias terms.
        eps: Epsilon value passed to a layer normalization module applied
            prior to the linear transformation.
            If None, no layer normalization is applied.
        norm_first: Whether to apply layer normalization before or after
            the projection transformation if eps is not None.
        flatten: Whether to flatten the embeddings before returning.
        dtype: The data type of the computations.
    """
    embed_dim: int
    patch_size: int = 16
    bias: bool = True
    eps: Optional[float] = None
    norm_first: bool = False
    flatten: bool = True
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input: Array) -> Array:
        """
        Extracts patch embeddings from the input.

        Args:
            input: Input to extract patch embeddings from.

        Returns:
            Flattened patch embeddings.
        """
        layer_norm = nn.LayerNorm(self.eps) if self.eps else lambda x: x

        if self.norm_first:
            input = layer_norm(input)

        output = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding='valid',
            use_bias=self.bias,
            dtype=self.dtype,
            )(input)

        if not self.norm_first:
            output = layer_norm(output)

        if self.flatten:
            output = jnp.reshape(output, (len(output), -1, self.embed_dim))

        return output


class ClsToken(nn.Module):
    """
    Concatenates a class token to the beginning of the input.
    """
    @nn.compact
    def __call__(self, input: Array) -> Array:
        """
        Concatenates a class token to the beginning of the input.

        Args:
            input: Input to concatenate a class token to.

        Returns:
            Input with a class token concatenated to it.
        """
        embed_dim = input.shape[-1]
        cls_token = self.param(
            name='cls_token',
            init_fn=lambda prng: (embed_dim ** -0.5) * jax.random.normal(prng, (embed_dim,), dtype=input.dtype),
            )
        cls_token = jnp.broadcast_to(cls_token, (len(input), 1, embed_dim))
        return jnp.concatenate([cls_token, input], axis=1)


class PosEmbed(nn.Module):
    """
    Adds position embedding vectors to the input.
    """
    @nn.compact
    def __call__(self, input: Array) -> Array:
        """
        Adds position embedding vectors to the input.

        Args:
            input: Input to add position embedding vectors to.

        Returns:
            Input with position embedding vectors added to it.
        """
        shape = input.shape[-2:]
        pos_embed = self.param(
            name='pos_embed',
            init_fn=lambda prng: (shape[-1] ** -0.5) * jax.random.normal(prng, shape, dtype=input.dtype),
            )
        return pos_embed+input


class VisionTransformer(nn.Module):
    """
    Vision transformer.

    Attributes:
        depth: Number of transformer block.
        embed_dim: Embedding dimension.
        n_heads: Number of heads for attention.
        patch_size: Patch size.
        expansion_factor: Expansion factor for the hidden layer of the MLPs.
        act: Activation function applied in the hidden layer of the MLPs.
        pre_norm: Whether to place a layer normalization module immediately
            before the transformer blocks.
        attention_bias: Whether the QKV and projection linear layers in the
            attention modules should contain bias terms.
        mlp_bias: Whether the linear layers in the MLPs should contain bias
            terms.
        eps: Epsilon value passed to the layer normalization modules.
        global_pool: Whether to globally average pool the tokens before
            returning. If False, only the class token is returned.
        grad_checkpoint: Whether to perform gradient checkpointing on the
            transformer blocks. If True, intermediate activations are not stored
            and are recomputed during backpropagation.
        dtype: The data type of the computations.
    """
    depth: int
    embed_dim: int
    n_heads: int
    patch_size: int = 16
    expansion_factor: float = 4.
    act: Callable = gelu
    pre_norm: bool = True
    attention_bias: bool = True
    mlp_bias: bool = True
    eps: float = 1e-5
    global_pool: bool = False
    grad_checkpoint: bool = False
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input: Array) -> Array:
        """
        Passes the input through the vision transformer.

        Args:
            input: Input passed through the vision transformer.

        Returns:
            Output of the vision transformer.
        """
        output = PatchEmbed(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            bias=not self.pre_norm,
            dtype=self.dtype,
            )(input)
        output = ClsToken()(output)
        output = PosEmbed()(output)

        if self.pre_norm:
            output = nn.LayerNorm(
                epsilon=self.eps,
                dtype=self.dtype,
                )(output)

        output = Transformer(
            depth=self.depth,
            n_heads=self.n_heads,
            expansion_factor=self.expansion_factor,
            act=self.act,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            eps=self.eps,
            grad_checkpoint=self.grad_checkpoint,
            dtype=self.dtype,
            )(output)

        if self.global_pool:
            pooled = global_avg_pool(output)

        else:
            pooled = output[:, 0]

        pooled = nn.LayerNorm(
                epsilon=self.eps,
                dtype=self.dtype,
                )(pooled)

        return pooled


class TextTransformer(nn.Module):
    """
    Text transformer.

    Attributes:
        depth: Number of transformer block.
        embed_dim: Embedding dimension.
        n_heads: Number of heads for attention.
        vocab_size: Size of vocabulary.
        expansion_factor: Expansion factor for the hidden layer of the MLPs.
        act: Activation function applied in the hidden layer of the MLPs.
        attention_bias: Whether the QKV and projection linear layers in the
            attention modules should contain bias terms.
        mlp_bias: Whether the linear layers in the MLPs should contain bias
            terms.
        eps: Epsilon value passed to the layer normalization modules.
        global_pool: Whether to globally average pool the tokens before
            returning. If False, only the end-of-stream token is returned.
        grad_checkpoint: Whether to perform gradient checkpointing on the
            transformer blocks. If True, intermediate activations are not stored
            and are recomputed during backpropagation.
        dtype: The data type of the computations.
    """
    depth: int
    embed_dim: int
    n_heads: int
    vocab_size: int = 49408
    expansion_factor: float = 4.
    act: Callable = gelu
    attention_bias: bool = True
    mlp_bias: bool = True
    eps: float = 1e-5
    global_pool: bool = False
    grad_checkpoint: bool = False
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input: Array) -> Array:
        """
        Passes the input through the text transformer.

        Args:
            input: Input passed through the text transformer.

        Returns:
            Output of the text transformer.
        """
        output = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            dtype=self.dtype,
            )(input)
        output = PosEmbed()(output)

        output = Transformer(
            depth=self.depth,
            n_heads=self.n_heads,
            expansion_factor=self.expansion_factor,
            act=self.act,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            eps=self.eps,
            grad_checkpoint=self.grad_checkpoint,
            dtype=self.dtype,
            )(output, mask=nn.make_causal_mask(input, dtype=input.dtype))

        # Unlike ViTs, layer normalization is applied before extracting tokens.
        output = nn.LayerNorm(
                epsilon=self.eps,
                dtype=self.dtype,
                )(output)

        if self.global_pool:
            pooled = global_avg_pool(output)

        else:
            # The end-of-stream token is always the highest.
            pooled = output[jnp.arange(len(input)), jnp.argmax(input, axis=1)]

        return pooled
