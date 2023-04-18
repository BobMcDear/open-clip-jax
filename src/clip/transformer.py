"""
Transformer model, with support for vision and text.
"""


from typing import Any, Callable

import jax
from flax import linen as nn
from jax import numpy as jnp
from jax._src.numpy.lax_numpy import _ScalarMeta

from .basic_layers import MLP, MultiHeadAttention, global_avg_pool


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
    act: Callable = nn.gelu
    attention_bias: bool = True
    mlp_bias: bool = True
    eps: float = 1e-5
    dtype: _ScalarMeta = jnp.float32

    @nn.compact
    def __call__(self, input: Any) -> Any:
        residual = input
        output = nn.LayerNorm(
            epsilon=self.eps,
            dtype=self.dtype,
            )(input)
        output = MultiHeadAttention(
            num_heads=self.n_heads,
            use_bias=self.attention_bias,
            dtype=self.dtype,
            )(output)
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
    Transformer model.

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
        dtype: The data type of the computations.
    """
    depth: int
    n_heads: int
    expansion_factor: float = 4.
    act: Callable = nn.gelu
    attention_bias: bool = True
    mlp_bias: bool = True
    eps: float = 1e-5
    dtype: _ScalarMeta = jnp.float32

    @nn.compact
    def __call__(self, input: Any) -> Any:
        for _ in range(self.depth):
            input = TransformerBlock(
                n_heads=self.n_heads,
                expansion_factor=self.expansion_factor,
                act=self.act,
                attention_bias=self.attention_bias,
                mlp_bias=self.mlp_bias,
                eps=self.eps,
                dtype=self.dtype,
                )(input)
        return input


class PatchEmbed(nn.Module):
    """
    Patch embedding for vision transformers.

    Attributes:
        embed_dim: Embedding dimension.
        patch_size: Patch size.
        bias: Whether the projection transformation should contain bias terms.
        dtype: The data type of the computations.
    """
    embed_dim: int
    patch_size: int = 16
    bias: bool = True
    dtype: _ScalarMeta = jnp.float32

    @nn.compact
    def __call__(self, input: Any) -> Any:
        output = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding='valid',
            use_bias=self.bias,
            dtype=self.dtype,
            )(input)
        output = jnp.reshape(output, (len(output), -1, self.embed_dim))
        return output


class ClsToken(nn.Module):
    """
    Concatenates a class token to the beginning of the input.
    """
    @nn.compact
    def __call__(self, input: Any) -> Any:
        shape = (1, input.shape[-1])
        cls_token = self.param(
            name='cls_token',
            init_fn=lambda prng: jax.random.normal(prng, shape, dtype=input.dtype),
            )
        cls_token = jnp.broadcast_to(cls_token, (len(input), 1, shape[-1]))
        return jnp.concatenate([cls_token, input], axis=1)


class PosEmbed(nn.Module):
    """
    Adds position embedding vectors to the input.
    """
    @nn.compact
    def __call__(self, input: Any) -> Any:
        shape = (1, *input.shape[-2:])
        pos_embed = self.param(
            name='pos_embed',
            init_fn=lambda prng: jax.random.normal(prng, shape, dtype=input.dtype),
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
        dtype: The data type of the computations.
    """
    depth: int
    embed_dim: int
    n_heads: int
    patch_size: int = 16
    expansion_factor: float = 4.
    act: Callable = nn.gelu
    pre_norm: bool = True
    attention_bias: bool = True
    mlp_bias: bool = True
    eps: float = 1e-5
    dtype: _ScalarMeta = jnp.float32

    @nn.compact
    def __call__(self, input: Any) -> Any:
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
            dtype=self.dtype,
            )(output)

        output = global_avg_pool(output)
        output = nn.LayerNorm(
                epsilon=self.eps,
                dtype=self.dtype,
                )(output)

        return output


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
        dtype: The data type of the computations.
    """
    depth: int
    embed_dim: int
    n_heads: int
    vocab_size: int = 50304
    expansion_factor: float = 4.
    act: Callable = nn.gelu
    attention_bias: bool = True
    mlp_bias: bool = True
    eps: float = 1e-5
    dtype: _ScalarMeta = jnp.float32

    @nn.compact
    def __call__(self, input: Any) -> Any:
        output = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            dtype=self.dtype,
            )(input)
        output = ClsToken()(output)
        output = PosEmbed()(output)

        output = Transformer(
            depth=self.depth,
            n_heads=self.n_heads,
            expansion_factor=self.expansion_factor,
            act=self.act,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            eps=self.eps,
            dtype=self.dtype,
            )(output)

        output = global_avg_pool(output)
        output = nn.LayerNorm(
                epsilon=self.eps,
                dtype=self.dtype,
                )(output)

        return output
