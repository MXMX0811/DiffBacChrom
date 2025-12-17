import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    q, k, v: (B, L, D)
    returns: (B, L, D)
    """
    bsz, seq_len, dim = q.shape
    head_dim = dim // num_heads
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(bsz, seq_len, dim)
    return out


#################################################################################
#                         Embedding Layers for Timesteps                        #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t: torch.Tensor, dtype: torch.dtype | None = None, **kwargs):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if dtype is not None:
            t_freq = t_freq.to(dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                   HiC Encoder                                 #
#################################################################################


class EncoderBlock(nn.Module):
    """Transformer block used in HiC encoder."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HiCEncoder(nn.Module):
    """
    ViT-based encoder for square Hi-C matrices.
    - input_size: W (H=W)
    - embed_dim: internal ViT width
    - depth, num_heads: ViT depth/heads
    - out_dim: project to DiT hidden_size (e.g., 1152 for DiT-XL)
    Reuses the existing 1D sin-cos pos embedding functions over bins.
    """

    def __init__(
        self,
        input_size: int,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        out_dim: int = 1152,
        dropout_prob: float = 0.1,
        use_learned_null: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim

        # For each bin i, take the Hi-C row H[:, :, i, :] (contacts to all bins, length W)
        # and project it to embed_dim as a token.
        self.row_embed = nn.Linear(input_size, embed_dim, bias=True)

        # sequence length for cond tokens is exactly W (one token per bin)
        s_cond = input_size

        # fixed 1D sin-cos embedding over bins:
        self.pos_embed = nn.Parameter(torch.zeros(1, s_cond, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.proj_out = nn.Linear(embed_dim, out_dim, bias=True)  # map to DiT hidden size

        # dropout for condition tokens
        self.dropout_prob = float(dropout_prob)
        self.use_dropout = self.dropout_prob > 0
        self.use_learned_null = bool(use_learned_null)
        if self.use_learned_null:
            # (1, S_cond, out_dim)
            self.null_cond = nn.Parameter(torch.zeros(1, s_cond, out_dim))
        else:
            self.register_buffer("null_cond", torch.zeros(1, s_cond, out_dim), persistent=False)

        self._init_weights()

    def _init_weights(self):
        # Initialize and freeze pos_embed with 1D sin-cos over bin indices [0..W-1]
        grid = np.arange(self.input_size, dtype=np.float32)  # (W,)
        pos = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # initialize row_embed like a Linear layer
        w = self.row_embed.weight.data
        nn.init.xavier_uniform_(w)
        nn.init.constant_(self.row_embed.bias, 0)

    def token_drop(self, batch_size: int, device: torch.device, force_drop_ids: torch.Tensor | None = None):
        if force_drop_ids is None:
            drop_ids = torch.rand(batch_size, device=device) < self.dropout_prob
        else:
            drop_ids = (force_drop_ids == 1)
        return drop_ids

    def encode(self, H: torch.Tensor):
        """
        Encoding without dropout (for inference).
        input: H (B, C, W, W)
        Output: cond_tokens (B, W, out_dim), one token per bin.
        """
        B, C, W, W2 = H.shape
        assert W == W2 == self.input_size, "Hi-C must be square W x W"

        # collapse channel dimension (usually C=1) into a single Hi-C map per sample
        H_mean = H.mean(dim=1)  # (B, W, W)

        # for each bin i, use its Hi-C row as input features
        x = self.row_embed(H_mean)  # (B, W, embed_dim)

        pos = self.pos_embed.to(x.dtype)  # (1, W, embed_dim)
        x = x + pos  # (B, W, embed_dim)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.proj_out(x)  # (B, W, out_dim)
        return x

    def forward(self, H: torch.Tensor, train: bool, force_drop_ids: torch.Tensor | None = None):
        """
        For training with condition dropout.
        """
        cond = self.encode(H)   # (B, W, out_dim)
        B = cond.shape[0]
        if (train and self.use_dropout) or (force_drop_ids is not None):
            drop_ids = self.token_drop(B, cond.device, force_drop_ids)  # (B,)
            if drop_ids.any():
                null = self.null_cond.to(cond.dtype).expand(B, -1, -1)
                cond = torch.where(drop_ids.view(B, 1, 1), null, cond)
        return cond

    @torch.no_grad()
    def null_tokens(self, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.null_cond.to(device=device, dtype=dtype).expand(batch, -1, -1)


#################################################################################
#                                   Positional Embeds                           #
#################################################################################


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scaling_factor=None, offset=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


#################################################################################
#                                Sequence Embedding                             #
#################################################################################


class PatchEmbed(nn.Module):
    """1D sequence embedding (no spatial patching)."""

    def __init__(self, seq_len: int, in_chans: int, embed_dim: int, dtype=None, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = seq_len
        self.patch_size = (1,)  # kept for compatibility
        self.proj = nn.Linear(in_chans, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        # x: (B, T, C)
        assert x.dim() == 3, f"Expected (B, T, C) input, got {x.shape}"
        return self.proj(x)


#################################################################################
#                                 Core DiT Blocks                               #
#################################################################################


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]


def optimized_attention(qkv, num_heads):
    return scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], num_heads)


class SelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        pre_only: bool = False,
        qk_norm: Optional[str] = None,
        rmsnorm: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if not pre_only:
            self.proj = nn.Linear(dim, dim, dtype=dtype, device=device)
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(
                self.head_dim,
                elementwise_affine=True,
                eps=1.0e-6,
                dtype=dtype,
                device=device,
            )
            self.ln_k = RMSNorm(
                self.head_dim,
                elementwise_affine=True,
                eps=1.0e-6,
                dtype=dtype,
                device=device,
            )
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(
                self.head_dim,
                elementwise_affine=True,
                eps=1.0e-6,
                dtype=dtype,
                device=device,
            )
            self.ln_k = nn.LayerNorm(
                self.head_dim,
                elementwise_affine=True,
                eps=1.0e-6,
                dtype=dtype,
                device=device,
            )
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor):
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (q, k, v) = self.pre_attention(x)
        x = scaled_dot_product_attention(q, k, v, self.num_heads)
        x = self.post_attention(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class DismantledBlock(nn.Module):
    """A DiT block with gated adaptive layer norm (adaLN) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        x_block_self_attn: bool = False,
        dtype=None,
        device=None,
        **block_kwargs,
    ):
        super().__init__()
        if not rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size,
                elementwise_affine=False,
                eps=1e-6,
                dtype=dtype,
                device=device,
            )
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pre_only=pre_only,
            qk_norm=qk_norm,
            rmsnorm=rmsnorm,
            dtype=dtype,
            device=device,
        )
        if x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            self.x_block_self_attn = True
            self.attn2 = SelfAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                pre_only=False,
                qk_norm=qk_norm,
                rmsnorm=rmsnorm,
                dtype=dtype,
                device=device,
            )
        else:
            self.x_block_self_attn = False
        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(
                    hidden_size,
                elementwise_affine=False,
                eps=1e-6,
                dtype=dtype,
                device=device,
            )
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                )
            else:
                self.mlp = SwiGLUFeedForward(
                    dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256
                )
        self.scale_mod_only = scale_mod_only
        if x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            n_mods = 9
        elif not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device
            ),
        )
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
        assert x is not None, "pre_attention called with None input"
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.adaLN_modulation(c).chunk(6, dim=1)
                )
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                    c
                ).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

    def pre_attention_x(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert self.x_block_self_attn
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        x_norm = self.norm1(x)
        qkv = self.attn.pre_attention(modulate(x_norm, shift_msa, scale_msa))
        qkv2 = self.attn2.pre_attention(modulate(x_norm, shift_msa2, scale_msa2))
        return (
            qkv,
            qkv2,
            (
                x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                gate_msa2,
            ),
        )

    def post_attention_x(
        self,
        attn,
        attn2,
        x,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        gate_msa2,
        attn1_dropout: float = 0.0,
    ):
        assert not self.pre_only
        if attn1_dropout > 0.0:
            # Use torch.bernoulli to implement dropout, only dropout the batch dimension
            attn1_dropout = torch.bernoulli(
                torch.full((attn.size(0), 1, 1), 1 - attn1_dropout, device=attn.device)
            )
            attn_ = (
                gate_msa.unsqueeze(1) * self.attn.post_attention(attn) * attn1_dropout
            )
        else:
            attn_ = gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + attn_
        attn2_ = gate_msa2.unsqueeze(1) * self.attn2.post_attention(attn2)
        x = x + attn2_
        mlp_ = gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        x = x + mlp_
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        if self.x_block_self_attn:
            (q, k, v), (q2, k2, v2), intermediates = self.pre_attention_x(x, c)
            attn = scaled_dot_product_attention(q, k, v, self.attn.num_heads)
            attn2 = scaled_dot_product_attention(q2, k2, v2, self.attn2.num_heads)
            return self.post_attention_x(attn, attn2, *intermediates)
        else:
            (q, k, v), intermediates = self.pre_attention(x, c)
            attn = scaled_dot_product_attention(q, k, v, self.attn.num_heads)
            return self.post_attention(attn, *intermediates)


def block_mixing(context, x, context_block, x_block, c):
    assert context is not None, "block_mixing called with None context"
    context_qkv, context_intermediates = context_block.pre_attention(context, c)

    if x_block.x_block_self_attn:
        x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
    else:
        x_qkv, x_intermediates = x_block.pre_attention(x, c)

    q, k, v = tuple(
        torch.cat(tuple(qkv[i] for qkv in [context_qkv, x_qkv]), dim=1)
        for i in range(3)
    )
    attn = scaled_dot_product_attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = (
        attn[:, : context_qkv[0].shape[1]],
        attn[:, context_qkv[0].shape[1] :],
    )

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        # keep a tensor output to make checkpointing stable
        context = x_attn[:, :0]

    if x_block.x_block_self_attn:
        x_q2, x_k2, x_v2 = x_qkv2
        attn2 = scaled_dot_product_attention(x_q2, x_k2, x_v2, x_block.attn2.num_heads)
        x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
    else:
        x = x_block.post_attention(x_attn, *x_intermediates)

    return context, x


class JointBlock(nn.Module):
    """just a small wrapper to serve as a fsdp unit"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        qk_norm = kwargs.pop("qk_norm", None)
        x_block_self_attn = kwargs.pop("x_block_self_attn", False)
        self.context_block = DismantledBlock(
            *args, pre_only=pre_only, qk_norm=qk_norm, **kwargs
        )
        self.x_block = DismantledBlock(
            *args,
            pre_only=False,
            qk_norm=qk_norm,
            x_block_self_attn=x_block_self_attn,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        return block_mixing(
            *args, context_block=self.context_block, x_block=self.x_block, **kwargs
        )


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        total_out_channels: Optional[int] = None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.linear = (
            nn.Linear(
                hidden_size, out_channels, bias=True, dtype=dtype, device=device
            )
            if (total_out_channels is None)
            else nn.Linear(
                hidden_size, total_out_channels, bias=True, dtype=dtype, device=device
            )
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device
            ),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MMDiTX(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        input_size: int,
        in_channels: int = 4,
        hidden_size: Optional[int] = 1152,
        depth: int = 28,
        num_heads: Optional[int] = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        adm_in_channels: Optional[int] = None,
        context_embedder_config: Optional[Dict] = None,
        register_length: int = 0,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
        hic_embed_dim: Optional[int] = None,
        hic_depth: int = 8,
        hic_num_heads: Optional[int] = None,
        hic_mlp_ratio: float = 4.0,
        hic_dropout_prob: float = 0.1,
        hic_use_learned_null: bool = True,
        x_block_self_attn_layers: Optional[List[int]] = None,
        qkv_bias: bool = True,
        qk_norm: Optional[str] = None,
        dtype=None,
        device=None,
        verbose: bool = False,
    ):
        super().__init__()
        if verbose:
            print(
                f"mmditx initializing with: {input_size=}, {in_channels=}, {hidden_size=}, {depth=}, {mlp_ratio=}, {learn_sigma=}, {adm_in_channels=}, {context_embedder_config=}, {register_length=}, {rmsnorm=}, {scale_mod_only=}, {swiglu=}, {out_channels=}, {pos_embed_scaling_factor=}, {pos_embed_offset=}, {pos_embed_max_size=}, {hic_embed_dim=}, {hic_depth=}, {hic_num_heads=}, {hic_mlp_ratio=}, {hic_dropout_prob=}, {hic_use_learned_null=}, {qk_norm=}, {qkv_bias=}, {dtype=}, {device=}"
            )
        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size
        self.x_block_self_attn_layers = x_block_self_attn_layers or []

        # use mmdit defaults by default; fall back to legacy scaling when explicit None is passed
        if hidden_size is None:
            hidden_size = 64 * depth
        if num_heads is None:
            num_heads = depth

        self.num_heads = num_heads
        self.seq_len = input_size

        self.x_embedder = PatchEmbed(
            input_size,
            in_channels,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device)

        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = nn.Linear(adm_in_channels, hidden_size, bias=True, dtype=dtype, device=device)
        else:
            self.y_embedder = None

        self.context_embedder = nn.Identity()
        if context_embedder_config is not None:
            if context_embedder_config["target"] == "torch.nn.Linear":
                self.context_embedder = nn.Linear(
                    **context_embedder_config["params"], dtype=dtype, device=device
                )

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(
                torch.randn(1, register_length, hidden_size, dtype=dtype, device=device)
            )
        else:
            self.register = None

        pos_len = self.pos_embed_max_size if self.pos_embed_max_size is not None else input_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_len, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )

        # Hi-C encoder (one token per bin)
        hic_embed_dim = hidden_size if hic_embed_dim is None else hic_embed_dim
        hic_num_heads = num_heads if hic_num_heads is None else hic_num_heads
        self.hic_encoder = HiCEncoder(
            input_size=input_size,
            embed_dim=hic_embed_dim,
            depth=hic_depth,
            num_heads=hic_num_heads,
            mlp_ratio=hic_mlp_ratio,
            out_dim=hidden_size,
            dropout_prob=hic_dropout_prob,
            use_learned_null=hic_use_learned_null,
        )

        self.joint_blocks = nn.ModuleList(
            [
                JointBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    pre_only=i == depth - 1,
                    rmsnorm=rmsnorm,
                    scale_mod_only=scale_mod_only,
                    swiglu=swiglu,
                    qk_norm=qk_norm,
                    x_block_self_attn=(i in self.x_block_self_attn_layers),
                    dtype=dtype,
                    device=device,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size, self.out_channels, dtype=dtype, device=device
        )

        self.initialize_weights()

    def get_pos_embed(self, seq_len: int) -> torch.Tensor:
        if self.pos_embed.shape[1] == seq_len:
            return self.pos_embed
        if self.pos_embed.shape[1] > seq_len:
            start = (self.pos_embed.shape[1] - seq_len) // 2
            return self.pos_embed[:, start : start + seq_len, :]
        raise ValueError(f"pos_embed length {self.pos_embed.shape[1]} smaller than requested {seq_len}")

    def unpatchify(self, x, seq_len: Optional[int] = None):
        """
        x: (N, T, C_out)
        imgs: (N, C_out, T)
        """
        imgs = x.permute(0, 2, 1)  # (N, C, T)
        if seq_len is not None:
            imgs = imgs[..., :seq_len]
        return imgs

    def forward_core_with_concat(
        self,
        x: torch.Tensor,
        c_mod: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        skip_layers: Optional[List[int]] = None,
        controlnet_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if skip_layers is None:
            skip_layers = []
        if self.register_length > 0 and self.register is not None:
            context = torch.cat(
                (
                    repeat(self.register, "1 ... -> b ...", b=x.shape[0]),
                    context if context is not None else torch.Tensor([]).type_as(x),
                ),
                1,
            )

        for i, block in enumerate(self.joint_blocks):
            if i in skip_layers:
                continue
            context, x = block(context, x, c=c_mod)
            if controlnet_hidden_states is not None:
                controlnet_block_interval = len(self.joint_blocks) // len(
                    controlnet_hidden_states
                )
                x = x + controlnet_hidden_states[i // controlnet_block_interval]

        x = self.final_layer(x, c_mod)  # (N, T, out_channels)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        H: torch.Tensor,
        cfg_scale: float = 1.0,
        return_full_batch: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        controlnet_hidden_states: Optional[torch.Tensor] = None,
        skip_layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT for 1D sequences with Hi-C conditioning.
        x: (N, T, C_in) latent sequence
        t: (N,) diffusion timesteps
        H: (N, C_hic, W, W) Hi-C matrices
        """
        seq_len = x.shape[1]
        pos = self.get_pos_embed(seq_len).to(dtype=x.dtype, device=x.device)

        if (cfg_scale == 1.0) or self.training:
            cond_tokens = self.hic_encoder(H, train=self.training, force_drop_ids=force_drop_ids)

            x = self.x_embedder(x)
            x = x + pos
            t_emb = self.t_embedder(t, dtype=x.dtype)
            if y is not None and self.y_embedder is not None:
                y_emb = self.y_embedder(y)
                t_emb = t_emb + y_emb

            context_tokens = self.context_embedder(cond_tokens if context is None else context)

            x = self.forward_core_with_concat(x, t_emb, context_tokens, skip_layers, controlnet_hidden_states)
            x = self.unpatchify(x, seq_len=seq_len)
            return x

        # CFG inference path
        N = x.shape[0]
        assert N % 2 == 0, "CFG requires even batch size"
        half = N // 2

        x_half = x[:half]
        t_half = t[:half]
        H_half = H[:half]

        combined_x = torch.cat([x_half, x_half], dim=0)
        combined_t = torch.cat([t_half, t_half], dim=0)

        cond_half = self.hic_encoder.encode(H_half)
        uncond_half = self.hic_encoder.null_tokens(
            half, dtype=cond_half.dtype, device=cond_half.device
        )
        combined_cond = torch.cat([cond_half, uncond_half], dim=0)

        x = self.x_embedder(combined_x)
        x = x + pos
        t_emb = self.t_embedder(combined_t, dtype=x.dtype)
        if y is not None and self.y_embedder is not None:
            y_cond, y_uncond = torch.split(y, half, dim=0)
            y_emb = torch.cat([y_cond, y_uncond], dim=0)
            t_emb = t_emb + self.y_embedder(y_emb)

        context_tokens = self.context_embedder(combined_cond if context is None else context)

        x = self.forward_core_with_concat(x, t_emb, context_tokens, skip_layers, controlnet_hidden_states)
        x = self.unpatchify(x, seq_len=seq_len)

        cond_out, uncond_out = torch.split(x, half, dim=0)
        guided = uncond_out + cfg_scale * (cond_out - uncond_out)

        if return_full_batch:
            return torch.cat([guided, guided], dim=0)
        return guided

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by 1D sin-cos embedding
        grid = np.arange(self.pos_embed.shape[1], dtype=np.float32)
        if self.pos_embed_scaling_factor is not None:
            grid = grid / self.pos_embed_scaling_factor
        if self.pos_embed_offset is not None:
            grid = grid - self.pos_embed_offset
        pos = get_1d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1], grid
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # Initialize x_embedder like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize hic_encoder null token:
        if getattr(self.hic_encoder, "use_learned_null", False):
            with torch.no_grad():
                nn.init.normal_(self.hic_encoder.null_cond, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.context_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL(**kwargs):
    return MMDiTX(depth=28, hidden_size=1152, num_heads=16, **kwargs)


def DiT_L(**kwargs):
    return MMDiTX(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def DiT_B(**kwargs):
    return MMDiTX(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_S(**kwargs):
    return MMDiTX(depth=12, hidden_size=384, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L':  DiT_L,
    'DiT-B':  DiT_B,
    'DiT-S':  DiT_S,
}
