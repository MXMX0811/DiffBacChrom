import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.vision_transformer import Attention, Mlp

from . import cnn
from .pos_embed import get_1d_sincos_pos_embed_from_grid


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                         Embedding Layers for Timesteps                        #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
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


class HiCEncoder8f(nn.Module):
    """
    Downsamples Hi-C with CNNEncoder (8x spatial reduction) before the transformer-based HiCEncoder.
    Keeps the same interface as HiCEncoder so it can be dropped in as a replacement.
    """
    def __init__(
        self,
        input_size: int,
        embed_dim: int = 512,
        vit_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        out_dim: int = 1152,
        dropout_prob: float = 0.1,
        use_learned_null: bool = True,
        cnn_in_channels: int = 1,
        cnn_ch: int = 128,
        cnn_ch_mult=(1, 2, 4, 4),
        cnn_num_res_blocks: int = 2,
        cnn_z_channels: int = 16,
    ):
        super().__init__()
        if input_size % 8 != 0:
            raise ValueError("HiCEncoder8f expects input_size divisible by 8 (three stride-2 downsamples).")

        self.input_size = input_size
        self.downsample_factor = 8
        reduced_size = input_size // self.downsample_factor

        self.cnn_encoder = cnn.CNNEncoder(
            ch=cnn_ch,
            ch_mult=cnn_ch_mult,
            num_res_blocks=cnn_num_res_blocks,
            in_channels=cnn_in_channels,
            z_channels=cnn_z_channels,
        )
        self.hic_encoder = HiCEncoder(
            input_size=reduced_size,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_dim=out_dim,
            dropout_prob=dropout_prob,
            use_learned_null=use_learned_null,
        )

        # expose common attributes so existing checks keep working
        self.proj_out = self.hic_encoder.proj_out
        self.use_learned_null = self.hic_encoder.use_learned_null
        self.null_cond = self.hic_encoder.null_cond

    def encode(self, H: torch.Tensor):
        H_down = self.cnn_encoder(H)
        return self.hic_encoder.encode(H_down)

    def forward(self, H: torch.Tensor, train: bool, force_drop_ids: torch.Tensor | None = None):
        H_down = self.cnn_encoder(H)
        return self.hic_encoder(H_down, train=train, force_drop_ids=force_drop_ids)

    @torch.no_grad()
    def null_tokens(self, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.hic_encoder.null_tokens(batch=batch, dtype=dtype, device=device)
    
    

