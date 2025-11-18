# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


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
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
    - in_channels: usually 1 (single-channel Hi-C after normalization)
    - patch_size: unused in 1D bin encoder (kept for API compatibility)  # MODIFIED
    - embed_dim: internal ViT width
    - depth, num_heads: ViT depth/heads
    - out_dim: project to DiT hidden_size (e.g., 1152 for DiT-XL)
    Reuses the existing 1D sin-cos pos embedding functions over bins.       # MODIFIED
    """
    def __init__(
        self,
        input_size: int = 928,
        in_channels: int = 1,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        out_dim: int = 1152,
        dropout_prob: float = 0.1, 
        use_learned_null: bool = True, 
        use_upper_tri: bool = True,    # kept for compatibility, ignored # MODIFIED
    ):
        super().__init__()
        self.input_size = input_size           # W
        self.in_channels = in_channels         # MODIFIED
        self.embed_dim = embed_dim             # MODIFIED
        self.use_upper_tri = use_upper_tri     # MODIFIED: no longer used internally

        # MODIFIED: instead of 2D PatchEmbed, use a 1D "row" embedder over bins
        # For each bin i, take the Hi-C row H[:, :, i, :] (contacts to all bins, length W)
        # and project it to embed_dim as a token.
        self.row_embed = nn.Linear(input_size, embed_dim, bias=True)      # MODIFIED

        # sequence length for cond tokens is exactly W (one token per bin)
        s_cond = input_size                                                # MODIFIED

        # Will use fixed 1D sin-cos embedding over bins:
        self.pos_embed = nn.Parameter(torch.zeros(1, s_cond, embed_dim), requires_grad=False)  # MODIFIED
        
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.proj_out = nn.Linear(embed_dim, out_dim, bias=True)  # map to DiT hidden size

        # dropout for condition tokens
        # learnable null cond token
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
        # MODIFIED: Initialize and freeze pos_embed with 1D sin-cos over bin indices [0..W-1]
        grid = np.arange(self.input_size, dtype=np.float32)  # (W,)          # MODIFIED
        pos = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid)  # MODIFIED
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))    # MODIFIED

        # MODIFIED: initialize row_embed like a Linear layer
        w = self.row_embed.weight.data                                         # MODIFIED
        nn.init.xavier_uniform_(w)                                             # MODIFIED
        nn.init.constant_(self.row_embed.bias, 0)                              # MODIFIED

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
        Output: cond_tokens (B, W, out_dim), one token per bin.          # MODIFIED
        """
        B, C, W, W2 = H.shape                                              # MODIFIED
        assert W == W2 == self.input_size, "Hi-C must be square W x W"     # MODIFIED

        # MODIFIED: collapse channel dimension (usually C=1) into a single Hi-C map per sample
        H_mean = H.mean(dim=1)                                            # (B, W, W)  # MODIFIED

        # MODIFIED: for each bin i, use its Hi-C row as input features
        # H_mean[:, i, :] is the contact profile of bin i to all bins (length W)
        # row_embed maps (W,) -> (embed_dim,)
        # We apply row_embed along the last dimension.
        x = self.row_embed(H_mean)                                        # (B, W, embed_dim)  # MODIFIED

        pos = self.pos_embed.to(x.dtype)                                  # (1, W, embed_dim)  # MODIFIED
        x = x + pos                                                       # (B, W, embed_dim)  # MODIFIED

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.proj_out(x)                    # (B, W, out_dim)         # MODIFIED
        return x

    def forward(self, H: torch.Tensor, train: bool, force_drop_ids: torch.Tensor | None = None):
        """
        For training with condition dropout.
        """
        cond = self.encode(H)   # (B, W, out_dim)                          # MODIFIED
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
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    DiT block with:
      - Self-attention
      - Cross-attention (optional)
      - MLP
      - adaLN-zero modulation
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_cross_attn: bool = True, **block_kwargs):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True,    # crucial: allow (B, T, C)
        )

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)

        # adaLN-zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size)  # 9 groups needed now
        )

    def forward(self, x, c, cond_tokens):
        """
        x: (B, T_latent, D)
        c: (B, D)               # timestep embedding
        cond_tokens: (B, S_cond, D)  # Hi-C encoder tokens
        """
        # slice adaLN params
        (
            shift_self, scale_self, gate_self,
            shift_cross, scale_cross, gate_cross,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)

        x = x + gate_self.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_self, scale_self))

        if self.use_cross_attn:
            q = modulate(self.norm_cross(x), shift_cross, scale_cross)
            cross_out, _ = self.cross_attn(q, cond_tokens, cond_tokens, need_weights=False)
            x = x + gate_cross.unsqueeze(1) * cross_out

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # MODIFIED: output dimension changed from patch_size * patch_size * out_channels to out_channels
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)  # MODIFIED
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=928,  # MODIFIED: sequence length (should match HiC W, e.g., 928)
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cross_attn_interval: int = 4,   # use cross-attn every N layers
        learn_sigma=True,
        gradient_checkpointing=True,   # allow gradient checkpointing to save memory
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # MODIFIED: use Linear as 1D sequence embedder instead of 2D PatchEmbed
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)  # MODIFIED
        self.seq_len = input_size  # MODIFIED: interpret input_size as sequence length
        num_patches = self.seq_len  # MODIFIED

        self.t_embedder = TimestepEmbedder(hidden_size)
        # MODIFIED: pass input_size from DiT to HiC encoder so W is shared
        self.hic_encoder = HiCEncoder(input_size=input_size, out_dim=hidden_size)  # MODIFIED
        assert self.hic_encoder.proj_out.out_features == hidden_size

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_cross = (cross_attn_interval is not None) and ((i + 1) % cross_attn_interval == 0)
            self.blocks.append(
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_cross_attn=use_cross)
            )
            
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # MODIFIED: Initialize (and freeze) pos_embed by 1D sin-cos embedding over sequence length
        pos = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.seq_len, dtype=np.float32))  # MODIFIED
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))  # MODIFIED

        # MODIFIED: Initialize x_embedder like nn.Linear
        w = self.x_embedder.weight.data  # MODIFIED
        nn.init.xavier_uniform_(w)       # MODIFIED
        nn.init.constant_(self.x_embedder.bias, 0)  # MODIFIED

        # Initialize hic_encoder:
        if getattr(self.hic_encoder, "use_learned_null", False):
            with torch.no_grad():
                nn.init.normal_(self.hic_encoder.null_cond, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, C_out)
        imgs: (N, C_out, T) sequence outputs (channels-first)
        """  # MODIFIED_COMMENT: docstring updated for sequence output
        # MODIFIED: for sequence, simply return (N, C, T)
        c = self.out_channels  # MODIFIED
        imgs = x.permute(0, 2, 1)  # (N, C, T)  # MODIFIED
        return imgs  # MODIFIED

    def forward(self, x, t, H):
        """
        Forward pass of DiT.
        x: (N, T, C_in) tensor of sequence inputs (e.g., per-token latent representations)
        t: (N,) tensor of diffusion timesteps
        H: tensor hic matrices
        """  # MODIFIED_COMMENT: updated x description to sequence format
        cond_tokens = self.hic_encoder(H, train=self.training)
        y = cond_tokens.mean(dim=1)  
        
        # MODIFIED: expect x as (N, T, C) and use Linear embedder
        x = self.x_embedder(x)  # (N, T, D)  # MODIFIED
        x = x + self.pos_embed.to(x.dtype)  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        c = t + y
        
        for block in self.blocks:
            if self.gradient_checkpointing:
                x = cp.checkpoint(block, x, c, cond_tokens, use_reentrant=False)
            else:
                x = block(x, c, cond_tokens)          # cross-attn injects condition each layer
                
        x = self.final_layer(x, c)                # (N, T, out_channels)  # MODIFIED
        x = self.unpatchify(x)                   # (N, out_channels, T)   # MODIFIED
        return x

    def forward_with_cfg(self, x, t, H, cfg_scale):
        """
        CFG forward pass of DiT inference on sequence inputs.
        """  # MODIFIED_COMMENT: clarify it's for sequence inputs
        N = x.shape[0]
        assert N % 2 == 0
        half = N // 2

        x_half = x[:half]
        t_half = t[:half]
        H_half = H[:half]

        combined_x = torch.cat([x_half, x_half], dim=0)
        combined_t = torch.cat([t_half, t_half], dim=0)

        cond_half = self.hic_encoder.encode(H_half)
        uncond_half = self.hic_encoder.null_tokens(half, dtype=cond_half.dtype, device=cond_half.device)

        y_cond = cond_half.mean(dim=1)                 # (half, D)
        y_uncond = torch.zeros_like(y_cond)
        
        combined_cond = torch.cat([cond_half, uncond_half], dim=0)

        # MODIFIED: expect combined_x as (N, T, C) and use Linear embedder
        x = self.x_embedder(combined_x)  # (N, T, D)  # MODIFIED
        x = x + self.pos_embed.to(x.dtype)
        t = self.t_embedder(combined_t)
        c = torch.cat([t[:half] + y_cond, t[half:] + y_uncond], dim=0)

        for block in self.blocks:
            x = block(x, c, combined_cond)

        model_out = self.final_layer(x, c)      # (N, T, out_channels)  # MODIFIED
        model_out = self.unpatchify(model_out)  # (N, out_channels, T)  # MODIFIED
        
        cond_out, uncond_out = torch.split(model_out, half, dim=0)       # (N/2, out_ch, T)  # MODIFIED

        guided_half = uncond_out + cfg_scale * (cond_out - uncond_out)   # (N/2, out_ch, T)  # MODIFIED

        out = torch.cat([guided_half, guided_half], dim=0)               # (N, out_ch, T)    # MODIFIED
        return out


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

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

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
