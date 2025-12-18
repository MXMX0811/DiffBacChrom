import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.vision_transformer import Attention, Mlp
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


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    DiT block with:
      - Self-attention
      - Cross-attention condition
      - Global condition (optional)
      - MLP
      - adaLN-zero modulation
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True,    # allow (B, T, C)
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
        c: (B, D)               # timestep embedding + global cond
        cond_tokens: (B, S_cond, D)  # Hi-C encoder tokens
        """
        # slice adaLN params
        (
            shift_self, scale_self, gate_self,
            shift_cross, scale_cross, gate_cross,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)

        x = x + gate_self.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_self, scale_self))

        q = modulate(self.norm_cross(x), shift_cross, scale_cross)
        cross_out, _ = self.cross_attn(q, cond_tokens, cond_tokens, need_weights=False)
        x = x + gate_cross.unsqueeze(1) * cross_out

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
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
    Diffusion model with a Transformer backbone for 1D sequences.
    """
    def __init__(
        self,
        input_size: int,       # sequence length (should match HiC W)
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        use_global_cond: bool = True,
        learn_sigma: bool = True,
        gradient_checkpointing: bool = True,   # allow gradient checkpointing to save memory
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.use_global_cond = use_global_cond

        # 1D sequence embedder
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.seq_len = input_size
        num_patches = self.seq_len  # sequence length

        self.t_embedder = TimestepEmbedder(hidden_size)
        # pass input_size from DiT to HiC encoder so W is shared
        self.hic_encoder = HiCEncoder(input_size=input_size, out_dim=hidden_size, embed_dim=hidden_size, num_heads=num_heads)
        assert self.hic_encoder.proj_out.out_features == hidden_size

        # fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            )
            
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by 1D sin-cos embedding over sequence length
        pos = get_1d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1], np.arange(self.seq_len, dtype=np.float32)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # Initialize x_embedder like nn.Linear
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w)
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize hic_encoder null token:
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
        """
        imgs = x.permute(0, 2, 1)  # (N, C, T)
        return imgs

    def forward(self, x, t, H, cfg_scale: float = 1.0, return_full_batch: bool = True):
        """
        If cfg_scale == 1.0 or self.training: normal forward.
        If cfg_scale != 1.0 and not self.training: do CFG by batch-doubling.
        return_full_batch controls whether to return N or N/2 samples in CFG mode.
        """
        if (cfg_scale == 1.0) or self.training:
            # Normal path (training or no guidance)
            cond_tokens = self.hic_encoder(H, train=self.training)

            x = self.x_embedder(x)
            x = x + self.pos_embed.to(x.dtype)
            t_emb = self.t_embedder(t)

            if self.use_global_cond:
                t_emb = t_emb + cond_tokens.mean(dim=1)

            for block in self.blocks:
                if self.gradient_checkpointing:
                    x = cp.checkpoint(block, x, t_emb, cond_tokens, use_reentrant=False)
                else:
                    x = block(x, t_emb, cond_tokens)

            x = self.final_layer(x, t_emb)
            x = self.unpatchify(x)
            return x

        # CFG path (inference only)
        N = x.shape[0]
        assert N % 2 == 0
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
        x = x + self.pos_embed.to(x.dtype)
        t_emb = self.t_embedder(combined_t)

        if self.use_global_cond:
            y_cond = cond_half.mean(dim=1)
            y_uncond = uncond_half.mean(dim=1)
            t_emb = torch.cat([t_emb[:half] + y_cond, t_emb[half:] + y_uncond], dim=0)

        for block in self.blocks:
            # 也可以在 CFG path 里用 checkpoint，但会更慢
            x = block(x, t_emb, combined_cond)

        model_out = self.final_layer(x, t_emb)
        model_out = self.unpatchify(model_out)

        cond_out, uncond_out = torch.split(model_out, half, dim=0)
        guided = uncond_out + cfg_scale * (cond_out - uncond_out)

        if return_full_batch:
            return torch.cat([guided, guided], dim=0)
        return guided


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L':  DiT_L,
    'DiT-B':  DiT_B,
    'DiT-S':  DiT_S,
}
