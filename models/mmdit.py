import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.vision_transformer import Attention, Mlp
from .pos_embed import get_1d_sincos_pos_embed_from_grid
from .hic_encoder import HiCEncoder8f


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
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    DiT block with:
      - Self-attention
      - Joint attention with condition tokens (optional, MM-DiT style)
      - Modality-specific MLPs
      - adaLN-zero modulation
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        # Joint attention (MM-DiT style) over latent + condition tokens
        self.norm_joint_x = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_joint_c = nn.LayerNorm(hidden_size, eps=1e-6)
        self.joint_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        # Modality-specific MLPs
        self.norm2_x = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2_c = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)
        self.mlp_cond = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)

        # adaLN-zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # groups: self(x) 3, joint(x) 3, joint(c) 3, mlp(x) 3, mlp(c) 3 -> 15
            nn.Linear(hidden_size, 15 * hidden_size)
        )

    def forward(self, x, t, cond_tokens):
        """
        x: (B, T_latent, D)
        c: (B, D)               # timestep embedding + global cond
        cond_tokens: (B, S_cond, D)  # Hi-C encoder tokens (updated in-place when joint attention is used)
        """
        # slice adaLN params
        (
            shift_self, scale_self, gate_self,
            shift_joint_x, scale_joint_x, gate_joint_x,
            shift_joint_t, scale_joint_t, gate_joint_t,
            shift_mlp_x, scale_mlp_x, gate_mlp_x,
            shift_mlp_t, scale_mlp_t, gate_mlp_t,
        ) = self.adaLN_modulation(t).chunk(15, dim=1)

        x = x + gate_self.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_self, scale_self))

        # Joint attention over concatenated latent and condition tokens
        x_mod = modulate(self.norm_joint_x(x), shift_joint_x, scale_joint_x)
        cond_mod = modulate(self.norm_joint_c(cond_tokens), shift_joint_t, scale_joint_t)
        joint = torch.cat([x_mod, cond_mod], dim=1)
        joint_out = self.joint_attn(joint)
        x_out, cond_out = joint_out.split([x.shape[1], cond_tokens.shape[1]], dim=1)
        x = x + gate_joint_x.unsqueeze(1) * x_out
        cond_tokens = cond_tokens + gate_joint_t.unsqueeze(1) * cond_out
        
        # For condition entanglement analysis, return updated cond_tokens
        # Used in visualize_cond_trace.py
        cond_post_attn = cond_tokens.detach().cpu()

        # Per-modality MLP updates
        x = x + gate_mlp_x.unsqueeze(1) * self.mlp(modulate(self.norm2_x(x), shift_mlp_x, scale_mlp_x))
        cond_tokens = cond_tokens + gate_mlp_t.unsqueeze(1) * self.mlp_cond(modulate(self.norm2_c(cond_tokens), shift_mlp_t, scale_mlp_t))

        return x, cond_tokens, cond_post_attn


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
        learn_sigma=True,
        gradient_checkpointing=True,   # allow gradient checkpointing to save memory
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads

        # 1D sequence embedder
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.seq_len = input_size
        num_patches = self.seq_len  # sequence length

        self.t_embedder = TimestepEmbedder(hidden_size)
        # pass input_size from DiT to HiC encoder so W is shared
        self.hic_encoder = HiCEncoder8f(
            input_size=input_size, 
            out_dim=hidden_size, 
            embed_dim=hidden_size // 2, 
            vit_depth=4,
            num_heads=num_heads,
        )
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

    def forward(self, x, t, H, cfg_scale: float = 1.0):
        """
        Unified forward for SD3-style MM-DiT.

        Args:
            x: (N, T, C_in)
            t: (N,)
            H: (N, C_hic, W, W)
            cfg_scale: guidance scale (1.0 = no CFG)

        Returns:
            (N, out_channels, T)
        """
        # For tracing condition tokens
        cond_trace = []
        
        # --------------------------------------------------
        # Case 1: training OR no guidance
        # --------------------------------------------------
        if self.training or cfg_scale == 1.0:
            # Hi-C tokens (with dropout during training)
            cond_tokens = self.hic_encoder(H, train=self.training)
            
            cond_trace.append(cond_tokens.detach().cpu())

            # latent embedding
            x = self.x_embedder(x)
            x = x + self.pos_embed.to(x.dtype)

            # timestep embedding
            t_emb = self.t_embedder(t)

            # joint MM-DiT blocks
            for block in self.blocks:
                if self.gradient_checkpointing and self.training:
                    x, cond_tokens, cond_post_attn = cp.checkpoint(
                        block, x, t_emb, cond_tokens, use_reentrant=False
                    )
                else:
                    x, cond_tokens, cond_post_attn = block(x, t_emb, cond_tokens)
                cond_trace.append(cond_post_attn)

            x = self.final_layer(x, t_emb)
            return self.unpatchify(x), cond_trace

        # --------------------------------------------------
        # Case 2: CFG inference (SD3-style joint MM-DiT)
        # --------------------------------------------------
        N = x.shape[0]
        assert N % 2 == 0, "CFG requires even batch size"
        half = N // 2

        # split inputs
        x_half = x[:half]
        t_half = t[:half]
        H_half = H[:half]

        # duplicate latents and timesteps
        x = torch.cat([x_half, x_half], dim=0)
        t = torch.cat([t_half, t_half], dim=0)

        # conditional / unconditional Hi-C tokens
        cond_tokens = self.hic_encoder.encode(H_half)
        uncond_tokens = self.hic_encoder.null_tokens(
            half, dtype=cond_tokens.dtype, device=cond_tokens.device
        )
        cond_tokens = torch.cat([cond_tokens, uncond_tokens], dim=0)
        
        cond_trace.append(cond_tokens.detach().cpu())

        # latent embedding
        x = self.x_embedder(x)
        x = x + self.pos_embed.to(x.dtype)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # joint MM-DiT blocks (cond_tokens WILL be updated)
        for block in self.blocks:
            x, cond_tokens, cond_post_attn = block(x, t_emb, cond_tokens)
            cond_trace.append(cond_post_attn)

        # output
        x = self.final_layer(x, t_emb)
        x = self.unpatchify(x)

        # split cond / uncond outputs
        x_cond, x_uncond = torch.split(x, half, dim=0)

        # classifier-free guidance
        x_guided = x_uncond + cfg_scale * (x_cond - x_uncond)

        # return full batch (SD-style sampler compatibility)
        return torch.cat([x_guided, x_guided], dim=0), cond_trace


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
