import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


# ------------------------------
# GroupNorm + Conv1d block
# ------------------------------
class NormConv1d(nn.Module):
    """GN + SiLU + Conv1d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=min(num_groups, in_channels),
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    
    def forward(self, x):
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x


# ------------------------------
# 1D ResNet Block
# ------------------------------
class ResBlock1D(nn.Module):
    """ResNet block for 1D sequences"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block1 = NormConv1d(in_channels, out_channels)
        self.block2 = NormConv1d(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        h = self.block1(x)
        h = self.block2(self.dropout(h))
        return h + self.skip(x)


# ------------------------------
# Down / Up sample (borrowed from sdvae1d)
# ------------------------------
class Downsample(nn.Module):
    """Strided conv with explicit padding to downsample by 2."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # pad on the right to keep even lengths well-behaved
        x = F.pad(x, (0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample followed by 3x3 conv."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ------------------------------
# Encoder
# ------------------------------
class StructureEncoder1D(nn.Module):
    """1D VAE Encoder"""
    def __init__(
        self,
        in_channels=16,
        hidden_channels=128,
        num_res_blocks=34,
        z_channels=4,
        dropout=0.0,
        use_downsample=False,
        ch_mult=(1, 2, 4),
    ):
        super().__init__()
        
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)

        self.use_downsample = use_downsample
        mults = ch_mult if use_downsample else (1,)
        self.ch_mult = mults

        # hierarchical blocks per resolution (borrow channel schedule from sdvae)
        self.down = nn.ModuleList()
        in_ch = hidden_channels
        for i_level, mult in enumerate(mults):
            block = nn.ModuleList()
            out_ch = hidden_channels * mult
            for _ in range(num_res_blocks):
                block.append(ResBlock1D(in_ch, out_ch, dropout))
                in_ch = out_ch
            level = nn.Module()
            level.block = block
            if use_downsample and i_level != len(mults) - 1:
                level.downsample = Downsample(in_ch)
            self.down.append(level)

        self.norm_out = nn.GroupNorm(min(32, in_ch), in_ch, eps=1e-6, affine=True)
        self.conv_out_mu = nn.Conv1d(in_ch, z_channels, 3, padding=1)
        self.conv_out_logvar = nn.Conv1d(in_ch, z_channels, 3, padding=1)
    
    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(len(self.down)):
            for blk in self.down[i_level].block:
                h = blk(h)
            if self.use_downsample and hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out_mu(h), self.conv_out_logvar(h)


# ------------------------------
# Decoder
# ------------------------------
class StructureDecoder1D(nn.Module):
    """1D VAE Decoder"""
    def __init__(
        self,
        out_channels=16,
        hidden_channels=128,
        num_res_blocks=34,
        z_channels=4,
        dropout=0.0,
        use_downsample=False,
        ch_mult=(1, 2, 4),
    ):
        super().__init__()
        
        mults = ch_mult if use_downsample else (1,)
        self.use_downsample = use_downsample
        self.ch_mult = mults

        deepest_ch = hidden_channels * mults[-1]
        self.conv_in = nn.Conv1d(z_channels, deepest_ch, 3, padding=1)
        
        self.up = nn.ModuleList()
        in_ch = deepest_ch
        for i_level in reversed(range(len(mults))):
            block = nn.ModuleList()
            out_ch = hidden_channels * mults[i_level]
            for _ in range(num_res_blocks):
                block.append(ResBlock1D(in_ch, out_ch, dropout))
                in_ch = out_ch
            level = nn.Module()
            level.block = block
            if use_downsample and i_level != 0:
                level.upsample = Upsample(in_ch)
            self.up.append(level)
        
        self.norm_out = nn.GroupNorm(min(32, in_ch), in_ch, eps=1e-6, affine=True)
        self.conv_out = nn.Conv1d(in_ch, out_channels, 3, padding=1)
    
    def forward(self, z):
        h = self.conv_in(z)
        for i_level in range(len(self.up)):
            for blk in self.up[i_level].block:
                h = blk(h)
            if self.use_downsample and hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


# ------------------------------
# Diagonal Gaussian for 1D latents
# ------------------------------
class DiagonalGaussian1D:
    """Diagonal Gaussian distribution for (B, W, C)"""
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mu + eps * std

    def mode(self):
        return self.mu


# ------------------------------
# AutoencoderKL1D
# ------------------------------
class StructureAutoencoderKL1D(nn.Module):
    """1D Autoencoder KL with optional downsampling."""
    def __init__(
        self,
        in_channels=16,
        hidden_channels=128,
        num_res_blocks=18,
        z_channels=16,
        dropout=0.0,
        use_downsample=False,
        ch_mult=(1, 2, 4),
    ):
        super().__init__()
        
        self.encoder = StructureEncoder1D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            dropout=dropout,
            use_downsample=use_downsample,
            ch_mult=ch_mult,
        )
        self.decoder = StructureDecoder1D(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            dropout=dropout,
            use_downsample=use_downsample,
            ch_mult=ch_mult,
        )
        self.z_channels = z_channels
        self.in_channels = in_channels
        self.use_downsample = use_downsample
        self.ch_mult = ch_mult

    def encode(self, x_seq):
        """Input: (B, W, C) -> return latent_dist, mu, logvar (all seq-last)"""
        x_cwt = x_seq.permute(0, 2, 1)           # (B, C, W)
        mu_cwt, logvar_cwt = self.encoder(x_cwt) # (B, zc, W or W/2)

        mu = mu_cwt.permute(0, 2, 1)             # (B, W, zc)
        logvar = logvar_cwt.permute(0, 2, 1)
        dist = DiagonalGaussian1D(mu, logvar)

        return SimpleNamespace(latent_dist=dist, mu=mu, logvar=logvar)

    def decode(self, z_seq):
        """Input: (B, W, C) -> (B, W, C)"""
        z_cwt = z_seq.permute(0, 2, 1)           # (B, C, W)
        x_cwt = self.decoder(z_cwt)
        return x_cwt.permute(0, 2, 1)

    def forward(self, x_seq, scale_latent=1.0, sample_posterior=True):
        enc = self.encode(x_seq)
        z = enc.latent_dist.sample() if sample_posterior else enc.latent_dist.mode()
        z = z * scale_latent
        x_recon = self.decode(z)
        return x_recon, enc.mu, enc.logvar


def kl_loss(mu, logvar):
    """KL divergence term"""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()
