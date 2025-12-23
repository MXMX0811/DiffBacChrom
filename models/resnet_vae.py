import torch.nn as nn
import torch.nn.functional as F
import torch
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
# Encoder
# ------------------------------
class StructureEncoder1D(nn.Module):
    """1D VAE Encoder"""
    def __init__(self, in_channels=16, hidden_channels=128, num_res_blocks=34, z_channels=4, dropout=0.0):
        super().__init__()
        
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResBlock1D(hidden_channels, hidden_channels, dropout))
        self.res_blocks = nn.ModuleList(blocks)
        
        self.norm_out = nn.GroupNorm(min(32, hidden_channels), hidden_channels, eps=1e-6, affine=True)
        self.conv_out_mu = nn.Conv1d(hidden_channels, z_channels, 3, padding=1)
        self.conv_out_logvar = nn.Conv1d(hidden_channels, z_channels, 3, padding=1)
    
    def forward(self, x):
        h = self.conv_in(x)
        for blk in self.res_blocks:
            h = blk(h)
        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out_mu(h), self.conv_out_logvar(h)


# ------------------------------
# Decoder
# ------------------------------
class StructureDecoder1D(nn.Module):
    """1D VAE Decoder"""
    def __init__(self, out_channels=16, hidden_channels=128, num_res_blocks=34, z_channels=4, dropout=0.0):
        super().__init__()
        
        self.conv_in = nn.Conv1d(z_channels, hidden_channels, 3, padding=1)
        
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResBlock1D(hidden_channels, hidden_channels, dropout))
        self.res_blocks = nn.ModuleList(blocks)
        
        self.norm_out = nn.GroupNorm(min(32, hidden_channels), hidden_channels, eps=1e-6, affine=True)
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, 3, padding=1)
    
    def forward(self, z):
        h = self.conv_in(z)
        for blk in self.res_blocks:
            h = blk(h)
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
    """1D Autoencoder KL"""
    def __init__(self, in_channels=16, hidden_channels=128, num_res_blocks=18, z_channels=16, dropout=0.0):
        super().__init__()
        
        self.encoder = StructureEncoder1D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            dropout=dropout,
        )
        self.decoder = StructureDecoder1D(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            dropout=dropout,
        )
        self.z_channels = z_channels
        self.in_channels = in_channels

    def encode(self, x_seq):
        """Input: (B, W, C) -> return latent_dist, mu, logvar (all seq-last)"""
        x_cwt = x_seq.permute(0, 2, 1)           # (B, C, W)
        mu_cwt, logvar_cwt = self.encoder(x_cwt) # (B, zc, W)

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
