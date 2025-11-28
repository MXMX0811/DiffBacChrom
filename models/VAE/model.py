import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# GroupNorm + Conv1d block
# ------------------------------
class NormConv1d(nn.Module):
    """
    GN + SiLU + Conv1d, ConvBlock in ResNet Block.
    Input:  (B, C_in, T)
    Output:  (B, C_out, T)
    """
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
# 1D ResNet Block（without downsampling）
# ------------------------------
class ResBlock1D(nn.Module):
    """
    ResNet block for 1D sequences.
    in:  (B, C_in, T)
    out: (B, C_out, T)
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block1 = NormConv1d(in_channels, out_channels)
        self.block2 = NormConv1d(out_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        h = self.block1(x)
        h = self.block2(self.dropout(h))
        return h + self.skip(x)


# ==============================
# Encoder：without changing length
# ==============================
class StructureEncoder1D(nn.Module):
    """
    VAE 1D Encoder（SD-VAE-FT-MSE style，T=W）。
    Input:  x_seq  (B, C_in, W)
    Output:  mu, logvar  (B, z_channels, W)
    """
    def __init__(
        self,
        in_channels: int = 16,      # [x1,y1,z1,m1,x2,y2,z2,m2] for 2 beads in each hic bin
        hidden_channels: int = 128,
        num_res_blocks: int = 4,
        z_channels: int = 4,      # latent channel, corresponding to DiT.in_channels
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        blocks = []
        in_ch = hidden_channels
        for _ in range(num_res_blocks):
            blocks.append(ResBlock1D(in_ch, hidden_channels, dropout=dropout))
            in_ch = hidden_channels
        self.res_blocks = nn.ModuleList(blocks)
        
        self.norm_out = nn.GroupNorm(
            num_groups=min(32, hidden_channels),
            num_channels=hidden_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv_out_mu = nn.Conv1d(hidden_channels, z_channels, kernel_size=3, padding=1)
        self.conv_out_logvar = nn.Conv1d(hidden_channels, z_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        x: (B, C_in, W)
        return:
          mu:     (B, z_channels, W)
          logvar: (B, z_channels, W)
        """
        h = self.conv_in(x)
        
        for blk in self.res_blocks:
            h = blk(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        
        mu = self.conv_out_mu(h)
        logvar = self.conv_out_logvar(h)
        return mu, logvar


# ==============================
# Decoder：without changing length
# ==============================
class StructureDecoder1D(nn.Module):
    """
    VAE 1D Decoder
    Input:  z  (B, z_channels, W)
    Output:  x_recon (B, C_out, W)
    """
    def __init__(
        self,
        out_channels: int = 16,     # [x1,y1,z1,m1,x2,y2,z2,m2] for 2 beads in each hic bin
        hidden_channels: int = 128,
        num_res_blocks: int = 4,
        z_channels: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.conv_in = nn.Conv1d(z_channels, hidden_channels, kernel_size=3, padding=1)
        
        blocks = []
        in_ch = hidden_channels
        for _ in range(num_res_blocks):
            blocks.append(ResBlock1D(in_ch, hidden_channels, dropout=dropout))
            in_ch = hidden_channels
        self.res_blocks = nn.ModuleList(blocks)
        
        self.norm_out = nn.GroupNorm(
            num_groups=min(32, hidden_channels),
            num_channels=hidden_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, z):
        """
        z: (B, z_channels, W)
        return:
          x_recon: (B, C_out, W)
        """
        h = self.conv_in(z)
        
        for blk in self.res_blocks:
            h = blk(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        x_recon = self.conv_out(h)
        return x_recon


# ==============================
# AutoencoderKL1D
# ==============================
class StructureAutoencoderKL1D(nn.Module):
    """
      - per hic bin（seq len = W = hic_index number）
      - Input:  (B, W, 16)  [x1,y1,z1,m1,x2,y2,z2,m2] for 2 beads in each hic bin
      - latent: (B, W, z_channels) corresponding to HiCEncoder (B, W, D) in DiT
      - recon: (B, W, 8)
    Conv use (B, C, T) format
    """
    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 128,
        num_res_blocks: int = 4,
        z_channels: int = 16,
        dropout: float = 0.0,
    ):
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
        """
        x_seq: (B, W, C_in)   # C_in=8: [x1,y1,z1,m1,x2,y2,z2,m2]
        return:
          z_seq:   (B, W, z_channels)   # for DiT
          mu_seq:  (B, W, z_channels)
          logvar_seq: (B, W, z_channels)
        """
        # to (B, C_in, W)
        x_cwt = x_seq.permute(0, 2, 1)
        mu_cwt, logvar_cwt = self.encoder(x_cwt)  # (B, z_channels, W)
        
        # reparameterization trick: z = mu + eps * sigma
        std = torch.exp(0.5 * logvar_cwt)
        eps = torch.randn_like(std)
        z_cwt = mu_cwt + eps * std               # (B, z_channels, W)

        # 转回 (B, W, C)
        z_seq = z_cwt.permute(0, 2, 1)
        mu_seq = mu_cwt.permute(0, 2, 1)
        logvar_seq = logvar_cwt.permute(0, 2, 1)
        return z_seq, mu_seq, logvar_seq

    def decode(self, z_seq):
        """
        z_seq: (B, W, z_channels)
        return:
          x_recon_seq: (B, W, C_in)
        """
        z_cwt = z_seq.permute(0, 2, 1)         # (B, z_channels, W)
        x_recon_cwt = self.decoder(z_cwt)       # (B, C_in, W)
        x_recon_seq = x_recon_cwt.permute(0, 2, 1)
        return x_recon_seq

    def forward(self, x_seq):
        """
        x_seq: (B, W, C_in)
        return:
          x_recon_seq: (B, W, C_in)
          mu_seq, logvar_seq: (B, W, z_channels)
        """
        z_seq, mu_seq, logvar_seq = self.encode(x_seq)
        x_recon_seq = self.decode(z_seq)
        return x_recon_seq, mu_seq, logvar_seq


def kl_loss(mu, logvar):
    """
    mu, logvar: (B, C, T)
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())   # (B, C, T)
    return kl.mean()