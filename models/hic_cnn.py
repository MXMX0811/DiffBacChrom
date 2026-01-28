import einops
import torch
import torch.utils.checkpoint as cp


def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
    return torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
        dtype=dtype,
        device=device,
    )


class ResnetBlock(torch.nn.Module):
    def __init__(
        self, *, in_channels, out_channels=None, dtype=torch.float32, device=None
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.nin_shortcut = None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = map(
            lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(),
            (q, k, v),
        )
        hidden = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class CNNEncoder(torch.nn.Module):
    def __init__(
        self,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        in_channels=1,
        z_channels=16,
        dtype=torch.float32,
        gradient_checkpointing=True,
        device=None,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels,
            ch,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dtype=dtype,
                        device=device,
                    )
                )
                block_in = block_out
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, dtype=dtype, device=device)
            self.down.append(down)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        self.swish = torch.nn.SiLU(inplace=True)

    @torch.autocast("cuda", dtype=torch.float16)
    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                block = self.down[i_level].block[i_block]
                if self.gradient_checkpointing:
                    h = cp.checkpoint(block, hs[-1], use_reentrant=False)
                else:
                    h = block(hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                downsample = self.down[i_level].downsample
                if self.gradient_checkpointing:
                    hs.append(cp.checkpoint(downsample, hs[-1], use_reentrant=False))
                else:
                    hs.append(downsample(hs[-1]))
        # middle
        h = hs[-1]
        if self.gradient_checkpointing:
            h = cp.checkpoint(self.mid.block_1, h, use_reentrant=False)
            h = cp.checkpoint(self.mid.attn_1, h, use_reentrant=False)
            h = cp.checkpoint(self.mid.block_2, h, use_reentrant=False)
        else:
            h = self.mid.block_1(h)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h
