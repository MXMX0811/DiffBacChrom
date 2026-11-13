"""CrossDiT with bidirectional Mamba-2 in place of self-attention."""

from functools import partial

import torch.nn as nn

from .crossdit import DiTBlock, DiT_models as CrossDiT_models


class BidirectionalMamba2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Only Mamba experiments require the official CUDA/Triton dependency.
        from mamba_ssm import Mamba2

        # Official defaults: d_state=128, d_conv=4, expand=2, headdim=64.
        self.forward_mamba = Mamba2(d_model=hidden_size)
        self.backward_mamba = Mamba2(d_model=hidden_size)

    def forward(self, x):
        # Each call scans the whole sequence with fresh state, including RF steps.
        forward = self.forward_mamba(x)
        backward = self.backward_mamba(x.flip(1)).flip(1)
        return (forward + backward) * 0.5


class MambaDiTBlock(DiTBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(hidden_size, num_heads, mlp_ratio,
                         self_attn=BidirectionalMamba2(hidden_size))


# Reuse the same S/B/L/XL widths, depths and cross-attention heads.
DiT_models = {
    name: partial(model_fn, block_type=MambaDiTBlock)
    for name, model_fn in CrossDiT_models.items()
}
