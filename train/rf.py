import torch


class RF:
    def __init__(self, model):
        self.model = model

    def forward(self, data_latent: torch.Tensor, hic: torch.Tensor):
        """
        data_latent: (B, W, C)
        hic: (B, 1, W, W)
        """
        b = data_latent.size(0)
        t = torch.rand((b,), device=data_latent.device)
        texp = t.view([b, *([1] * (data_latent.dim() - 1))])

        noise = torch.randn_like(data_latent)
        zt = (1 - texp) * data_latent + texp * noise

        vtheta = self.model(zt, t, hic)
        if isinstance(vtheta, tuple):
            vtheta = vtheta[0]
        B, C, T = vtheta.shape
        assert C % 2 == 0
        C_half = C // 2
        vtheta, _ = torch.split(vtheta, C_half, dim=1)
        vtheta = vtheta.permute(0, 2, 1)  # (B, W, C_half) to match data_latent shape

        batchwise_mse = ((noise - data_latent - vtheta) ** 2).mean(dim=list(range(1, len(data_latent.shape))))
        return batchwise_mse.mean()

    @torch.no_grad()
    def sample(self, hic: torch.Tensor, sample_steps: int, shape: torch.Size, cfg_scale: float = 1.0):
        """
        hic: (B, 1, W, W)
        shape: (B, W, C)
        """
        z = torch.randn(shape, device=hic.device)
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt_tensor = torch.tensor([dt] * b, device=hic.device).view([b, *([1] * (z.dim() - 1))])

        for i in range(sample_steps, 0, -1):
            t = torch.full((b,), i / sample_steps, device=hic.device)
            vc = self.model(z, t, hic, cfg_scale=cfg_scale)
            if isinstance(vc, tuple):
                vc = vc[0]
            B, C, T = vc.shape
            C_half = C // 2
            vc, _ = torch.split(vc, C_half, dim=1)
            vc = vc.permute(0, 2, 1)  # (B, W, C)
            z = z - dt_tensor * vc
        return z