import torch


class RF:
    def __init__(self, model):
        self.model = model

    def forward(self, data_latent: torch.Tensor, hic: torch.Tensor, return_trace: bool = False):
        """RF loss for latents (B, T, C); optionally collect JointAttDiT traces."""
        b = data_latent.size(0)
        t = torch.rand((b,), device=data_latent.device)
        texp = t.view(b, 1, 1)
        noise = torch.randn_like(data_latent)
        zt = (1 - texp) * data_latent + texp * noise

        if return_trace:
            prediction, cond_trace = self.model(zt, t, hic, return_trace=True)
        else:
            prediction = self.model(zt, t, hic)
        # Existing heads output 2C channels. RF uses the first C as velocity;
        # preserve the second half and checkpoint shapes without training it.
        velocity, _ = prediction.chunk(2, dim=1)
        velocity = velocity.permute(0, 2, 1)
        loss = ((noise - data_latent - velocity) ** 2).mean(dim=(1, 2)).mean()
        return (loss, cond_trace) if return_trace else loss

    @torch.no_grad()
    def sample(self, hic: torch.Tensor, sample_steps: int, shape: torch.Size, cfg_scale: float = 1.0):
        """Sample B latents (B, T, C), using temporary 2B predictions for CFG."""
        z = torch.randn(shape, device=hic.device)
        b = z.size(0)
        dt = 1.0 / sample_steps
        use_cfg = cfg_scale != 1.0
        if use_cfg:
            # Diffusers 0.33.1 convention: unconditional first, conditional second.
            model_hic = torch.cat([hic, hic], dim=0)
            force_drop_ids = torch.cat([
                torch.ones(b, device=hic.device, dtype=torch.bool),
                torch.zeros(b, device=hic.device, dtype=torch.bool),
            ])

        for i in range(sample_steps, 0, -1):
            t = torch.full((b,), i / sample_steps, device=hic.device)
            if use_cfg:
                prediction = self.model(
                    torch.cat([z, z], dim=0), torch.cat([t, t], dim=0),
                    model_hic, force_drop_ids=force_drop_ids,
                )
                uncond, cond = prediction.chunk(2, dim=0)
                prediction = uncond + cfg_scale * (cond - uncond)
            else:
                prediction = self.model(z, t, hic)
            velocity, _ = prediction.chunk(2, dim=1)
            z = z - dt * velocity.permute(0, 2, 1)
        return z
