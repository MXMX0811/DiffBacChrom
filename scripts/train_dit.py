import argparse
import os
from functools import partial
from typing import Dict, List

import pandas as pd
import numpy as np 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffbacchrom.crossdit import DiT_models as CrossDiT_models
from diffbacchrom.mmdit import DiT_models as MMDiT_models
from diffbacchrom.vae import StructureAutoencoderKL1D
from scripts.dataloader import HiCStructureDataset, collate_fn


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
            B, C, T = vc.shape
            C_half = C // 2
            vc, _ = torch.split(vc, C_half, dim=1)
            vc = vc.permute(0, 2, 1)  # (B, W, C)
            z = z - dt_tensor * vc
        return z


def rebuild_structure_tables(
    struct_denorm: torch.Tensor,
    sample_ids: List[str],
    structure_files: List[str],
    struct_lookup: Dict[str, str],
    output_dir: str,
):
    """
    Rebuild TSVs matching the original format: for each hic_index two rows (bead1, bead2).
    struct_denorm: (B, W, 16) on CPU
    struct_lookup: kept for API compatibility (unused)
    """
    os.makedirs(output_dir, exist_ok=True)
    columns = ["hic_index", "x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"]

    for b_idx, sid in enumerate(sample_ids):
        tokens = struct_denorm[b_idx].cpu().numpy()  # (W,16)
        rows: List[List[float]] = []
        for hic_idx, token in enumerate(tokens):
            row1_vals = token[:8].tolist()
            row2_vals = token[8:].tolist()
            rows.append([hic_idx] + row1_vals)
            rows.append([hic_idx] + row2_vals)

        df = pd.DataFrame(rows, columns=columns)
        out_path = os.path.join(output_dir, f"{sid}_recon.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        print(f"Saved reconstruction to {out_path}")


def apply_mask_threshold(struct: torch.Tensor) -> torch.Tensor:
    """
    Binarize mask channels (>0.5 -> 1, otherwise 0) and zero coordinates where mask is 0.
    Expects last dimension ordered as (x, y, z, mask) repeated 4 times.
    """
    struct_view = struct.reshape(*struct.shape[:-1], 4, 4)
    masks = (struct_view[..., 3] > 0.5).float()
    struct_view[..., 3] = masks
    struct_view[..., 0:3] = struct_view[..., 0:3] * masks.unsqueeze(-1)
    return struct_view.reshape(struct.shape)

def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Return parameter count; include frozen params when trainable_only=False."""
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def format_params(n: int) -> str:
    return f"{n:,} ({n / 1e6:.2f}M)"

from torch.optim.lr_scheduler import LambdaLR

def get_scheduler(opt, warmup_steps=1000, total_steps=10000):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    return LambdaLR(opt, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data/train")
    parser.add_argument("--hic_dirname", type=str, default="Hi-C")
    parser.add_argument("--struct_dirname", type=str, default="structure")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_scale", type=float, default=1.335256, help="Latent scale used during training")
    parser.add_argument("--sample_steps", type=int, default=50, help="RF sampling steps")
    parser.add_argument("--model", type=str, default="CrossDiT", choices=["CrossDiT", "MMDiT"], help="Select backbone model")
    parser.add_argument(
        "--size",
        type=lambda s: s.upper(),
        default="L",
        choices=["S", "B", "L", "XL"],
        help="DiT model size (S/B/L/XL)",
    )
    parser.add_argument("--use_global_cond", type=bool, default=True, help="Whether CrossDiT uses global conditioning (CrossDiT only)")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Classifier-free guidance scale for inference")
    parser.add_argument("--grad_cp", type=bool, default=True, help="Use gradient checkpointing to save memory")
    parser.add_argument("--save_dir", type=str, default=None, help="Checkpoint directory (default: checkpoints/dit/<model>)")
    parser.add_argument("--vae_ckpt", type=str, default="checkpoints/vae/epoch_040.pt")
    parser.add_argument("--run_name", type=str, default="rf_dit_structure")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for cosine scheduler (set 0 to keep constant learning rate)",
    )
    args = parser.parse_args()

    if args.cfg_scale is None:
        args.cfg_scale = 1.5 if args.model == "MMDiT" else 1.0
    if args.model != "CrossDiT":
        if args.use_global_cond:
            print("Warning: --use_global_cond is only supported when model=CrossDiT; disabled.")
            args.use_global_cond = False

    if args.save_dir is None:
        args.save_dir = os.path.join("checkpoints", "dit", args.model + "-" + args.size)
        
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = HiCStructureDataset(
        root_dir=args.root_dir,
        hic_dirname=args.hic_dirname,
        struct_dirname=args.struct_dirname,
    )

    # Infer sequence length W from first sample (structure and Hi-C should match)
    first_hic_path, first_struct_path = dataset.samples[0]
    seq_len_struct = dataset._load_structure_seq(first_struct_path).shape[0]
    seq_len_hic = dataset._load_hic_matrix(first_hic_path).shape[-1]
    if seq_len_struct != seq_len_hic:
        raise ValueError(f"Mismatch between structure length ({seq_len_struct}) and Hi-C size ({seq_len_hic})")
    seq_len = seq_len_struct
    print(f"Inferred sequence length: W={seq_len}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=partial(collate_fn, train=True),
    )

    # lookup for structure file paths so we can restore column names
    struct_lookup = {os.path.basename(s_path): s_path for _, s_path in dataset.samples}

    vae = StructureAutoencoderKL1D().to(device)
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    dit_size_key = f"DiT-{args.size}"
    if args.model == "CrossDiT":
        model_fn = CrossDiT_models[dit_size_key]
        model_kwargs = {
            "input_size": seq_len,
            "in_channels": vae.z_channels,
            "use_global_cond": args.use_global_cond,
            "gradient_checkpointing": args.grad_cp,
        }
    elif args.model == "MMDiT":
        model_fn = MMDiT_models[dit_size_key]
        model_kwargs = {
            "input_size": seq_len,
            "in_channels": vae.z_channels,
            "gradient_checkpointing": args.grad_cp,
        }
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model = model_fn(**model_kwargs).to(device)

    # report parameter counts before training
    vae_params = count_params(vae, trainable_only=False)
    hic_params = count_params(model.hic_encoder, trainable_only=False)
    dit_backbone_params = sum(
        p.numel() for name, p in model.named_parameters() if "hic_encoder" not in name
    )
    print("Parameter counts (all params):")
    print(f"  VAE: {format_params(vae_params)}")
    print(f"  HiC encoder: {format_params(hic_params)}")
    print(f"  DiT backbone: {format_params(dit_backbone_params)}")
    
    '''
    For 1D-ResNet18 VAE and DiT-L
    Parameter counts (all params):
        VAE: 3,598,128 (3.60M)
        HiC encoder: 27,646,464 (27.65M)
        DiT backbone: 634,246,176 (634.25M)
    '''

    rf = RF(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    total_steps = args.epochs * len(dataloader)
    scheduler = None

    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be non-negative")
    if args.warmup_steps > 0:
        scheduler = get_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)

    wandb.init(project="rf_dit_structure", name=args.run_name)

    for epoch in range(args.epochs):
        rf.model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            hic = batch["hic"].to(device)  # (B,1,W,W)
            structure = batch["structure"].to(device)  # (B,W,16)

            with torch.no_grad():
                z = vae.encode(structure).latent_dist.sample().mul_(args.latent_scale)  # (B,W,C)

            optimizer.zero_grad()
            loss = rf.forward(z, hic)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rf.model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.6f}")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

        # ---------- inference sample ----------
        rf.model.eval()
        with torch.no_grad():
            vis_loader = DataLoader(
                dataset,
                batch_size=10,
                shuffle=True,
                num_workers=0,
                collate_fn=partial(collate_fn, train=False),
            )
            batch = next(iter(vis_loader))
            hic = batch["hic"].to(device)
            sample_ids = batch["sample_id"]
            structure_files = batch["structure_file"]

            seq_len = hic.shape[-1]
            
            sample_latent = rf.sample(
                hic, 
                sample_steps=args.sample_steps, 
                shape=(hic.shape[0], seq_len, vae.z_channels), 
                cfg_scale=args.cfg_scale
            )
            decoded = vae.decode(sample_latent / args.latent_scale)  # (B,W,16) in normalized space
            decoded = apply_mask_threshold(decoded)
            decoded_cpu = decoded.cpu()

            rebuild_structure_tables(
                decoded_cpu,
                sample_ids,
                structure_files,
                struct_lookup,
                output_dir=os.path.join(args.save_dir, f"samples_epoch{epoch+1:03d}"),
            )

        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pt")
        torch.save({"epoch": epoch + 1, "model": rf.model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
        wandb.save(ckpt_path)

    torch.save(rf.model.state_dict(), os.path.join(args.save_dir, "final.ckpt"))
    wandb.finish()


if __name__ == "__main__":
    main()
