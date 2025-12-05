import argparse
import os
from functools import partial
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import sys
sys.path.append(".")
from model import DiT_models  # noqa: E402
from models.VAE.model import StructureAutoencoderKL1D  # noqa: E402
from scripts.dataloader import HiCStructureDataset, collate_fn  # noqa: E402


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
    def sample(self, hic: torch.Tensor, sample_steps: int, shape: torch.Size):
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
            vc = self.model(z, t, hic)
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
    struct_lookup: basename(struct_path) -> full struct_path (for column names and bead indices)
    """
    os.makedirs(output_dir, exist_ok=True)
    coord_split = [
        ["x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"],
        ["x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"],
    ]

    for b_idx, (sid, sfile) in enumerate(zip(sample_ids, structure_files)):
        struct_path = struct_lookup.get(sfile)
        if struct_path is None:
            print(f"[Warn] structure file not found in lookup: {sfile}, skip saving.")
            continue

        template_df = pd.read_csv(struct_path, sep="\t")
        # ensure numeric columns are float to avoid dtype warnings on assignment
        for c in coord_split[0] + coord_split[1]:
            if c in template_df.columns:
                template_df[c] = template_df[c].astype(float)
        pred = struct_denorm[b_idx].cpu().numpy()  # (W,16)

        # iterate hic_index groups in original order
        grouped = template_df.groupby("hic_index", sort=False)
        token_idx = 0
        for hic_idx, grp in grouped:
            if len(grp) != 2:
                raise ValueError(f"Expected 2 rows per hic_index in {sfile}, got {len(grp)} at {hic_idx}")
            token = pred[token_idx]
            token_idx += 1

            row1_vals = token[:8]
            row2_vals = token[8:]

            row_indices = grp.index.tolist()
            template_df.loc[row_indices[0], coord_split[0]] = row1_vals
            template_df.loc[row_indices[1], coord_split[1]] = row2_vals

        out_path = os.path.join(output_dir, f"{sid}_recon.tsv")
        template_df.to_csv(out_path, sep="\t", index=False)
        print(f"Saved reconstruction to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data/train")
    parser.add_argument("--hic_dirname", type=str, default="Hi-C")
    parser.add_argument("--struct_dirname", type=str, default="structure")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_scale", type=float, default=1.335256)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="checkpoints/dit")
    parser.add_argument("--vae_ckpt", type=str, default="checkpoints/vae/epoch_040.pt")
    parser.add_argument("--run_name", type=str, default="rf_dit_structure")
    args = parser.parse_args()

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

    model = DiT_models["DiT-L"](input_size=seq_len, in_channels=vae.z_channels).to(device)

    rf = RF(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

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

            epoch_loss += loss.item()
            wandb.log({"train/loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.6f}")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

        # ---------- inference sample ----------
        rf.model.eval()
        with torch.no_grad():
            vis_loader = DataLoader(
                dataset,
                batch_size=min(5, args.batch_size),
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
            )
            decoded = vae.decode(sample_latent / args.latent_scale)  # (B,W,16) in normalized space
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
