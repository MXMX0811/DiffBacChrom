import os
import glob
import argparse
from typing import List

import pandas as pd
import torch

import sys
sys.path.append(".")
from models.VAE.model import StructureAutoencoderKL1D  # noqa: E402
from scripts.train_VAE import compute_vae_losses, COORD_IDX, MASK_IDX  # noqa: E402
from scripts.preprocess import center_batch, scale_batch  # noqa: E402


def rebuild_structure_tables(struct_pred: torch.Tensor, template_dfs: List[pd.DataFrame], output_files: List[str], output_dir: str):
    """
    Save reconstructed structures to TSVs matching original format.
    struct_pred: (B, W, 16) on CPU (normalized space)
    """
    os.makedirs(output_dir, exist_ok=True)
    coord_split = [
        ["x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"],
        ["x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"],
    ]

    for b_idx, (template_df, out_name) in enumerate(zip(template_dfs, output_files)):
        df = template_df.copy()
        # ensure coord/mask columns are float to avoid pandas dtype warnings on assignment
        for c in ["x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"]:
            if c in df.columns:
                df[c] = df[c].astype(float)
        pred = struct_pred[b_idx].cpu().numpy()  # (W,16) normalized

        grouped = df.groupby("hic_index", sort=False)
        token_idx = 0
        for _, grp in grouped:
            token = pred[token_idx]
            token_idx += 1

            row1_vals = token[:8]
            row2_vals = token[8:]

            row_indices = grp.index.tolist()
            df.loc[row_indices[0], coord_split[0]] = row1_vals
            df.loc[row_indices[1], coord_split[1]] = row2_vals

        out_path = os.path.join(output_dir, out_name)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"Saved reconstruction to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data/test", help="Root directory containing test TSV files")
    parser.add_argument("--ckpt", type=str, default="checkpoints/vae/epoch_040.pt", help="VAE checkpoint path")
    parser.add_argument("--save_recon", action="store_true", help="If set, save reconstructed TSV files")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory to save TSVs/plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataset: all *.tsv under data/test/** (recursively)
    struct_files = sorted(glob.glob(os.path.join(args.root_dir, "**", "*.tsv"), recursive=True))

    # model
    vae = StructureAutoencoderKL1D().to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    outputs_dir = args.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)

    all_recon = []
    all_templates = []
    all_output_names = []

    # loss components (match training)
    BETA_KL = 5e-3
    LAMBDA_MASK = 1.0
    bce_mask = torch.nn.BCEWithLogitsLoss().to(device)

    total_loss = 0.0
    total_coord = 0.0
    total_mask = 0.0
    total_kl = 0.0
    n_batches = 0
    latent_sq_sum = 0.0
    latent_count = 0

    first_template_df = None

    for struct_path in struct_files:
        df = pd.read_csv(struct_path, sep="\t")
        groups = df.groupby("hic_index", sort=False)
        tokens = []
        for _, g in groups:
            r1 = g.iloc[0]
            r2 = g.iloc[1]
            bead1_feat = [
                float(r1["x1"]), float(r1["y1"]), float(r1["z1"]), float(r1["mask1"]),
                float(r1["x2"]), float(r1["y2"]), float(r1["z2"]), float(r1["mask2"]),
            ]
            bead2_feat = [
                float(r2["x1"]), float(r2["y1"]), float(r2["z1"]), float(r2["mask1"]),
                float(r2["x2"]), float(r2["y2"]), float(r2["z2"]), float(r2["mask2"]),
            ]
            token = bead1_feat + bead2_feat
            tokens.append(token)

        struct_tensor = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0).to(device)  # (1,W,16)

        # normalize (center + scale) same as training
        struct_centered, _ = center_batch(struct_tensor)
        struct_norm, _ = scale_batch(struct_centered)

        with torch.no_grad():
            enc = vae.encode(struct_norm)
            z = enc.latent_dist.sample()
            recon = vae.decode(z)  # normalized
            recon_norm = recon.cpu()

            loss, coord_loss, mask_loss, kl = compute_vae_losses(
                struct_norm, recon, enc.mu, enc.logvar, bce_mask, beta_kl=BETA_KL, lambda_mask=LAMBDA_MASK
            )

            total_loss += loss.item()
            total_coord += coord_loss.item()
            total_mask += mask_loss.item()
            total_kl += kl.item()
            n_batches += 1
            latent_sq_sum += (enc.mu ** 2).sum().item()
            latent_count += enc.mu.numel()

        all_recon.append(recon_norm)
        all_templates.append(df)
        all_output_names.append(os.path.basename(struct_path))
        if first_template_df is None:
            first_template_df = df

    if n_batches > 0:
        avg_loss = total_loss / n_batches
        avg_coord = total_coord / n_batches
        avg_mask = total_mask / n_batches
        avg_kl = total_kl / n_batches
        print(f"Test set losses - total: {avg_loss:.6f}, coord: {avg_coord:.6f}, mask: {avg_mask:.6f}, kl: {avg_kl:.6f}")
        if latent_count > 0:
            latent_var = latent_sq_sum / latent_count
            latent_std = latent_var ** 0.5
            suggested_scale = 1.0 / latent_std
            print(f"Test latent_std: {latent_std:.6f}")
            print(f"Suggested DiT latent scale (1/std): {suggested_scale:.6f}")
        """
        checkpoints/vae/epoch_020.pt
        Test set losses - total: 0.007103, coord: 0.001924, mask: 0.000067, kl: 1.022382
        Test latent_std: 0.691148
        Suggested DiT latent scale (1/std): 1.446867
        
        checkpoints/vae/epoch_040.pt
        Test set losses - total: 0.006871, coord: 0.001911, mask: 0.000016, kl: 0.988756
        Test latent_std: 0.681476
        Suggested DiT latent scale (1/std): 1.467403
        """
    else:
        print("No batches processed; check dataset path.")

    recon_cat = torch.cat(all_recon, dim=0)
    if args.save_recon:
        rebuild_structure_tables(
            recon_cat,
            all_templates,
            all_output_names,
            output_dir=outputs_dir,
        )

    # decode noise samples using first template length
    if first_template_df is not None:
        seq_len = len(first_template_df.groupby("hic_index", sort=False))
        noise_batch = 10
        noise = torch.randn(noise_batch, seq_len, vae.z_channels, device=device)
        with torch.no_grad():
            recon_from_noise = vae.decode(noise)  # (B, W, 16) normalized
        noise_templates = [first_template_df.copy() for _ in range(noise_batch)]
        noise_output_names = [f"noise_sample_{i+1}.tsv" for i in range(noise_batch)]
        rebuild_structure_tables(
            recon_from_noise.cpu(),
            noise_templates,
            noise_output_names,
            output_dir=os.path.join(outputs_dir, "recon_from_noise"),
        )
        print(f"Saved {noise_batch} decoded noise samples to {os.path.join(outputs_dir, 'recon_from_noise')}")


if __name__ == "__main__":
    main()
