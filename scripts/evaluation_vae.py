import os
import glob
import argparse
from typing import List

import pandas as pd
import torch

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffbacchrom.vae import StructureAutoencoderKL1D  # noqa: E402
from scripts.train_vae import compute_vae_losses, COORD_IDX, MASK_IDX  # noqa: E402
from scripts.preprocess import center_batch, scale_batch  # noqa: E402
from scripts.sample_vae import rebuild_structure_tables  # noqa: E402


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
    mu_sum = None
    std_sum = None
    count_mu = 0

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
            # accumulate latent stats
            std = torch.exp(0.5 * enc.logvar).cpu()
            mu_cpu = enc.mu.cpu()
            if mu_sum is None:
                mu_sum = mu_cpu.sum(dim=0, keepdim=True)
                std_sum = std.sum(dim=0, keepdim=True)
            else:
                mu_sum += mu_cpu.sum(dim=0, keepdim=True)
                std_sum += std.sum(dim=0, keepdim=True)
            count_mu += mu_cpu.shape[0]
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
        # reuse rebuild_structure_tables from sample_vae via simple loop
        os.makedirs(outputs_dir, exist_ok=True)
        for b_idx, (template_df, out_name) in enumerate(zip(all_templates, all_output_names)):
            rebuild_structure_tables(
                recon_cat[b_idx:b_idx+1],
                template_df,
                output_dir=outputs_dir,
            )


if __name__ == "__main__":
    main()
