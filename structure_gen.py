import os
import argparse
import shutil
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffbacchrom.dit import DiT_models  # noqa: E402
from diffbacchrom.vae import StructureAutoencoderKL1D  # noqa: E402
from scripts.train_dit import RF  # noqa: E402

ORIGINAL_RMS = 11.1462
NORMALIZED_RMS = 1.1525


def load_hic_matrix(hic_path: str) -> torch.Tensor:
    df = pd.read_csv(hic_path, sep="\t")
    if "hic_index" not in df.columns:
        raise KeyError(f"'hic_index' column not found in {hic_path}")
    df = df.sort_values("hic_index").reset_index(drop=True)
    value_cols = [c for c in df.columns if c != "hic_index"]
    mat = df[value_cols].to_numpy(dtype=float)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Hi-C matrix is not square: {mat.shape}")
    return torch.from_numpy(mat).float().unsqueeze(0).unsqueeze(0)  # (1,1,W,W)

def save_structures(struct_denorm: torch.Tensor, out_dir: str, start_idx: int = 0):
    """
    struct_denorm: (B, W, 16) on CPU
    start_idx: global offset so filenames do not collide across batches
    """
    os.makedirs(out_dir, exist_ok=True)
    for b_idx in range(struct_denorm.shape[0]):
        tokens = struct_denorm[b_idx].cpu().numpy()  # (W,16)
        rows: List[List[float]] = []
        for hic_idx, token in enumerate(tokens):
            row1_vals = token[:8].tolist()
            row2_vals = token[8:].tolist()
            rows.append([hic_idx] + row1_vals)
            rows.append([hic_idx] + row2_vals)
        df = pd.DataFrame(rows, columns=["hic_index", "x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"])
        out_path = os.path.join(out_dir, f"sample_{start_idx + b_idx + 1:04d}.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        print(f"Saved {out_path}")

def apply_mask_threshold(struct: torch.Tensor) -> torch.Tensor:
    """
    Binarize mask channels (>0.5 -> 1, otherwise 0) and zero coordinates where mask is 0.
    Expects last dimension ordered as (x, y, z, mask) repeated 4 times.
    """
    struct_view = struct.reshape(*struct.shape[:-1], 4, 4)  # (..., 4 beads, xyz+mask)
    masks = (struct_view[..., 3] > 0.5).float()
    struct_view[..., 3] = masks
    struct_view[..., 0:3] = struct_view[..., 0:3] * masks.unsqueeze(-1)
    return struct_view.reshape(struct.shape)


def scale_xyz(struct: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale xyz channels by a factor without touching mask channels."""
    struct_view = struct.reshape(*struct.shape[:-1], 4, 4)
    struct_view[..., 0:3] = struct_view[..., 0:3] * scale
    return struct_view.reshape(struct.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hic_path", type=str, default="data/train/Pair_10/Pair_10_sim_hic_freq.tsv", help="Path to Hi-C tsv (e.g., data/train/Pair_X/Pair_X_sim_hic_freq.tsv)")
    parser.add_argument("--dit_ckpt", type=str, default="checkpoints/dit/epoch_016.pt", help="DiT checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, default="checkpoints/vae/epoch_040.pt", help="VAE checkpoint path")
    parser.add_argument("--sample_steps", type=int, default=50, help="RF sampling steps")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of sequences to generate")
    parser.add_argument("--latent_scale", type=float, default=1.335256, help="Latent scale used during training")
    parser.add_argument("--output_root", type=str, default="outputs/dit_samples", help="Output root directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hic = load_hic_matrix(args.hic_path).to(device)  # (1,1,W,W)
    seq_len = hic.shape[-1]
    hic_basename = os.path.splitext(os.path.basename(args.hic_path))[0]
    out_dir = os.path.join(args.output_root, hic_basename)
    os.makedirs(out_dir, exist_ok=True)

    vae = StructureAutoencoderKL1D().to(device)
    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(vae_ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    dit = DiT_models["DiT-L"](input_size=seq_len, in_channels=vae.z_channels).to(device)
    ckpt = torch.load(args.dit_ckpt, map_location="cpu")
    dit.load_state_dict(ckpt["model"])
    dit.eval()

    rf = RF(dit)

    # replicate hic for batch sampling
    # generate in smaller batches and save incrementally
    batch_size = min(50, args.num_samples)
    remaining = args.num_samples

    saved_count = 0
    pbar = tqdm(total=args.num_samples, desc="Generating samples")
    while remaining > 0:
        cur_bs = min(batch_size, remaining)
        hic_batch = hic.repeat(cur_bs, 1, 1, 1)  # (cur_bs,1,W,W)
        sample_latent = rf.sample(hic_batch, sample_steps=args.sample_steps, shape=(cur_bs, seq_len, vae.z_channels))
        decoded = vae.decode(sample_latent / args.latent_scale)  # normalized space
        decoded = apply_mask_threshold(decoded)
        decoded = scale_xyz(decoded, ORIGINAL_RMS / NORMALIZED_RMS)
        decoded_cpu = decoded.cpu()
        save_structures(decoded_cpu, out_dir, start_idx=saved_count)

        remaining -= cur_bs
        saved_count += cur_bs
        pbar.update(cur_bs)
    pbar.close()
    print(f"Saved {saved_count} samples to {out_dir}")

    # copy hic file
    shutil.copy(args.hic_path, os.path.join(out_dir, os.path.basename(args.hic_path)))
    print(f"Copied hic file to {out_dir}")


if __name__ == "__main__":
    main()
