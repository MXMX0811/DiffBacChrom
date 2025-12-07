import os
import sys
import argparse
import torch
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffbacchrom.vae import StructureAutoencoderKL1D  # noqa: E402
from scripts.preprocess import center_batch, scale_batch  # noqa: E402


def load_struct_tensor(struct_path: str, device: torch.device) -> torch.Tensor:
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
    return struct_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data/train", help="Train data root (Pair_* folders)")
    parser.add_argument("--vae_ckpt", type=str, default="checkpoints/vae/epoch_040.pt", help="VAE checkpoint")
    parser.add_argument("--num_pairs", type=int, default=20, help="Number of Pair_* to use")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = StructureAutoencoderKL1D().to(device)
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    pair_dirs = [
        d for d in sorted(os.listdir(args.root_dir))
        if d.startswith("Pair_") and os.path.isdir(os.path.join(args.root_dir, d))
    ][: args.num_pairs]
    if not pair_dirs:
        print("No Pair_* folders found.")
        return

    mu_sum = None
    std_sum = None
    count = 0

    for pair in pair_dirs:
        pair_dir = os.path.join(args.root_dir, pair)
        hic_filename = f"{pair}_sim_hic_freq.tsv"
        struct_files = [
            f for f in os.listdir(pair_dir)
            if f.endswith(".tsv") and f != hic_filename
        ]
        for sf in struct_files:
            s_path = os.path.join(pair_dir, sf)
            struct_tensor = load_struct_tensor(s_path, device)
            struct_centered, _ = center_batch(struct_tensor)
            struct_norm, _ = scale_batch(struct_centered)
            with torch.no_grad():
                enc = vae.encode(struct_norm)
            mu = enc.mu  # (1,W,C)
            std = torch.exp(0.5 * enc.logvar)
            if mu_sum is None:
                mu_sum = mu.sum(dim=0, keepdim=True)
                std_sum = std.sum(dim=0, keepdim=True)
            else:
                mu_sum += mu.sum(dim=0, keepdim=True)
                std_sum += std.sum(dim=0, keepdim=True)
            count += mu.shape[0]

    if count == 0:
        print("No structures processed.")
        return

    mu_mean = mu_sum / count
    std_mean = std_sum / count
    print(f"Processed {count} samples from {len(pair_dirs)} Pair_* folders")
    print(f"Latent mean avg (scalar): {mu_mean.mean().item():.6f}")
    print(f"Latent std  avg (scalar): {std_mean.mean().item():.6f}")


if __name__ == "__main__":
    main()
