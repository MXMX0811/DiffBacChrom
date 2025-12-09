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


def rebuild_structure_tables(struct_pred: torch.Tensor, template_df: pd.DataFrame, output_dir: str):
    """
    Save reconstructed structures to TSVs matching original format.
    struct_pred: (B, W, 16) on CPU (normalized space)
    """
    os.makedirs(output_dir, exist_ok=True)
    coord_split = [
        ["x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"],
        ["x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"],
    ]

    grouped = template_df.groupby("hic_index", sort=False)
    for b_idx in range(struct_pred.shape[0]):
        df = template_df.copy()
        for c in ["x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"]:
            if c in df.columns:
                df[c] = df[c].astype(float)
        pred = struct_pred[b_idx].cpu().numpy()  # (W,16) normalized

        token_idx = 0
        for _, grp in grouped:
            token = pred[token_idx]
            token_idx += 1

            row1_vals = token[:8]
            row2_vals = token[8:]

            row_indices = grp.index.tolist()
            df.loc[row_indices[0], coord_split[0]] = row1_vals
            df.loc[row_indices[1], coord_split[1]] = row2_vals

        df = df.drop(columns=[c for c in df.columns if c.startswith("bead_index")], errors="ignore")
        out_path = os.path.join(output_dir, f"noise_sample_{b_idx+1}.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        print(f"Saved reconstruction to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=str, required=True, help="Path to a template TSV for column/order")
    parser.add_argument("--vae_ckpt", type=str, default="checkpoints/vae/epoch_040.pt", help="VAE checkpoint path")
    parser.add_argument("--noise_batch", type=int, default=500, help="Number of samples to decode")
    parser.add_argument("--mu_mean", type=float, default=0.0, help="Scalar mean of latent distribution")
    parser.add_argument("--std_mean", type=float, default=1.0, help="Scalar std of latent distribution")
    parser.add_argument("--outputs_dir", type=str, default="outputs/vae_samples", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = StructureAutoencoderKL1D().to(device)
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # load template to infer seq_len
    df = pd.read_csv(args.template, sep="\t")
    struct_tensor = []
    groups = df.groupby("hic_index", sort=False)
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
        struct_tensor.append(token)
    struct_tensor = torch.tensor(struct_tensor, dtype=torch.float32).unsqueeze(0).to(device)  # (1,W,16)
    seq_len = struct_tensor.shape[1]

    mu = torch.full((1, seq_len, vae.z_channels), args.mu_mean, device=device)
    std = torch.full((1, seq_len, vae.z_channels), args.std_mean, device=device)
    noise = mu + std * torch.randn(args.noise_batch, seq_len, vae.z_channels, device=device)

    with torch.no_grad():
        recon_from_noise = vae.decode(noise)  # (B, W, 16) normalized
        recon_from_noise = apply_mask_threshold(recon_from_noise)

    rebuild_structure_tables(
        recon_from_noise.cpu(),
        df,
        output_dir=args.outputs_dir,
    )
    print(f"Saved {args.noise_batch} decoded noise samples to {args.outputs_dir}")


if __name__ == "__main__":
    main()
