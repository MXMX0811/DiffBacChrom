import os
import argparse
import shutil
import sys
from typing import List

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.crossdit import DiT_models as CrossDiT_models
from models.mambadit import DiT_models as MambaDiT_models
from models.mmdit import DiT_models as MMDiT_models
from models.mmditx import DiT_models as MMDiTX_models
from models.resnet_vae import StructureAutoencoderKL1D
from models.sd_vae import SDVAE
from train.rf import RF


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
    struct_denorm: (B, W, 16) or compatible shape on CPU
    start_idx: global offset so filenames do not collide across batches
    """
    os.makedirs(out_dir, exist_ok=True)

    if not isinstance(struct_denorm, torch.Tensor):
        struct_denorm = torch.as_tensor(struct_denorm)

    struct = struct_denorm
    if struct.ndim == 2:
        struct = struct.unsqueeze(0)
    elif struct.ndim == 4 and struct.shape[-2:] == (4, 4):
        struct = struct.reshape(struct.shape[0], struct.shape[1], 16)
    elif struct.ndim == 4 and struct.shape[1] == 1 and struct.shape[-1] == 16:
        struct = struct.squeeze(1)

    if struct.ndim != 3 or struct.shape[-1] != 16:
        raise ValueError(f"Expected struct shape (B, W, 16); got {tuple(struct.shape)}")

    start_idx = int(start_idx)
    if start_idx < 0:
        raise ValueError(f"start_idx must be >= 0; got {start_idx}")

    for b_idx in range(struct.shape[0]):
        tokens = struct[b_idx].cpu().numpy()  # (W,16)
        rows: List[List[float]] = []
        for hic_idx, token in enumerate(tokens):
            row1_vals = token[:8].tolist()
            row2_vals = token[8:].tolist()
            rows.append([hic_idx] + row1_vals)
            rows.append([hic_idx] + row2_vals)
        df = pd.DataFrame(
            rows,
            columns=["hic_index", "x1", "y1", "z1", "mask1", "x2", "y2", "z2", "mask2"],
        )
        out_path = os.path.join(out_dir, f"sample_{start_idx + b_idx + 1:04d}.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        print(f"Saved {out_path}")

def apply_mask_threshold(struct: torch.Tensor) -> torch.Tensor:
    """
    Binarize decoder mask logits (>0.5 -> 1, otherwise 0) and zero coordinates where mask is 0.
    Expects last dimension ordered as (x, y, z, mask) repeated 4 times.
    """
    struct_view = struct.reshape(*struct.shape[:-1], 4, 4)  # (..., 4 beads, xyz+mask)
    masks = (struct_view[..., 3] > 0.5).float()
    struct_view[..., 3] = masks
    struct_view[..., 0:3] = struct_view[..., 0:3] * masks.unsqueeze(-1)
    return struct_view.reshape(struct.shape)

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hic_path", type=str, default="data/test/Pair_74/Pair_74_sim_hic_freq.tsv", help="Path to Hi-C tsv (e.g., data/train/Pair_X/Pair_X_sim_hic_freq.tsv)")
    parser.add_argument("--dit_ckpt", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to the selected VAE checkpoint")
    parser.add_argument("--sample_steps", type=int, default=50, help="RF sampling steps")
    parser.add_argument("--model", type=str, default="CrossDiT", choices=["CrossDiT", "JointAttDiT", "MMDiTX", "MambaDiT"], help="Select backbone model")
    parser.add_argument(
        "--size",
        type=lambda s: s.upper(),
        default="S",
        choices=["S", "B", "L", "XL"],
        help="DiT model size (S/B/L/XL)",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale for inference")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of sequences to generate")
    parser.add_argument("--latent_scale", type=float, default=1.335256, help="Latent scale used during training")
    parser.add_argument("--output_root", type=str, default=None, help="Output root directory")
    parser.add_argument("--use_global_cond", action=argparse.BooleanOptionalAction, default=None, help="Whether CrossDiT/MambaDiT uses global conditioning")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of structures per sampling batch")
    parser.add_argument("--use_seq_compression", action=argparse.BooleanOptionalAction, default=None,
                        help="Use SD VAE with 4x compression (default: off for CrossDiT/MambaDiT, on for joint models)")
    args = parser.parse_args(argv)
    
    is_cross_model = args.model in ("CrossDiT", "MambaDiT")
    if args.use_seq_compression is None:
        args.use_seq_compression = not is_cross_model
    if not is_cross_model and not args.use_seq_compression:
        parser.error("JointAttDiT and MMDiTX sampling require use_seq_compression")
    if args.use_global_cond is None:
        args.use_global_cond = is_cross_model
    if not is_cross_model and args.use_global_cond:
        parser.error("use_global_cond is supported only by CrossDiT and MambaDiT")
    if args.batch_size <= 0 or args.sample_steps <= 0 or args.num_samples <= 0:
        parser.error("batch_size, sample_steps and num_samples must be positive")
    suffix = "-compression" if is_cross_model and args.use_seq_compression else ""
    model_name = args.model + '-' + args.size + suffix
    if args.dit_ckpt is None:
        args.dit_ckpt = os.path.join("checkpoints", "dit", model_name, "final.ckpt")
    if args.output_root is None:
        args.output_root = os.path.join("outputs", "dit_samples", model_name)

    return args


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hic = load_hic_matrix(args.hic_path).to(device)  # (1,1,W,W)
    seq_len = hic.shape[-1]
    hic_basename = os.path.splitext(os.path.basename(args.hic_path))[0]
    out_dir = os.path.join(args.output_root, hic_basename)
    os.makedirs(out_dir, exist_ok=True)

    vae = (SDVAE() if args.use_seq_compression else StructureAutoencoderKL1D()).to(device)
    latent_size = seq_len // vae.downsample_factor
    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(vae_ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    ckpt_data = torch.load(args.dit_ckpt, map_location="cpu")

    dit_size_key = f"DiT-{args.size}"
    if args.model in ("CrossDiT", "MambaDiT"):
        models = CrossDiT_models if args.model == "CrossDiT" else MambaDiT_models
        model_fn = models[dit_size_key]
        model_kwargs = {
            "input_size": seq_len,
            "latent_size": latent_size,
            "in_channels": vae.z_channels,
            "use_global_cond": args.use_global_cond,
            "seq_compression": args.use_seq_compression,
            "gradient_checkpointing": False,
        }
    elif args.model == "JointAttDiT":
        model_fn = MMDiT_models[dit_size_key]
        model_kwargs = {
            "input_size": seq_len,
            "latent_size": latent_size,
            "in_channels": vae.z_channels,
            "gradient_checkpointing": False,
        }
    elif args.model == "MMDiTX":
        model_fn = MMDiTX_models[dit_size_key]
        model_kwargs = {
            "input_size": seq_len,
            "latent_size": latent_size,
            "in_channels": vae.z_channels,
        }
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    dit = model_fn(**model_kwargs).to(device)
        
    state_dict = ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data else ckpt_data
    dit.load_state_dict(state_dict)
    dit.eval()

    rf = RF(dit)

    # replicate hic for batch sampling
    # generate in smaller batches and save incrementally
    remaining = args.num_samples

    saved_count = 0
    pbar = tqdm(total=args.num_samples, desc="Generating samples")
    while remaining > 0:
        cur_bs = min(args.batch_size, remaining)
        hic_batch = hic.repeat(cur_bs, 1, 1, 1)  # (cur_bs,1,W,W)
        sample_latent = rf.sample(
            hic_batch,
            sample_steps=args.sample_steps,
            shape=(cur_bs, latent_size, vae.z_channels),
            cfg_scale=args.cfg_scale,
        )
        decoded = vae.decode(sample_latent / args.latent_scale)  # normalized space
        decoded = apply_mask_threshold(decoded)
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
