"""Standalone training demonstration of intermediate-layer condition tokens.

Model weights are updated while collecting visualizations. Formal training
experiments use train/train_dit.py.
"""

import argparse
import os
import sys
from functools import partial
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import amp
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mmdit import DiT_models as MMDiT_models  # noqa: E402
from models.resnet_vae import StructureAutoencoderKL1D  # noqa: E402
from data.dataset import HiCStructureDataset, collate_fn  # noqa: E402
from train.rf import RF


def build_dataloader(args: argparse.Namespace) -> Tuple[DataLoader, int]:
    dataset = HiCStructureDataset(root_dir=args.root_dir)

    first_hic_path, first_struct_path = dataset.samples[0]
    seq_len_struct = dataset._load_structure_seq(first_struct_path).shape[0]
    seq_len_hic = dataset._load_hic_matrix(first_hic_path).shape[-1]
    if seq_len_struct != seq_len_hic:
        raise ValueError(f"Mismatch between structure length ({seq_len_struct}) and Hi-C size ({seq_len_hic})")
    seq_len = seq_len_struct

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=partial(collate_fn, train=True),
    )
    return dataloader, seq_len


def load_models(args: argparse.Namespace, seq_len: int, device: torch.device):
    vae = StructureAutoencoderKL1D().to(device)
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    dit_size_key = f"DiT-{args.size}"
    model_fn = MMDiT_models[dit_size_key]
    model = model_fn(
        input_size=seq_len,
        in_channels=vae.z_channels,
        gradient_checkpointing=args.grad_cp,
    ).to(device)

    if args.dit_ckpt is not None:
        print(f"Loading DiT checkpoint from {args.dit_ckpt}")
        ckpt = torch.load(args.dit_ckpt, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Loaded checkpoint with missing keys: {missing} and unexpected keys: {unexpected}")

    model.eval()
    return vae, model


def prepare_latents(structure: torch.Tensor, vae: StructureAutoencoderKL1D, latent_scale: float) -> torch.Tensor:
    with torch.no_grad():
        z = vae.encode(structure).latent_dist.sample().mul_(latent_scale)
    return z


def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def format_params(n: int) -> str:
    return f"{n:,} ({n / 1e6:.2f}M)"


def plot_tsne(cond_trace: List[torch.Tensor], seed: int) -> plt.Figure:
    num_layers = len(cond_trace)
    colors = plt.cm.Blues(np.linspace(0.25, 0.95, num_layers))

    pooled_list = []
    layer_ids = []
    for layer_idx, cond_tokens in enumerate(cond_trace):
        if cond_tokens.dim() != 3:
            raise ValueError(f"Expected cond_tokens to be 3D, got {cond_tokens.shape}")
        # Pool token dimension so each sample in the batch becomes one vector
        pooled = cond_tokens.mean(dim=1)  # (B, D)
        pooled_list.append(pooled.detach().cpu().float())
        layer_ids.extend([layer_idx] * pooled.shape[0])

    all_vectors = torch.cat(pooled_list, dim=0).numpy()  # (num_layers * B, D)
    total_points = all_vectors.shape[0]
    if total_points < 3:
        raise ValueError("t-SNE needs at least 3 samples; please increase batch size.")
    perplexity = min(30, total_points - 1)

    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    embedding = tsne.fit_transform(all_vectors)  # (num_layers * B, 2)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for idx, layer_idx in enumerate(layer_ids):
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            color=colors[layer_idx],
            alpha=0.9,
            s=24,
        )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Blues,
        norm=plt.Normalize(vmin=0, vmax=num_layers - 1),
    )
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Layer depth (darker = deeper)")

    ax.set_title(f"cond_trace t-SNE ({total_points} points = layers * batch)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JointAttDiT to demonstrate intermediate-layer condition tokens with t-SNE.")
    parser.add_argument("--root_dir", type=str, default="data/train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--latent_scale", type=float, default=1.335256)
    parser.add_argument(
        "--model",
        type=str,
        default="JointAttDiT",
        choices=["JointAttDiT"],
        help="Visualization is only supported for JointAttDiT (mmdit.py).",
    )
    parser.add_argument(
        "--size",
        type=lambda s: s.upper(),
        default="L",
        choices=["S", "B", "L", "XL"],
        help="DiT model size (S/B/L/XL)",
    )
    parser.add_argument(
        "--dit_ckpt",
        type=str,
        default=None,
        help="Optional path to a JointAttDiT checkpoint. Leave empty to train from scratch.",
    )
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument("--train_steps", type=int, default=1000, help="Number of training steps to run.")
    parser.add_argument("--log_interval", type=int, default=100, help="Plot and log t-SNE every N steps.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--run_name", type=str, default="cond_trace_tsne")
    parser.add_argument("--grad_cp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16"],
        help="Precision strategy: fp32 or autocast (fp16). Default: fp16 for JointAttDiT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model != "JointAttDiT":
        raise ValueError("Visualization requires model=JointAttDiT so that cond_trace is returned.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.precision is None:
        args.precision = "fp16"
    use_amp = args.precision == "fp16" and device.type == "cuda"
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; select --precision fp32 on CPU")

    dataloader, seq_len = build_dataloader(args)
    vae, model = load_models(args, seq_len, device)

    # Report parameter counts (align with train_dit)
    vae_params = count_params(vae, trainable_only=False)
    hic_params = count_params(model.hic_encoder, trainable_only=False)
    dit_backbone_params = sum(p.numel() for name, p in model.named_parameters() if "hic_encoder" not in name)
    print("Parameter counts (all params):")
    print(f"  VAE: {format_params(vae_params)}")
    print(f"  HiC encoder: {format_params(hic_params)}")
    print(f"  DiT backbone: {format_params(dit_backbone_params)}")

    wandb.init(project="rf_dit_structure", name=args.run_name, config=vars(args))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scaler = amp.GradScaler(enabled=use_amp)
    rf = RF(model)
    model.train()

    data_iter = iter(dataloader)
    progress = tqdm(total=args.train_steps, desc="Training steps")
    global_step = 0
    while global_step < args.train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        hic = batch["hic"].to(device)
        structure = batch["structure"].to(device)
        z = prepare_latents(structure, vae, args.latent_scale)

        collect_trace = (global_step + 1) % args.log_interval == 0 or (global_step + 1) == args.train_steps
        optimizer.zero_grad()
        with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            if collect_trace:
                loss, cond_trace = rf.forward(z, hic, return_trace=True)
            else:
                loss = rf.forward(z, hic)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        log_data = {"train/loss": loss.item(), "step": global_step + 1}

        if collect_trace:
            fig = plot_tsne(cond_trace, seed=args.tsne_seed + global_step)
            log_data["cond_trace_tsne"] = wandb.Image(fig, caption=f"step {global_step+1}")
            plt.close(fig)

        wandb.log(log_data, step=global_step + 1)
        global_step += 1
        progress.update(1)
    progress.close()

    wandb.finish()


if __name__ == "__main__":
    main()
