"""Standalone training demonstration of intermediate-layer condition tokens.

Model weights are updated while collecting visualizations. Formal training
experiments and this demo share model construction and latent preparation from
train/train_dit.py.
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

from data.dataset import HiCStructureDataset, collate_fn  # noqa: E402
from train.rf import RF
from train.train_dit import (
    build_parser, parse_config, load_models as load_training_models,
    prepare_latents, count_params, format_params, get_scheduler,
)


def build_dataloader(args: argparse.Namespace) -> Tuple[DataLoader, int]:
    dataset = HiCStructureDataset(root_dir=args.train_dir)

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
    vae, model = load_training_models(args, seq_len, device)

    if args.dit_ckpt is not None:
        print(f"Loading DiT checkpoint from {args.dit_ckpt}")
        ckpt = torch.load(args.dit_ckpt, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    return vae, model


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


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    parser.description = "Train the formal JointAttDiT model while visualizing intermediate condition tokens."
    parser.set_defaults(config=os.path.join(PROJECT_ROOT, "train", "configs", "train_mmdit.yaml"))
    parser.add_argument("--dit_ckpt", type=str, default=None,
                        help="Optional JointAttDiT weights to initialize the demonstration.")
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument("--train_steps", type=int, default=1000, help="Number of demonstration training steps.")
    parser.add_argument("--log_interval", type=int, default=100, help="Plot and log t-SNE every N steps.")
    args = parse_config(parser, argv)
    if args.model != "JointAttDiT":
        parser.error("This condition-trace demonstration supports JointAttDiT")
    if args.train_steps <= 0 or args.log_interval <= 0:
        parser.error("train_steps and log_interval must be positive")
    if not 0 <= args.warmup_steps < args.train_steps:
        parser.error("warmup_steps must be non-negative and smaller than train_steps")
    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    wandb.init(project="diffbacchrom-cond-trace", name=args.run_name, config=vars(args))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scaler = amp.GradScaler(enabled=use_amp)
    scheduler = None
    if args.warmup_steps > 0:
        scheduler = get_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.train_steps)
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
        if scheduler is not None:
            scheduler.step()

        log_data = {"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"],
                    "step": global_step + 1}

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
