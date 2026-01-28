import argparse
import yaml
import os
from functools import partial
from typing import Dict, List

import pandas as pd
import numpy as np 
import torch
from torch.utils.data import DataLoader
from torch import amp
from tqdm import tqdm
import wandb

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.crossdit import DiT_models as CrossDiT_models
from models.mmdit import DiT_models as MMDiT_models
from models.mmditx import DiT_models as MMDiTX_models
from models.resnet_vae import StructureAutoencoderKL1D
from models.sd_vae import SDVAE
from data.dataset import HiCStructureDataset, collate_fn
from rf import RF


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
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_scale", type=float, default=1.335256, help="Latent scale used during training")
    parser.add_argument("--sample_steps", type=int, default=50, help="RF sampling steps")
    parser.add_argument("--model", type=str, default="JointAttDiT", choices=["CrossDiT", "JointAttDiT", "MMDiTX"], help="Select backbone model")
    parser.add_argument(
        "--size",
        type=lambda s: s.upper(),
        default="L",
        choices=["S", "B", "L", "XL"],
        help="DiT model size (S/B/L/XL)",
    )
    parser.add_argument("--use_seq_compression", type=bool, default=False)
    parser.add_argument("--use_global_cond", type=bool, default=True, help="Whether CrossDiT uses global conditioning (CrossDiT only)")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Classifier-free guidance scale for inference")
    parser.add_argument("--grad_cp", type=bool, default=True, help="Use gradient checkpointing to save memory")
    parser.add_argument("--save_dir", type=str, default=None, help="Checkpoint directory (default: checkpoints/dit/<model>)")
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="rf_dit_structure")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Warmup steps for cosine scheduler (set 0 to keep constant learning rate)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16"],
        help="Precision strategy: fp32 or autocast (fp16). Default: CrossDiT->fp32, others->fp16.",
    )
    args = parser.parse_args()

    if args.cfg_scale is None:
        args.cfg_scale = 1.0 if args.model == "CrossDiT" else 1.5
    if args.warmup_steps is None:
        args.warmup_steps = 0 if args.model == "CrossDiT" else 1000
    if args.precision is None:
        args.precision = "fp32" if args.model == "CrossDiT" else "fp16"
        
    if args.model != "CrossDiT":
        if args.use_global_cond:
            print("Warning: --use_global_cond is only supported when model=CrossDiT; disabled.")
            args.use_global_cond = False
        if not args.use_seq_compression:
            print("Warning: --use_seq_compression must be True for JointAttDiT and MMDiTX; enabled.")
            args.use_seq_compression = True
            
    if args.model == "MMDiTX":        
        if args.grad_cp:
            print("Warning: --grad_cp is only supported when model=CrossDiT or JointAttDiT; disabled.")
            args.grad_cp = False
    
    if args.vae_ckpt is None:
        if args.use_seq_compression:
            args.vae_ckpt = os.path.join("checkpoints", "vae", "sdvae1d" + "final.pt")
        else:
            args.vae_ckpt = os.path.join("checkpoints", "vae", "vae1d" + "final.pt")

    if args.save_dir is None:
        args.save_dir = os.path.join("checkpoints", "dit", args.model + "-" + args.size)
    
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
        
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = args.precision == "fp16" and device.type == "cuda"
    if args.precision == "fp16" and device.type != "cuda":
        print("AMP selected but CUDA is not available; falling back to fp32.")
        args.precision = "fp32"
        use_amp = False

    train_set = HiCStructureDataset(root_dir=args.train_dir)
    val_set = HiCStructureDataset(root_dir=args.val_dir)

    # Infer sequence length W from first sample (structure and Hi-C should match)
    first_hic_path, first_struct_path = train_set.samples[0]
    seq_len_struct = train_set._load_structure_seq(first_struct_path).shape[0]
    seq_len_hic = train_set._load_hic_matrix(first_hic_path).shape[-1]
    if seq_len_struct != seq_len_hic:
        raise ValueError(f"Mismatch between structure length ({seq_len_struct}) and Hi-C size ({seq_len_hic})")
    seq_len = seq_len_struct
    print(f"Inferred sequence length: W={seq_len}")
    
    dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=partial(collate_fn, train=True),
    )
    
    val_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=partial(collate_fn, train=False),
    )

    if not args.use_seq_compression:
        vae = StructureAutoencoderKL1D().to(device)
    else:
        vae = SDVAE().to(device)
        
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    dit_size_key = f"DiT-{args.size}"
    if args.model == "CrossDiT":
        model_fn = CrossDiT_models[dit_size_key]
        model_kwargs = {
            "input_size": seq_len // (4 if args.use_seq_compression else 1),
            "in_channels": 16,
            "use_global_cond": args.use_global_cond,
            "seq_compression": args.use_seq_compression,
            "gradient_checkpointing": args.grad_cp,
        }
    elif args.model == "JointAttDiT":
        model_fn = MMDiT_models[dit_size_key]
        model_kwargs = {
            "input_size": seq_len // 4,
            "in_channels": 16,
            "gradient_checkpointing": args.grad_cp,
        }
    elif args.model == "MMDiTX":
        model_fn = MMDiTX_models[dit_size_key]
        model_kwargs = {
            "input_size": seq_len // 4,
            "in_channels": 16,
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
    scaler = amp.GradScaler(enabled=use_amp)
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
            with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = rf.forward(z, hic)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(rf.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
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

        rf.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                hic = val_batch["hic"].to(device)
                structure = val_batch["structure"].to(device)

                z = vae.encode(structure).latent_dist.sample().mul_(args.latent_scale)
                with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = rf.forward(z, hic)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.6f}")
        wandb.log({"val/loss": avg_val_loss, "epoch": epoch + 1})

        ckpt_payload = {
            "epoch": epoch + 1,
            "model": rf.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
        }
        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pt")
        torch.save(ckpt_payload, ckpt_path)
        wandb.save(ckpt_path)

    torch.save(rf.model.state_dict(), os.path.join(args.save_dir, "final.ckpt"))
    wandb.finish()


if __name__ == "__main__":
    main()
