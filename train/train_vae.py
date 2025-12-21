import os
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
import wandb

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.vae1d import StructureAutoencoderKL1D
from models.sdvae1d import SDVAE
from data.dataset import HiCStructureDataset, collate_fn

# shared indices
COORD_IDX = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
MASK_IDX = [3, 7, 11, 15]


def kl_loss_seq(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B,T,D)
    return kl.mean()


def compute_vae_losses(x, recon_x, mu, logvar, bce_mask, kl_weight: float, lambda_mask: float):
    """
    x/recon_x: (B, T, 16) normalized
    """
    coords = x[..., COORD_IDX]           # (B, T, 12)
    recon_coords = recon_x[..., COORD_IDX]

    mask_target = x[..., MASK_IDX]       # (B, T, 4), 0/1
    mask_pred = recon_x[..., MASK_IDX]   # (B, T, 4) logits

    m0 = mask_target[..., 0:1]
    m1 = mask_target[..., 1:2]
    m2 = mask_target[..., 2:3]
    m3 = mask_target[..., 3:4]

    w0 = m0.expand_as(coords[..., 0:3])
    w1 = m1.expand_as(coords[..., 3:6])
    w2 = m2.expand_as(coords[..., 6:9])
    w3 = m3.expand_as(coords[..., 9:12])
    coord_weight = torch.cat([w0, w1, w2, w3], dim=-1)  # (B, T, 12)

    # 坐标加权 MSE
    coord_mse = (recon_coords - coords) ** 2 * coord_weight   # (B, T, 12)

    # 先在坐标维求和 → 每个 token 的误差 & 权重
    coord_mse = coord_mse.sum(dim=-1)        # (B, T)
    coord_weight_sum = coord_weight.sum(dim=-1)  # (B, T)

    # 再在序列维求和 → 每个样本的总误差 & 总权重
    per_seq_err = coord_mse.sum(dim=-1)          # (B,)
    per_seq_weight = coord_weight_sum.sum(dim=-1).clamp_min(1.0)  # (B,)

    coord_loss_per_seq = per_seq_err / per_seq_weight   # (B,)
    coord_loss = coord_loss_per_seq.mean()              # scalar

    mask_loss = bce_mask(mask_pred, mask_target)
    kl = kl_loss_seq(mu, logvar)

    loss = coord_loss + lambda_mask * mask_loss + kl_weight * kl
    return loss, coord_loss, mask_loss, kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--run_name", type=str, default="diffbacchrom-vae")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, choices=["vae1d", "sdvae1d"], default="sdvae1d")
    parser.add_argument("--kl_weight", type=float, default=5e-3, help="KL loss weight")
    parser.add_argument("--mask_weight", type=float, default=1.0, help="Mask loss weight")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--use_seq_compression", type=bool, default=False)
    args = parser.parse_args()
    
    if args.kl_weight is None:
        args.kl_weight = 5e-3 if args.model == "vae1d" else 1e-6
    if args.lr is None:
        args.lr = 1e-4 if args.model == "vae1d" else 1e-5
    if args.mask_weight is None:
        args.mask_weight = 1.0 if args.model == "vae1d" else 0.5
        
    ROOT_DIR = "data/train"
    HIC_DIRNAME = "Hi-C"
    STRUCT_DIRNAME = "structure"
    SEQ_LEN = 928
    IN_CHANNELS = 16
    SAVE_DIR = os.path.join("checkpoints", "vae", args.model)
    NUM_WORKERS = 4

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = HiCStructureDataset(
        root_dir=ROOT_DIR,
        hic_dirname=HIC_DIRNAME,
        struct_dirname=STRUCT_DIRNAME,
        expected_size=SEQ_LEN,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, train=True),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if args.model == "vae1d":
        model = StructureAutoencoderKL1D(
            in_channels=IN_CHANNELS, 
            num_res_blocks=18, 
            use_downsample=args.use_seq_compression
        ).to(device)

        def forward_batch(x):
            return model(x)

    else:
        model = SDVAE(dtype=torch.float32, device=device).to(device)

        def forward_batch(x):
            return model(x)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce_mask = torch.nn.BCEWithLogitsLoss().to(device)

    wandb.init(
        project="diffbacchrom-vae",
        name=args.run_name,
        config=dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=SEQ_LEN,
            lr=args.lr,
            kl_weight=args.kl_weight,
            lambda_mask=args.mask_weight,
            model=args.model,
        ),
    )
    
    global_step = 0
    
    total_steps = args.epochs * len(dataloader)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0
        total_coord = 0
        total_mask = 0
        total_kl = 0
        n = 0
        
        latent_sq_sum = 0.0

        for batch_idx, batch in enumerate(dataloader):
            x = batch["structure"].to(device)  # (B,928,16)

            optimizer.zero_grad()
            recon_x, mu, logvar = forward_batch(x)
            
            # lambda_mask = LAMBDA_MASK * (0.1 + 0.9 * (1 - batch_idx / total_steps))
            loss, coord_loss, mask_loss, kl = compute_vae_losses(
                x, recon_x, mu, logvar, bce_mask, kl_weight=args.kl_weight, lambda_mask=args.mask_weight
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mu32 = mu.detach().float()
                latent_sq_sum += mu32.pow(2).mean().item()   # 统计均值，不会溢出

            total_loss += loss.item()
            total_coord += coord_loss.item()
            total_mask += mask_loss.item()
            total_kl += kl.item()
            n += 1
            global_step += 1

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/coord_loss": coord_loss.item(),
                    "train/mask_loss": mask_loss.item(),
                    "train/kl_loss": kl.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"[Epoch {epoch} | Step {batch_idx+1}/{len(dataloader)}] "
                    f"Loss={loss:.6f} Coord={coord_loss:.6f} "
                    f"Mask={mask_loss:.6f} KL={kl:.6f}"
                )

        avg_loss = total_loss / n
        avg_coord = total_coord / n
        avg_mask = total_mask / n
        avg_kl = total_kl / n

        # per-epoch latent scale estimate
        latent_var_epoch = latent_sq_sum / max(n, 1)
        latent_std_epoch = latent_var_epoch ** 0.5
        suggested_scale_epoch = 1.0 / latent_std_epoch

        print(
            f"Epoch {epoch} | "
            f"avg_loss={avg_loss:.6f}  "
            f"avg_coord={avg_coord:.6f}  "
            f"avg_mask={avg_mask:.6f}  "
            f"avg_kl={avg_kl:.6f}  "
            f"scale_est={suggested_scale_epoch:.6f}"
        )

        wandb.log(
            {
                "epoch/loss": avg_loss,
                "epoch/coord_loss": avg_coord,
                "epoch/mask_loss": avg_mask,
                "epoch/kl_loss": avg_kl,
                "epoch/latent_scale_est": suggested_scale_epoch,
            },
            step=global_step,
        )

        if epoch % 5 == 0:
            ckpt_path = f"{SAVE_DIR}/epoch_{epoch:03d}.pt"
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                ckpt_path,
            )
            wandb.save(ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

    wandb.finish()
    print("\n=== TRAINING COMPLETED ===\n")


if __name__ == "__main__":
    main()
