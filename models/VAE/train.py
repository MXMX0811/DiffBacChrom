import os
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
import wandb

import sys
sys.path.append(".")
from models.VAE.model import StructureAutoencoderKL1D
from scripts.dataloader import HiCStructureDataset, collate_fn

# shared indices
COORD_IDX = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
MASK_IDX = [3, 7, 11, 15]


def kl_loss_seq(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B,T,D)
    return kl.mean()


def compute_vae_losses(x, recon_x, mu, logvar, bce_mask, beta_kl: float, lambda_mask: float):
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
    coord_weight = torch.cat([w0, w1, w2, w3], dim=-1)  # (B,T,12)

    coord_mse = (recon_coords - coords) ** 2 * coord_weight
    denom = coord_weight.sum().clamp_min(1.0)
    coord_loss = coord_mse.sum() / denom

    mask_loss = bce_mask(mask_pred, mask_target)
    kl = kl_loss_seq(mu, logvar)

    loss = coord_loss + lambda_mask * mask_loss + beta_kl * kl
    return loss, coord_loss, mask_loss, kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--run_name", type=str, default="structure-vae")
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    ROOT_DIR = "data/train"
    HIC_DIRNAME = "Hi-C"
    STRUCT_DIRNAME = "structure"
    SEQ_LEN = 928
    IN_CHANNELS = 16
    LR = 1e-4
    BETA_KL = 5e-3
    SAVE_DIR = "checkpoints/vae"
    NUM_WORKERS = 4
    LAMBDA_MASK = 1.0

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

    model = StructureAutoencoderKL1D(in_channels=IN_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    bce_mask = torch.nn.BCEWithLogitsLoss().to(device)

    wandb.init(
        project="structure-vae",
        name=args.run_name,
        config=dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=SEQ_LEN,
            lr=LR,
            beta_kl=BETA_KL,
            lambda_mask=LAMBDA_MASK,
        ),
    )

    global_step = 0
    latent_sq_sum = 0.0
    latent_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0
        total_coord = 0
        total_mask = 0
        total_kl = 0
        n = 0

        for batch_idx, batch in enumerate(dataloader):
            x = batch["structure"].to(device)  # (B,928,16)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)

            loss, coord_loss, mask_loss, kl = compute_vae_losses(
                x, recon_x, mu, logvar, bce_mask, beta_kl=BETA_KL, lambda_mask=LAMBDA_MASK
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                latent_sq_sum += (mu ** 2).sum().item()
                latent_count += mu.numel()

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
        latent_var_epoch = latent_sq_sum / max(latent_count, 1)
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
