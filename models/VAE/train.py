import os
import argparse

import torch
from torch.utils.data import DataLoader
import wandb

from models.VAE.model import StructureVAE   # 修改成你的真实VAE类名
from scripts.dataloader import HiCStructureDataset, collate_fn


# ====================== KL LOSS（序列版：928×16 -> μ/σ） ======================

def kl_loss_seq(mu, logvar):
    """
    标准 VAE KL 散度：对每个序列位置求KL，再对batch平均，返回标量
    mu/logvar 形状 (B, T, D)
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B,T,D)
    return kl.mean()  # 全局平均作为 final KL loss


# ====================== TRAIN.PY 主程序（无train_one_epoch函数） ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--run_name", type=str, default="structure-vae")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # ====== 固定训练参数（核心你可以改）======
    ROOT_DIR = "data"
    HIC_DIRNAME = "Hi-C"
    STRUCT_DIRNAME = "structure"
    SEQ_LEN = 928              # hic_index bins
    IN_CHANNELS = 16           # 结构输入维度 = bead_XYZ×2 + mask×2
    LATENT_DIM = 64
    LR = 1e-4
    BETA_KL = 1e-3
    SAVE_DIR = "checkpoints/vae"
    NUM_WORKERS = 4

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====== DataLoader ======
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
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ====== INIT MODEL ======
    model = StructureVAE(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ====== init wandb ======
    wandb.init(
        project="structure-vae",
        name=args.run_name,
        config=dict(
            epochs=args.epochs,
            latent_dim=LATENT_DIM,
            batch_size=args.batch_size,
            seq_len=SEQ_LEN,
            lr=LR,
            beta_kl=BETA_KL,
        ),
    )

    # ========================== TRAIN LOOP ==========================

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        n = 0

        for batch_idx, batch in enumerate(dataloader):
            x = batch["structure"].to(device)  # (B,928,16)

            optimizer.zero_grad()

            # ===== forward =====
            recon_x, mu, logvar = model(x)

            # ===== loss =====
            recon_loss = torch.mean((recon_x - x) ** 2)
            kl = kl_loss_seq(mu, logvar)
            loss = recon_loss + BETA_KL * kl

            loss.backward()
            optimizer.step()

            # ===== accumulate =====
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()
            n += 1
            global_step += 1

            # wandb logging
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/recon_loss": recon_loss.item(),
                    "train/kl_loss": kl.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            if (batch_idx + 1) % 50 == 0:
                print(f"[Epoch {epoch} | Step {batch_idx+1}/{len(dataloader)}] "
                      f"Loss={loss:.6f} Recon={recon_loss:.6f} KL={kl:.6f}")

        # ===== epoch summary =====
        print(f"==> Epoch {epoch} DONE | avg_loss={total_loss/n:.6f} "
              f"avg_recon={total_recon/n:.6f} avg_kl={total_kl/n:.6f}\n")

        wandb.log(
            {
                "epoch/loss": total_loss/n,
                "epoch/recon_loss": total_recon/n,
                "epoch/kl_loss": total_kl/n,
            },
            step=global_step,
        )

        # ===== save checkpoint =====
        ckpt_path = f"{SAVE_DIR}/epoch_{epoch:03d}.pt"
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
            ckpt_path,
        )
        wandb.save(ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")

    wandb.finish()
    print("\n=== TRAINING COMPLETED ===\n")


if __name__ == "__main__":
    main()