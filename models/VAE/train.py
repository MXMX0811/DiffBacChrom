import os
import argparse

import torch
from torch.utils.data import DataLoader
import wandb

import sys
sys.path.append(".")
from models.VAE.model import StructureAutoencoderKL1D
from scripts.dataloader import HiCStructureDataset, collate_fn

from functools import partial


def kl_loss_seq(mu, logvar):
    """
    标准 VAE KL 散度：对每个序列位置求KL，再对batch平均，返回标量
    mu/logvar 形状 (B, T, D)
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B,T,D)
    return kl.mean()  # 全局平均作为 final KL loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--run_name", type=str, default="structure-vae")
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    ROOT_DIR = "data"
    HIC_DIRNAME = "Hi-C"
    STRUCT_DIRNAME = "structure"
    SEQ_LEN = 928              # hic_index bins
    IN_CHANNELS = 16           # 结构输入维度 = bead_XYZ×2 + mask×2（共16维）
    LR = 1e-4
    BETA_KL = 5e-3
    SAVE_DIR = "checkpoints/vae"
    NUM_WORKERS = 4
    LAMBDA_MASK = 1.0          # MODIFIED: mask BCE 的权重

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

    # ====== INIT MODEL ======
    model = StructureAutoencoderKL1D(in_channels=IN_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # MODIFIED: BCE loss 用于 mask 通道（logits -> {0,1}）
    bce_mask = torch.nn.BCEWithLogitsLoss().to(device)

    # ====== init wandb ======
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

    # ========================== TRAIN LOOP ==========================

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0
        total_coord = 0      # MODIFIED
        total_mask = 0       # MODIFIED
        total_kl = 0
        n = 0

        for batch_idx, batch in enumerate(dataloader):
            x = batch["structure"].to(device)  # (B,928,16)

            optimizer.zero_grad()

            # ===== forward =====
            recon_x, mu, logvar = model(x)

            # ===== 拆分坐标和mask（按照我们约定的16维布局）=====  # MODIFIED
            # token维度含义：
            # [0] x1_beadA   [1] y1_beadA   [2] z1_beadA   [3] mask1_beadA
            # [4] x2_beadA   [5] y2_beadA   [6] z2_beadA   [7] mask2_beadA
            # [8] x1_beadB   [9] y1_beadB  [10] z1_beadB  [11] mask1_beadB
            # [12] x2_beadB [13] y2_beadB [14] z2_beadB [15] mask2_beadB

            # 坐标通道索引（12维）
            coord_idx = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
            # mask 通道索引（4维）
            mask_idx = [3, 7, 11, 15]

            coords = x[..., coord_idx]           # (B, T, 12)
            recon_coords = recon_x[..., coord_idx]

            mask_target = x[..., mask_idx]       # (B, T, 4), 0/1
            mask_pred = recon_x[..., mask_idx]   # (B, T, 4) logits

            # ===== 对坐标使用 mask 加权 MSE =====  # MODIFIED
            # 每个 mask 对应三个坐标：
            # mask_target[...,0] -> coords[...,0:3]
            # mask_target[...,1] -> coords[...,3:6]
            # mask_target[...,2] -> coords[...,6:9]
            # mask_target[...,3] -> coords[...,9:12]

            m0 = mask_target[..., 0:1]  # (B,T,1)
            m1 = mask_target[..., 1:2]
            m2 = mask_target[..., 2:3]
            m3 = mask_target[..., 3:4]

            w0 = m0.expand_as(coords[..., 0:3])   # (B,T,3)
            w1 = m1.expand_as(coords[..., 3:6])
            w2 = m2.expand_as(coords[..., 6:9])
            w3 = m3.expand_as(coords[..., 9:12])

            coord_weight = torch.cat([w0, w1, w2, w3], dim=-1)  # (B,T,12)

            coord_mse = (recon_coords - coords) ** 2 * coord_weight
            # 只在 mask=1 的位置归一化
            denom = coord_weight.sum().clamp_min(1.0)
            coord_loss = coord_mse.sum() / denom

            # ===== mask loss：对mask通道做 BCEWithLogitsLoss =====  # MODIFIED
            mask_loss = bce_mask(mask_pred, mask_target)

            # ===== KL loss =====
            kl = kl_loss_seq(mu, logvar)

            # ===== 总 loss =====  # MODIFIED
            loss = coord_loss + LAMBDA_MASK * mask_loss + BETA_KL * kl

            loss.backward()
            optimizer.step()

            # ===== accumulate =====
            total_loss += loss.item()
            total_coord += coord_loss.item()
            total_mask += mask_loss.item()
            total_kl += kl.item()
            n += 1
            global_step += 1

            # wandb logging
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

        # ===== epoch summary =====
        avg_loss = total_loss / n
        avg_coord = total_coord / n
        avg_mask = total_mask / n
        avg_kl = total_kl / n

        print(
            f"Epoch {epoch} | "
            f"avg_loss={avg_loss:.6f}  "
            f"avg_coord={avg_coord:.6f}  "
            f"avg_mask={avg_mask:.6f}  "
            f"avg_kl={avg_kl:.6f}"
        )

        wandb.log(
            {
                "epoch/loss": avg_loss,
                "epoch/coord_loss": avg_coord,
                "epoch/mask_loss": avg_mask,
                "epoch/kl_loss": avg_kl,
            },
            step=global_step,
        )

        # ===== save checkpoint =====
        if epoch % 10 == 0:
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
