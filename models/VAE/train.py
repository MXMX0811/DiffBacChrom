import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

# 你的模型和数据加载
from models.VAE.model import StructureAutoencoderKL1D   # 如果类名不同，请改这里
from scripts.dataloader import HiCStructureDataset, collate_fn


# ====================== KL loss（序列版） ======================

def kl_loss_seq(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    标准 VAE 的 KL 散度:
        KL( q(z|x) || N(0, I) ) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    这里 mu, logvar 形状为 (B, T, D)，
    我们对 (T, D) 做平均，再在 batch 上做平均，得到一个标量。
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, T, D)
    kl = kl.mean(dim=(1, 2))  # (B,)
    kl = kl.mean()            # 标量
    return kl


# ====================== 单个 epoch 的训练 ======================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    beta_kl: float,
    global_step: int,
    log_interval: int = 100,
) -> int:
    model.train()
    running_recon = 0.0
    running_kl = 0.0
    running_total = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        x = batch["structure"].to(device)  # (B, 928, 16)

        optimizer.zero_grad()

        # 假设 StructureVAE 的 forward 返回 (recon_x, mu, logvar)
        recon_x, mu, logvar = model(x)  # recon_x: (B, 928, 16)

        # 重建损失：MSE
        recon_loss = torch.mean((recon_x - x) ** 2)

        # KL loss
        kl = kl_loss_seq(mu, logvar)

        loss = recon_loss + beta_kl * kl
        loss.backward()
        optimizer.step()

        running_recon += recon_loss.item()
        running_kl += kl.item()
        running_total += loss.item()
        n_batches += 1
        global_step += 1

        # wandb 逐步记录
        wandb.log(
            {
                "train/loss": loss.item(),
                "train/recon_loss": recon_loss.item(),
                "train/kl_loss": kl.item(),
                "train/epoch": epoch,
                "train/step": global_step,
                "train/lr": optimizer.param_groups[0]["lr"],
            },
            step=global_step,
        )

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f}  Recon: {recon_loss.item():.6f}  KL: {kl.item():.6f}"
            )

    avg_recon = running_recon / n_batches
    avg_kl = running_kl / n_batches
    avg_total = running_total / n_batches

    print(
        f"==> Epoch [{epoch}] done. "
        f"Avg Loss: {avg_total:.6f}  Avg Recon: {avg_recon:.6f}  Avg KL: {avg_kl:.6f}"
    )

    wandb.log(
        {
            "epoch/loss": avg_total,
            "epoch/recon_loss": avg_recon,
            "epoch/kl_loss": avg_kl,
        },
        step=global_step,
    )

    return global_step


# ====================== 主函数 ======================

def main():
    parser = argparse.ArgumentParser(description="Train structure VAE")

    # 只保留少量关键参数，其他都固定在代码中
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run_name", type=str, default="structure-vae")
    args = parser.parse_args()

    # ---------- 固定参数 ----------
    ROOT_DIR = "data"
    HIC_DIRNAME = "Hi-C"
    STRUCT_DIRNAME = "structure"
    SEQ_LEN = 928           # hic_index 范围 0..927
    IN_CHANNELS = 16        # 每个 bin 的结构特征维度 (两 bead × 两链 × xyz+mask)
    LATENT_DIM = 64         # 每个位置的 latent 维度
    LR = 1e-4
    BETA_KL = 1e-3
    NUM_WORKERS = 4
    LOG_INTERVAL = 50
    SAVE_DIR = "checkpoints/vae"

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- 构建 Dataset / DataLoader ----------
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
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---------- 构建 VAE 模型 ----------
    # 如果你的类名 / 构造函数参数不一样，请改这里
    model = StructureAutoencoderKL1D(
        in_channels=IN_CHANNELS,
        latent_dim=LATENT_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---------- 初始化 wandb ----------
    wandb.init(
        project="structure-vae",
        name=args.run_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seq_len": SEQ_LEN,
            "in_channels": IN_CHANNELS,
            "latent_dim": LATENT_DIM,
            "lr": LR,
            "beta_kl": BETA_KL,
            "dataset_size": len(dataset),
        },
    )

    # 记录模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")
    wandb.config.update({"num_params": num_params})

    # ---------- 训练循环 ----------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        global_step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            beta_kl=BETA_KL,
            global_step=global_step,
            log_interval=LOG_INTERVAL,
        )

        # 保存 checkpoint
        ckpt_path = os.path.join(SAVE_DIR, f"vae_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")
        wandb.save(ckpt_path)

    print("Training finished.")
    wandb.finish()


if __name__ == "__main__":
    main()