import os
import glob
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data.transforms import center_batch, scale_batch, random_rotate_batch


class HiCStructureDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data",
        hic_ext: str = ".tsv",
        struct_ext: str = ".tsv",
        expected_size: int = 928,
    ):
        """
        root_dir/
          Pair_1/
            Pair_1_sim_hic_freq.tsv
            xxx_structure_1.tsv
            xxx_structure_2.tsv
          Pair_2/
            Pair_2_sim_hic_freq.tsv
            ...
        """
        super().__init__()
        self.root_dir = root_dir

        self.hic_ext = hic_ext
        self.struct_ext = struct_ext
        self.expected_size = expected_size  # W（目前不再强制检查）

        self.samples: List[Tuple[str, str]] = []
        self._build_index()

    def _build_index(self):
        """
        新版本索引构建：
        - 遍历 root_dir 下所有以 'Pair_' 开头的子目录（Pair_1, Pair_2, ...）
        - 在每个 Pair_X 下找 Pair_X_sim_hic_freq.tsv 作为 Hi-C 文件
        - 同目录下其余 .tsv 文件视为结构文件
        - 每个 (hic, struct) 组合形成一个样本
        """
        pair_dirs = [
            d for d in sorted(os.listdir(self.root_dir))
            if d.startswith("Pair_") and os.path.isdir(os.path.join(self.root_dir, d))
        ]

        if len(pair_dirs) == 0:
            raise RuntimeError(f"No Pair_* folders found in {self.root_dir}")

        for pair_name in pair_dirs:
            pair_dir = os.path.join(self.root_dir, pair_name)

            # Hi-C 文件名：Pair_X_sim_hic_freq.tsv
            hic_filename = f"{pair_name}_sim_hic_freq{self.hic_ext}"
            hic_path = os.path.join(pair_dir, hic_filename)
            if not os.path.exists(hic_path):
                print(f"[Warning] Hi-C file not found for {pair_name}: {hic_path}")
                continue

            # 结构文件：同目录下所有 .tsv，排除 hic 文件本身
            struct_files = sorted(
                f for f in os.listdir(pair_dir)
                if f.endswith(self.struct_ext) and f != hic_filename
            )
            if len(struct_files) == 0:
                print(f"[Warning] no structure {self.struct_ext} found in {pair_dir}")
                continue

            for s_fname in struct_files:
                s_path = os.path.join(pair_dir, s_fname)
                self.samples.append((hic_path, s_path))

        if len(self.samples) == 0:
            raise RuntimeError("No (Hi-C, structure) pairs found.")

        print(f"Found {len(self.samples)} samples in dataset.")

    # ---------- Hi-C 读取 ----------

    def _load_hic_matrix(self, hic_path: str) -> torch.Tensor:
        """
        读取 Hi-C tsv：
            hic_index  0  1  2 ...
            0          ...
            1          ...
        不再强制检查大小为 expected_size，只要求是方阵。
        """
        df = pd.read_csv(hic_path, sep="\t")

        if "hic_index" not in df.columns:
            raise KeyError(f"'hic_index' column not found in {hic_path}")

        # 仍然按 hic_index 排序，保证行顺序稳定
        df = df.sort_values("hic_index").reset_index(drop=True)

        value_cols = [c for c in df.columns if c != "hic_index"]
        mat = df[value_cols].to_numpy(dtype=np.float32)  # (W, W)

        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"Hi-C matrix in {hic_path} is not square: {mat.shape}")

        hic_tensor = torch.from_numpy(mat).float().unsqueeze(0)  # (1, W, W)
        return hic_tensor

    # ---------- 结构：2*W 行 → W × 16 ----------

    def _load_structure_seq(self, struct_path: str) -> torch.Tensor:
        """
        结构 tsv 格式（按你给的示例）：
            hic_index  bead_index1  x1  y1  z1  mask1  bead_index2  x2  y2  z2  mask2

        每个 hic_index 对应两行（两个 bead 槽位），我们不再检查总行数是否是 2*expected_size，
        也不检查 unique hic_index 数是否等于 expected_size，而是按实际文件内容动态决定长度 W。

        每个 hic_index 生成 16 维 token：
            [x1_A, y1_A, z1_A, mask1_A, x2_A, y2_A, z2_A, mask2_A,
             x1_B, y1_B, z1_B, mask1_B, x2_B, y2_B, z2_B, mask2_B]
        """
        df = pd.read_csv(struct_path, sep="\t")

        required_cols = [
            "hic_index",
            "bead_index1", "x1", "y1", "z1", "mask1",
            "bead_index2", "x2", "y2", "z2", "mask2",
        ]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in {struct_path}")

        # 按 hic_index 分组，不改变出现顺序
        groups = df.groupby("hic_index", sort=False)

        tokens = []
        for hic_idx, g in groups:
            # 仍然假设每个 hic_index 恰有两行（两个 bead 槽位）
            if len(g) != 2:
                raise ValueError(
                    f"For hic_index={hic_idx} in {struct_path}, expected 2 rows, got {len(g)}."
                )

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

            token = bead1_feat + bead2_feat  # (16,)
            tokens.append(token)

        feat_arr = np.asarray(tokens, dtype=np.float32)  # (W, 16)，W 为实际 hic_index 数量
        feat_tensor = torch.from_numpy(feat_arr)
        return feat_tensor

    # ---------- Dataset 接口 ----------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        hic_path, struct_path = self.samples[idx]

        hic_tensor = self._load_hic_matrix(hic_path)        # (1, W, W)
        struct_seq = self._load_structure_seq(struct_path)  # (W, 16)

        sample_id = os.path.splitext(os.path.basename(hic_path))[0]
        struct_name = os.path.basename(struct_path)

        return {
            "hic": hic_tensor,
            "structure": struct_seq,
            "sample_id": sample_id,
            "structure_file": struct_name,
        }


# ---------- collate_fn & DataLoader ----------

def collate_fn(batch: List[Dict[str, Any]], train: bool) -> Dict[str, Any]:
    """
    仍然简单 stack；假设同一个 batch 中 W 一样大（你现在的数据应该还是同一分辨率），
    如果将来要支持变长序列，需要在这里做 padding。
    """
    hic_list = [item["hic"] for item in batch]
    struct_list = [item["structure"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    struct_files = [item["structure_file"] for item in batch]

    hic = torch.stack(hic_list, dim=0)          # (B, 1, W, W)
    structure = torch.stack(struct_list, dim=0) # (B, W, 16)
    
    # Preprocess: centering, scaling
    structure, centroid = center_batch(structure)
    structure, scale = scale_batch(structure)
    if train:
        structure = random_rotate_batch(structure)
        
    return {
        "hic": hic,
        "structure": structure,
        "centroid": centroid,
        "scale": scale,
        "sample_id": sample_ids,
        "structure_file": struct_files,
    }


if __name__ == "__main__":
    dataset = HiCStructureDataset(
        root_dir="data",
        hic_dirname="Hi-C",      # 虽然不再使用，但接口保留
        struct_dirname="structure",
        expected_size=928,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    for batch in loader:
        hic = batch["hic"]              # (B, 1, W, W)
        structure = batch["structure"]  # (B, W, 16)
        print("Hi-C batch shape:", hic.shape)
        print("Structure batch shape:", structure.shape)
        print("Sample IDs:", batch["sample_id"])
        break
