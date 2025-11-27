import os
import glob
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class HiCStructureDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data",
        hic_dirname: str = "Hi-C",
        struct_dirname: str = "structure",
        hic_ext: str = ".tsv",
        struct_ext: str = ".tsv",
        expected_size: int = 928,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.hic_dir = os.path.join(root_dir, hic_dirname)
        self.struct_root_dir = os.path.join(root_dir, struct_dirname)
        self.hic_ext = hic_ext
        self.struct_ext = struct_ext
        self.expected_size = expected_size  # W

        self.samples: List[Tuple[str, str]] = []
        self._build_index()

    def _build_index(self):
        hic_files = sorted(glob.glob(os.path.join(self.hic_dir, f"*{self.hic_ext}")))
        if len(hic_files) == 0:
            raise RuntimeError(f"No Hi-C {self.hic_ext} files found in {self.hic_dir}")

        for hic_path in hic_files:
            base = os.path.splitext(os.path.basename(hic_path))[0]
            struct_dir = os.path.join(self.struct_root_dir, base)
            if not os.path.isdir(struct_dir):
                print(f"[Warning] structure folder not found for {base}: {struct_dir}")
                continue

            struct_files = sorted(glob.glob(os.path.join(struct_dir, f"*{self.struct_ext}")))
            if len(struct_files) == 0:
                print(f"[Warning] no structure {self.struct_ext} found in {struct_dir}")
                continue

            for s_path in struct_files:
                self.samples.append((hic_path, s_path))

        if len(self.samples) == 0:
            raise RuntimeError("No (Hi-C, structure) pairs found.")

        print(f"Found {len(self.samples)} samples in dataset.")

    # ---------- Hi-C 读取（和之前一样，如果你已经有就可以沿用） ----------
    def _load_hic_matrix(self, hic_path: str) -> torch.Tensor:
        df = pd.read_csv(hic_path, sep="\t")

        if "hic_index" not in df.columns:
            raise KeyError(f"'hic_index' column not found in {hic_path}")

        # 这里我仍然按 hic_index 排序，保证矩阵确实是 0..W-1 的顺序
        # （如果你确信文件本身就是这个顺序，也可以去掉这一行）
        df = df.sort_values("hic_index").reset_index(drop=True)

        if len(df) != self.expected_size:
            raise ValueError(
                f"Hi-C size mismatch in {hic_path}: got {len(df)}, expected {self.expected_size}"
            )

        value_cols = [c for c in df.columns if c != "hic_index"]
        mat = df[value_cols].to_numpy(dtype=np.float32)  # (W, W)

        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"Hi-C matrix in {hic_path} is not square: {mat.shape}")

        hic_tensor = torch.from_numpy(mat).float().unsqueeze(0)  # (1, W, W)
        return hic_tensor

    # ---------- 结构 1856 行 → 928 × 16 ----------
    def _load_structure_seq(self, struct_path: str) -> torch.Tensor:
        """
        结构 tsv 格式（按你给的示例）：
            hic_index  bead_index1  x1  y1  z1  mask1  bead_index2  x2  y2  z2  mask2

        总行数 = 2 * W（例如 1856）。
        同一个 hic_index 恰好对应两行，分别是这两个 bead 的槽位信息。
        每行的结构特征是 8 维：
            [x1, y1, z1, mask1, x2, y2, z2, mask2]

        我们对每个 hic_index，把这两行按「原始出现顺序」当作 beadA / beadB，
        拼成一个 16 维 token：
            token_j = [feat_row1(8维), feat_row2(8维)]

        最终得到：
            feat_tensor: (W, 16)
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

        # 不按 bead_index 重排，只保留原始行顺序
        # 如果你确信文件已经按 hic_index 从 0..W-1 排好，这里也不用再 sort
        # 只检查行数是否为 2*W
        if len(df) != 2 * self.expected_size:
            raise ValueError(
                f"Structure row count mismatch in {struct_path}: "
                f"got {len(df)}, expected {2 * self.expected_size}"
            )

        # 按 hic_index 分组，但不改变 group 的出现顺序
        groups = df.groupby("hic_index", sort=False)

        if len(groups) != self.expected_size:
            raise ValueError(
                f"Number of unique hic_index in {struct_path} is {len(groups)}, "
                f"expected {self.expected_size}"
            )

        tokens = []
        for hic_idx, g in groups:
            # 每个 hic_index 必须有两行
            if len(g) != 2:
                raise ValueError(
                    f"For hic_index={hic_idx} in {struct_path}, expected 2 rows, got {len(g)}."
                )

            # 不按 bead_index1 排序，直接用文件中的先后顺序
            r1 = g.iloc[0]
            r2 = g.iloc[1]

            # 每行 8 维特征：x1,y1,z1,mask1,x2,y2,z2,mask2
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

        feat_arr = np.asarray(tokens, dtype=np.float32)  # (W, 16)
        feat_tensor = torch.from_numpy(feat_arr)         # (W, 16)
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
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    hic_list = [item["hic"] for item in batch]
    struct_list = [item["structure"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    struct_files = [item["structure_file"] for item in batch]

    hic = torch.stack(hic_list, dim=0)          # (B, 1, W, W)
    structure = torch.stack(struct_list, dim=0) # (B, W, 16)

    return {
        "hic": hic,
        "structure": structure,
        "sample_id": sample_ids,
        "structure_file": struct_files,
    }


if __name__ == "__main__":
    dataset = HiCStructureDataset(
        root_dir="data",
        hic_dirname="Hi-C",
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
        hic = batch["hic"]              # (B, 1, 928, 928)
        structure = batch["structure"]  # (B, 928, 16)
        print("Hi-C batch shape:", hic.shape)
        print("Structure batch shape:", structure.shape)
        print("Sample IDs:", batch["sample_id"])
        break