import os
import glob
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from scipy.io import loadmat


class HiCStructureDataset(Dataset):
    """
    Read Hi-C matrices and structure sequences from data/Hi-C and data/structure

    Directory structure：
        root_dir/
            Hi-C/
                sample1.mat
                sample2.mat
                ...
            structure/
                sample1/
                    struct_001.tsv
                    struct_002.tsv
                    ...
                sample2/
                    struct_001.tsv
                    ...
    """

    def __init__(
        self,
        root_dir: str = "data",
        hic_dirname: str = "Hi-C",
        struct_dirname: str = "structure",
        hic_ext: str = ".mat",
        struct_ext: str = ".tsv",
        expected_size: int = 928,
        feature_cols: List[str] = None,
    ):
        """
        参数：
            root_dir: 数据根目录（包含 Hi-C 和 structure 子目录）
            hic_dirname: 存放 Hi-C .mat 的子目录名
            struct_dirname: 存放结构文件夹的子目录名
            hic_ext: Hi-C 文件扩展名
            struct_ext: 结构文件扩展名
            expected_size: 期望的 Hi-C 矩阵大小 W（也对应序列长度）
            feature_cols: 用于结构特征的列名列表，默认为
                          ["x1","y1","z1","m1","x2","y2","z2","m2"]
        """
        super().__init__()
        self.root_dir = root_dir
        self.hic_dir = os.path.join(root_dir, hic_dirname)
        self.struct_root_dir = os.path.join(root_dir, struct_dirname)
        self.hic_ext = hic_ext
        self.struct_ext = struct_ext
        self.expected_size = expected_size

        # 默认使用的 8 个特征列：第一条链 + 第二条链
        if feature_cols is None:
            self.feature_cols = ["x1", "y1", "z1", "m1", "x2", "y2", "z2", "m2"]
        else:
            self.feature_cols = feature_cols

        # 收集所有 (hic_path, struct_path) pair
        self.samples: List[Tuple[str, str]] = []
        self._build_index()

    def _build_index(self):
        """
        扫描 Hi-C 目录和 structure 目录，建立
        self.samples = [(hic_path, struct_tsv_path), ...]
        """
        if not os.path.isdir(self.hic_dir):
            raise FileNotFoundError(f"Hi-C directory not found: {self.hic_dir}")
        if not os.path.isdir(self.struct_root_dir):
            raise FileNotFoundError(f"Structure directory not found: {self.struct_root_dir}")

        hic_files = sorted(glob.glob(os.path.join(self.hic_dir, f"*{self.hic_ext}")))
        if len(hic_files) == 0:
            raise RuntimeError(f"No .mat files found in {self.hic_dir}")

        for hic_path in hic_files:
            base = os.path.splitext(os.path.basename(hic_path))[0]  # e.g. sample1
            struct_dir = os.path.join(self.struct_root_dir, base)
            if not os.path.isdir(struct_dir):
                # 若找不到对应结构文件夹，可选择跳过或报错，这里选择跳过
                # 你也可以改成 raise
                print(f"[Warning] structure folder not found for {base}: {struct_dir}")
                continue

            struct_files = sorted(glob.glob(os.path.join(struct_dir, f"*{self.struct_ext}")))
            if len(struct_files) == 0:
                print(f"[Warning] no structure .tsv found in {struct_dir}")
                continue

            # 每个结构文件都和这个 hic_path 配对
            for s_path in struct_files:
                self.samples.append((hic_path, s_path))

        if len(self.samples) == 0:
            raise RuntimeError("No (Hi-C, structure) pairs found. Check your folder structure.")

        print(f"Found {len(self.samples)} samples in dataset.")

    # ---------- 载入 Hi-C .mat 辅助函数 ----------

    def _load_hic_matrix(self, hic_path: str) -> torch.Tensor:
        """
        从 .mat 文件中取出 2D Hi-C 矩阵，形状应为 (W, W)。

        这里假设 .mat 文件中有一个 2D 数组变量。
        若变量名不确定，就取第一个符合条件的。
        """
        mat = loadmat(hic_path)

        # mat 是一个 dict，里面有很多键：__header__, __version__, __globals__, 以及真正的数据变量
        hic_array = None
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 2:
                hic_array = v
                break

        if hic_array is None:
            raise RuntimeError(f"No 2D array found in {hic_path}")

        if hic_array.shape[0] != hic_array.shape[1]:
            raise ValueError(f"Hi-C matrix in {hic_path} is not square: {hic_array.shape}")

        if hic_array.shape[0] != self.expected_size:
            raise ValueError(
                f"Hi-C size mismatch in {hic_path}: got {hic_array.shape[0]}, "
                f"expected {self.expected_size}"
            )

        # 转成 float32 tensor，增加一个 channel 维度： (1, W, W)
        hic_tensor = torch.from_numpy(hic_array).float().unsqueeze(0)
        return hic_tensor

    # ---------- 载入结构 tsv 辅助函数 ----------

    def _load_structure_seq(self, struct_path: str) -> torch.Tensor:
        """
        从 tsv 中读取结构序列，返回 shape = (W, C_feat) 的 tensor。

        要求 tsv 至少包含 self.feature_cols 中的列。
        若有 hic_index 列，则会按 hic_index 排序以对齐 Hi-C 矩阵。
        """
        df = pd.read_csv(struct_path, sep="\t")

        # 如果有 hic_index 列，则按它排序，确保与 Hi-C bin 对齐
        if "hic_index" in df.columns:
            df = df.sort_values("hic_index").reset_index(drop=True)

        # 检查长度是否匹配 expected_size
        if len(df) != self.expected_size:
            raise ValueError(
                f"Structure length mismatch in {struct_path}: got {len(df)}, "
                f"expected {self.expected_size}"
            )

        # 取出特征列，例如 ["x1","y1","z1","m1","x2","y2","z2","m2"]
        for col in self.feature_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in {struct_path}")

        feat = df[self.feature_cols].to_numpy(dtype=np.float32)  # (W, C_feat)
        feat_tensor = torch.from_numpy(feat)  # (W, C_feat)
        return feat_tensor

    # ---------- Dataset 接口 ----------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        hic_path, struct_path = self.samples[idx]

        hic_tensor = self._load_hic_matrix(hic_path)        # (1, W, W)
        struct_seq = self._load_structure_seq(struct_path)  # (W, C_feat)

        sample_id = os.path.splitext(os.path.basename(hic_path))[0]
        struct_name = os.path.basename(struct_path)

        return {
            "hic": hic_tensor,
            "structure": struct_seq,
            "sample_id": sample_id,
            "structure_file": struct_name,
        }


# ---------- 简单的 collate_fn（直接 stack） ----------

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    batch: list of dict，来自 HiCStructureDataset.__getitem__
    输出：
        hic:        (B, 1, W, W)
        structure:  (B, W, C_feat)
        sample_id:  list[str]
        structure_file: list[str]
    """
    hic_list = [item["hic"] for item in batch]
    struct_list = [item["structure"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    struct_files = [item["structure_file"] for item in batch]

    hic = torch.stack(hic_list, dim=0)          # (B, 1, W, W)
    structure = torch.stack(struct_list, dim=0) # (B, W, C_feat)

    return {
        "hic": hic,
        "structure": structure,
        "sample_id": sample_ids,
        "structure_file": struct_files,
    }


# ---------- 使用示例 ----------

if __name__ == "__main__":
    dataset = HiCStructureDataset(
        root_dir="data",
        hic_dirname="Hi-C",
        struct_dirname="structure",
        expected_size=928,  # Hi-C 矩阵大小 & 序列长度
        # 如果你的 tsv 里列名不一样，可以在这里改：
        feature_cols=["x1", "y1", "z1", "m1", "x2", "y2", "z2", "m2"],
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,   # Windows 下建议先用 0，Linux 可以调大
        collate_fn=collate_fn,
    )

    for batch in loader:
        hic = batch["hic"]              # (B, 1, 928, 928)
        structure = batch["structure"]  # (B, 928, 8)
        print("Hi-C batch shape:", hic.shape)
        print("Structure batch shape:", structure.shape)
        print("Sample IDs:", batch["sample_id"])
        break