import torch
import math

COORD_IDX = [0,1,2, 4,5,6, 8,9,10, 12,13,14]
MASK_IDX  = [3,7,11,15]

def center_batch(struct_batch, eps=1e-8):
    """
    struct_batch: (B, W, 16)
    返回:
      centered_batch: (B, W, 16)
      centroid: (B, 1, 3) 每个样本的全局质心
    """
    device = struct_batch.device
    coord_idx = torch.tensor(COORD_IDX, device=device)
    mask_idx  = torch.tensor(MASK_IDX,  device=device)

    # 取出所有坐标 & mask
    coords = struct_batch[..., coord_idx]      # (B, W, 12)
    masks  = struct_batch[..., mask_idx]      # (B, W, 4)

    # 每个 mask 对应 3 个坐标分量（x,y,z），repeat_interleave 展开
    masks_for_coords = masks.repeat_interleave(3, dim=-1)  # (B, W, 12)
    valid = masks_for_coords > 0.5                         # bool

    # 视为一堆 3D 点：reshape 成 (B, N_beads, 3)
    B, W, C12 = coords.shape
    coords_3d = coords.view(B, W * 4, 3)                   # 4 beads/bin
    valid_1d  = valid.view(B, W * 4, 3)[..., 0]           # (B, W*4)，每个 bead 一个标记

    # 防止全0
    num_valid = valid_1d.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)

    # 对每个样本求质心
    # 先把 invalid 的 coord 置0
    coords_valid = coords_3d * valid_1d.unsqueeze(-1)            # (B, N, 3)
    centroid = coords_valid.sum(dim=1, keepdim=True) / num_valid.unsqueeze(-1)  # (B,1,3)

    # 平移
    coords_centered = coords_3d - centroid  # (B, N, 3)

    # 放回原来的结构
    centered = struct_batch.clone()
    centered[..., coord_idx] = coords_centered.view(B, W, 12)
    return centered, centroid  # centroid 如果以后想反变换可以保存

def scale_batch(centered_batch, eps=1e-8):
    """
    centered_batch: 已经做过中心化的 (B, W, 16)
    返回:
      scaled_batch: (B, W, 16)
      scale: (B, 1, 1) 每个样本的缩放因子
    """
    device = centered_batch.device
    coord_idx = torch.tensor(COORD_IDX, device=device)
    mask_idx  = torch.tensor(MASK_IDX,  device=device)

    coords = centered_batch[..., coord_idx]   # (B, W, 12)
    masks  = centered_batch[..., mask_idx]    # (B, W, 4)
    masks_for_coords = masks.repeat_interleave(3, dim=-1)  # (B, W, 12)
    valid = masks_for_coords > 0.5

    B, W, C12 = coords.shape
    coords_3d = coords.view(B, W * 4, 3)
    valid_1d  = valid.view(B, W * 4, 3)[..., 0]  # (B, N_beads)

    num_valid = valid_1d.sum(dim=1, keepdim=True).clamp(min=1.0)

    # 只对有效点求平均平方半径
    coords_valid = coords_3d * valid_1d.unsqueeze(-1)
    sq_norm = (coords_valid ** 2).sum(dim=-1)               # (B, N)
    mean_sq = sq_norm.sum(dim=1, keepdim=True) / num_valid  # (B,1)
    scale = torch.sqrt(mean_sq).clamp(min=eps)              # (B,1)

    coords_scaled = coords_3d / scale.unsqueeze(-1)         # (B, N, 3)

    scaled = centered_batch.clone()
    scaled[..., coord_idx] = coords_scaled.view(B, W, 12)
    return scaled, scale  # scale 同样可以保存以做反变换

def random_rotation_matrices(batch_size, device):
    """
    生成 batch_size 个随机旋转矩阵 (B,3,3)，在 SO(3) 上均匀分布。
    """
    u1 = torch.rand(batch_size, device=device)
    u2 = torch.rand(batch_size, device=device)
    u3 = torch.rand(batch_size, device=device)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1)     * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1)     * torch.cos(2 * math.pi * u3)
    # 四元数 (q1,q2,q3,q4)

    # 转 3x3 旋转矩阵
    x, y, z, w = q1, q2, q3, q4
    B = batch_size

    R = torch.empty(B, 3, 3, device=device)
    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)
    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - x*w)
    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x*x + y*y)
    return R

def random_rotate_batch(struct_batch):
    """
    struct_batch: (B, W, 16)  应该已经做过 center + scale
    返回: rotated_batch (B, W, 16)
    """
    device = struct_batch.device
    coord_idx = torch.tensor(COORD_IDX, device=device)

    coords = struct_batch[..., coord_idx]      # (B, W, 12)
    B, W, C12 = coords.shape

    # (B, N, 3)
    coords_3d = coords.view(B, W * 4, 3)

    # 生成每个样本一个旋转矩阵
    R = random_rotation_matrices(B, device=device)  # (B,3,3)

    # 应用旋转：coords' = coords @ R^T
    coords_rot = torch.bmm(coords_3d, R.transpose(1, 2))  # (B, N, 3)

    rotated = struct_batch.clone()
    rotated[..., coord_idx] = coords_rot.view(B, W, 12)
    return rotated
