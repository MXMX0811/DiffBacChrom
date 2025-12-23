import os
import glob
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyvista as pv
import torch

COLORMAP = "Spectral"

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.resnet_vae import StructureAutoencoderKL1D  # noqa: E402
from data.transforms import center_batch, scale_batch  # noqa: E402


def extract_chain_coords(struct_tensor: torch.Tensor, chain: str) -> Tuple[List[float], List[float], List[float]]:
    """
    struct_tensor: (W,16) normalized
    chain: "orig" or "copy"
    - orig uses x1/y1/z1/m1 from both rows (indices 0-3 and 8-11)
    - copy uses x2/y2/z2/m2 from both rows (indices 4-7 and 12-15)
    Returns lists of x,y,z with ordering along sequence, only where mask==1.
    """
    coords_x, coords_y, coords_z = [], [], []
    if chain == "orig":
        starts = [0, 8]
    else:  # "copy"
        starts = [4, 12]

    for token in struct_tensor:
        for start in starts:
            x, y, z, m = token[start:start + 4]
            if m > 0.5:
                coords_x.append(float(x))
                coords_y.append(float(y))
                coords_z.append(float(z))
    return coords_x, coords_y, coords_z


def collect_chain_points(struct_tensor: torch.Tensor, chain: str, step: int | None = None) -> Tuple[List[List[float]], List[int]]:
    """Return ordered list of points (and their token indices) for a chain; optional subsample every `step` tokens."""
    points = []
    indices = []
    if chain == "orig":
        starts = [0, 8]
    else:
        starts = [4, 12]
    for idx, token in enumerate(struct_tensor):
        if step is not None and (idx % step) != 0:
            continue
        for start in starts:
            x, y, z, m = token[start:start + 4]
            if m > 0.5:
                points.append([float(x), float(y), float(z)])
                indices.append(idx)
    return points, indices


def add_polyline(
    pl: pv.Plotter,
    points: np.ndarray,
    indices: np.ndarray | None = None,
    width: float = 2.0,
    clim: tuple[float, float] | None = None,
    cmap: str = COLORMAP,
    color: str | None = None,
    **kwargs,
):
    """
    Draw a tubular curve through the given points.
    - points: (N,3) array
    - indices: (N,) array of hic indices for coloring (if provided)
    - width: roughly controls tube diameter (radius ~ width*0.1)
    """
    if points.shape[0] < 2:
        return
    cells = np.hstack(([points.shape[0]], np.arange(points.shape[0], dtype=np.int32))).astype(np.int32)
    poly = pv.PolyData(points, lines=cells)
    if indices is not None:
        poly["idx"] = indices.astype(np.float32)
        tube = poly.tube(radius=width * 0.1, n_sides=24)
        pl.add_mesh(tube, scalars="idx", cmap=cmap, clim=clim, smooth_shading=True, show_scalar_bar=False)
    else:
        tube = poly.tube(radius=width * 0.1, n_sides=24)
        pl.add_mesh(tube, color=color or "#4682B4", smooth_shading=True, show_scalar_bar=False)


def plot_samples(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    save_path: str,
    names: List[str] | None = None,
    num_show: int = 5,
    random_select: bool = True,
):
    """
    originals/reconstructions: (N, W, 16) normalized.
    Randomly selects up to num_show samples to visualize.
    4-row grid: original full points, reconstruction full points, original full lines, reconstruction full lines.
    """
    def set_camera_for_points(pl: pv.Plotter, pts: np.ndarray, shrink: float = 0.9):
        if pts.size == 0:
            return
        center = pts.mean(axis=0)
        max_range = np.ptp(pts, axis=0).max()
        max_range = max(max_range, 1e-3)
        offset = max_range * 2.0
        pl.camera.focal_point = center
        pl.camera.position = center + np.array([offset, offset, offset])
        pl.camera.up = (0, 0, 1)
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = (max_range * 0.5) * shrink

    total = originals.size(0)
    num_show = min(num_show, total)
    if random_select:
        sel_idx = torch.randperm(total)[:num_show]
    else:
        sel_idx = torch.arange(num_show)
    originals = originals[sel_idx]
    reconstructions = reconstructions[sel_idx]
    names = [names[i] for i in sel_idx] if names else None
    max_idx = max(0, originals.shape[1] - 1)
    clim = (0.0, float(max_idx))

    plotter = pv.Plotter(shape=(4, num_show), window_size=(num_show * 520, 1200), off_screen=True)

    for i in range(num_show):
        title = names[i] if names else ""

        # row 0: original full points
        plotter.subplot(0, i)
        if title:
            plotter.add_text(title, position=(plotter.window_size[0] / num_show * (i + 0.5), plotter.window_size[1] - 20), font_size=18, color="black")
        pts_accum = []
        for chain in ["orig", "copy"]:
            pts, idxs = collect_chain_points(originals[i], chain)
            if len(pts) == 0:
                continue
            pts_arr = np.asarray(pts)
            idx_arr = np.asarray(idxs)
            pts_accum.append(pts_arr)
            plotter.add_points(
                pts_arr,
                scalars=idx_arr,
                cmap=COLORMAP,
                clim=clim,
                point_size=20,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )
        if pts_accum:
            set_camera_for_points(plotter, np.concatenate(pts_accum, axis=0), shrink=0.9)
        plotter.remove_bounds_axes()

        # row 1: reconstruction full points
        plotter.subplot(1, i)
        pts_accum = []
        for chain in ["orig", "copy"]:
            pts, idxs = collect_chain_points(reconstructions[i], chain)
            if len(pts) == 0:
                continue
            pts_arr = np.asarray(pts)
            idx_arr = np.asarray(idxs)
            pts_accum.append(pts_arr)
            plotter.add_points(
                pts_arr,
                scalars=idx_arr,
                cmap=COLORMAP,
                clim=clim,
                point_size=20,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )
        if pts_accum:
            set_camera_for_points(plotter, np.concatenate(pts_accum, axis=0), shrink=0.9)
        plotter.remove_bounds_axes()

        # row 2: original full lines (no points)
        plotter.subplot(2, i)
        all_pts = []
        for chain in ["orig", "copy"]:
            pts, idxs = collect_chain_points(originals[i], chain)
            if len(pts) == 0:
                continue
            pts_arr = np.asarray(pts)
            idx_arr = np.asarray(idxs, dtype=np.float32)
            all_pts.append(pts_arr)
            add_polyline(plotter, pts_arr, idx_arr, width=0.3, clim=clim, cmap=COLORMAP)
        if all_pts:
            set_camera_for_points(plotter, np.concatenate(all_pts, axis=0), shrink=0.9)
        plotter.remove_bounds_axes()

        # row 3: reconstruction full lines (no points)
        plotter.subplot(3, i)
        all_pts = []
        for chain in ["orig", "copy"]:
            pts, idxs = collect_chain_points(reconstructions[i], chain)
            if len(pts) == 0:
                continue
            pts_arr = np.asarray(pts)
            idx_arr = np.asarray(idxs, dtype=np.float32)
            all_pts.append(pts_arr)
            add_polyline(plotter, pts_arr, idx_arr, width=0.3, clim=clim, cmap=COLORMAP)
        if all_pts:
            set_camera_for_points(plotter, np.concatenate(all_pts, axis=0), shrink=0.9)
        plotter.remove_bounds_axes()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plotter.screenshot(save_path, scale=3.0)  # scale up for ~300 dpi output
    plotter.close()
    print(f"Saved visualization to {save_path}")


def load_struct_tensor(struct_path: str, device: torch.device) -> Tuple[torch.Tensor, pd.DataFrame]:
    df = pd.read_csv(struct_path, sep="\t")
    groups = df.groupby("hic_index", sort=False)
    tokens = []
    for _, g in groups:
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
        token = bead1_feat + bead2_feat
        tokens.append(token)
    struct_tensor = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0).to(device)  # (1,W,16)
    return struct_tensor, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data/test", help="Root directory containing test TSV files")
    parser.add_argument("--ckpt", type=str, default="checkpoints/vae/epoch_040.pt", help="VAE checkpoint path")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory to save plots")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    struct_files = sorted(glob.glob(os.path.join(args.root_dir, "**", "*.tsv"), recursive=True))
    if len(struct_files) == 0:
        print(f"No TSV files found under {args.root_dir}")
        return

    vae = StructureAutoencoderKL1D().to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    originals = []
    reconstructions = []
    names = []

    for struct_path in struct_files:
        struct_tensor, _ = load_struct_tensor(struct_path, device)
        struct_centered, _ = center_batch(struct_tensor)
        struct_norm, _ = scale_batch(struct_centered)
        with torch.no_grad():
            z = vae.encode(struct_norm).latent_dist.sample()
            recon = vae.decode(z)  # normalized
        originals.append(struct_norm.cpu())
        reconstructions.append(recon.cpu())
        names.append(os.path.splitext(os.path.basename(struct_path))[0])

    if len(originals) == 0:
        print("No samples to visualize.")
        return

    orig_cat = torch.cat(originals, dim=0)
    recon_cat = torch.cat(reconstructions, dim=0)

    os.makedirs(args.outputs_dir, exist_ok=True)
    plot_samples(
        orig_cat,
        recon_cat,
        save_path=os.path.join(args.outputs_dir, "recon_vis.png"),
        names=names,
        num_show=args.num_samples,
        random_select=True,
    )


if __name__ == "__main__":
    main()
