"""Small shared utilities for the Layer-1 CPU sandbox. STATUS: implemented, CPU-tested."""
from __future__ import annotations
import numpy as np


def save_ply(path: str, points: np.ndarray, colors: np.ndarray = None):
    """Write an ASCII PLY point cloud (viewable in MeshLab / CloudCompare / Colab open3d)."""
    n = points.shape[0]
    has_color = colors is not None
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_color:
            c = np.clip(colors * 255 if colors.max() <= 1.0001 else colors, 0, 255).astype(np.uint8)
            for i in range(n):
                p = points[i]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[i,0]} {c[i,1]} {c[i,2]}\n")
        else:
            for i in range(n):
                p = points[i]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def ensure_dir(path: str):
    import os
    os.makedirs(path, exist_ok=True)
    return path
