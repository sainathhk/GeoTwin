"""
metrics_geometry.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Section B of the evaluation framework: GEOMETRIC accuracy, in real-world
units (meters). Independent of anything in metrics_visual.py -- a
reconstruction can score well on PSNR while being geometrically wrong
(e.g. a textured but flat facade), and vice versa; that is precisely why
these are kept as separate metric families throughout the report.
"""
from __future__ import annotations

import dataclasses
import numpy as np
from scipy.spatial import cKDTree


@dataclasses.dataclass
class GeometryAccuracyResult:
    chamfer_distance: float          # symmetric mean nearest-neighbor distance (m), lower better
    point_to_point_rmse: float       # reconstruction -> GT nearest-neighbor RMSE (m)
    point_to_plane_rmse: float       # reconstruction -> GT nearest-neighbor, projected onto GT normal (m)
    mean_nn_distance_recon_to_gt: float
    mean_nn_distance_gt_to_recon: float
    n_recon_points: int
    n_gt_points: int


def evaluate_point_cloud_geometry(recon_pts: np.ndarray, gt_pts: np.ndarray,
                                   gt_normals: np.ndarray = None) -> GeometryAccuracyResult:
    if recon_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        nan = float("nan")
        return GeometryAccuracyResult(nan, nan, nan, nan, nan, recon_pts.shape[0], gt_pts.shape[0])

    tree_gt = cKDTree(gt_pts)
    tree_recon = cKDTree(recon_pts)

    d_r2g, idx_r2g = tree_gt.query(recon_pts, k=1)
    d_g2r, _ = tree_recon.query(gt_pts, k=1)

    chamfer = 0.5 * (float(np.mean(d_r2g)) + float(np.mean(d_g2r)))
    p2p_rmse = float(np.sqrt(np.mean(d_r2g ** 2)))

    if gt_normals is not None:
        nn_normals = gt_normals[idx_r2g]
        vec = recon_pts - gt_pts[idx_r2g]
        signed = np.einsum("ij,ij->i", vec, nn_normals)
        p2plane_rmse = float(np.sqrt(np.mean(signed ** 2)))
    else:
        p2plane_rmse = float("nan")

    return GeometryAccuracyResult(chamfer_distance=chamfer, point_to_point_rmse=p2p_rmse,
                                   point_to_plane_rmse=p2plane_rmse,
                                   mean_nn_distance_recon_to_gt=float(np.mean(d_r2g)),
                                   mean_nn_distance_gt_to_recon=float(np.mean(d_g2r)),
                                   n_recon_points=int(recon_pts.shape[0]), n_gt_points=int(gt_pts.shape[0]))


@dataclasses.dataclass
class DepthAccuracyResult:
    depth_rmse: float
    depth_mae: float
    relative_depth_error: float   # MAE / mean GT depth, unitless
    n_valid_px: int


def evaluate_depth_accuracy(depth_est: np.ndarray, depth_gt: np.ndarray) -> DepthAccuracyResult:
    valid = (depth_est > 0) & (depth_gt > 0)
    if valid.sum() == 0:
        return DepthAccuracyResult(float("nan"), float("nan"), float("nan"), 0)
    diff = depth_est[valid] - depth_gt[valid]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    rel = float(mae / max(np.mean(depth_gt[valid]), 1e-6))
    return DepthAccuracyResult(depth_rmse=rmse, depth_mae=mae, relative_depth_error=rel,
                                n_valid_px=int(valid.sum()))
