"""
prototype_point_repr.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

`PrototypePointRepresentation` -- the CPU-sandbox stand-in for the final
3-D representation. This is explicitly a simplified point cloud, NOT 3D
Gaussian Splatting; see layer3_gpu/ for the real (CUDA, unexecuted-here)
Gaussian representation. Renaming this to sound like "our 3DGS system"
would misrepresent what actually ran in Environment A.

What this module DOES do that is genuinely representation-agnostic, and
therefore directly reusable by the Layer-3 Gaussian pipeline: it fuses raw
per-frame backprojected depth into voxel-merged 3-D points and, for every
merged point, keeps the full list of contributing observations (which
frame, at what viewing angle, with what depth-consistency, frame quality,
and pose confidence). That per-point observation ledger is exactly the
input the confidence framework (confidence.py) needs, and in the GPU
system the identical ledger is accumulated per-GAUSSIAN instead of
per-point (see layer3_gpu/csrc/confidence_gaussian).
"""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np
import pandas as pd

from ..synthetic.camera_model import Intrinsics


@dataclasses.dataclass
class PrototypePointRepresentation:
    positions: np.ndarray          # (M,3) fused world-space positions
    colors: np.ndarray             # (M,3) in [0,1]
    observation_count: np.ndarray  # (M,) distinct frames that observed this point
    view_angle_min: np.ndarray     # (M,) radians
    view_angle_max: np.ndarray     # (M,)
    view_angle_spread: np.ndarray  # (M,) max-min, a triangulation-baseline-angle proxy
    mean_consistency: np.ndarray   # (M,) mean plane-sweep cost-curve consistency of contributing pixels
    mean_frame_quality: np.ndarray  # (M,)
    mean_pose_confidence: np.ndarray  # (M,)
    n_raw_observations: np.ndarray  # (M,) total contributing pixel-observations (>= observation_count)
    voxel_size: float


def _pixel_view_angles(H: int, W: int, K: Intrinsics) -> np.ndarray:
    """Angle (rad) between each pixel's ray and the optical axis -- depends only on
    pixel location + intrinsics, not on depth (see derivation in module docstring below)."""
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    rx = (us - K.cx) / K.fx
    ry = (vs - K.cy) / K.fy
    return np.arccos(1.0 / np.sqrt(rx ** 2 + ry ** 2 + 1.0)).astype(np.float32)


def fuse_point_cloud(depth_estimates, dynamic_masks, poses, rgb_frames,
                      frame_quality_scores, pose_confidences, K,
                      voxel_size: float = 0.4, min_consistency: float = 0.0) -> PrototypePointRepresentation:
    """
    depth_estimates: list[DepthEstimate] (one per kept frame, same order as poses/rgb_frames)
    dynamic_masks: list[DynamicMaskResult or None] (same order; None => no dynamic pixels excluded)
    poses: list[CameraPose] (ESTIMATED poses, e.g. from sensor_fusion.FusedTrajectory)
    frame_quality_scores: list[float] overall quality score in [0,1] per frame
    pose_confidences: list[float] in [0,1] per frame (e.g. FusedTrajectory.pose_confidence)
    K: a single Intrinsics shared by every frame (old behavior), OR a list of per-frame
        Intrinsics the same length as depth_estimates. 2026-09-21 fix: this function used to
        unproject EVERY frame's depth map into world-space points using ONE shared ray-direction
        grid (built from a single K, outside the per-frame loop) -- silently wrong the moment any
        frame's real focal length differs from that K, which real_pipeline.py's own per-frame K
        (see real_dataset_builder.py) had already made routine. depth_estimation.py's plane-sweep
        can compute a perfectly correct per-frame depth VALUE and this function would still place
        it at the wrong 3-D position, because "distance d along ray R_true" is not "distance d
        along ray R_wrong" -- a second, independent bug from the depth-estimation one, not a
        duplicate of it. real_pipeline.py now passes the same per-frame list to both.
    """
    H, W = depth_estimates[0].depth.shape
    n = len(depth_estimates)
    Ks = list(K) if isinstance(K, (list, tuple)) else [K] * n
    if len(Ks) != n:
        raise ValueError(f"K list has length {len(Ks)}, expected {n} (one per depth_estimates)")
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    # Cache by identity -- a real capture typically has only a handful of distinct K's (one per
    # distinct focal-length sample), not one per frame, so this is a small cache in practice.
    _ray_cache = {}

    def _rays_and_angles(K_i):
        key = id(K_i)
        if key not in _ray_cache:
            rx = (us - K_i.cx) / K_i.fx
            ry = (vs - K_i.cy) / K_i.fy
            angle = np.arccos(1.0 / np.sqrt(rx ** 2 + ry ** 2 + 1.0)).astype(np.float32)
            _ray_cache[key] = (rx, ry, angle)
        return _ray_cache[key]

    rows = []
    for i, (est, pose) in enumerate(zip(depth_estimates, poses)):
        ray_x, ray_y, view_angle_grid = _rays_and_angles(Ks[i])
        depth = est.depth
        valid = depth > 0
        if min_consistency > 0:
            valid &= (est.consistency >= min_consistency)
        if dynamic_masks is not None and dynamic_masks[i] is not None:
            valid &= ~dynamic_masks[i].mask
        if not np.any(valid):
            continue

        d = depth[valid]
        cam_pts = np.stack([ray_x[valid] * d, ray_y[valid] * d, d], axis=-1)
        world_pts = cam_pts @ pose.R_cw() + pose.position
        colors = rgb_frames[i][valid].astype(np.float32) / 255.0

        n_pts = world_pts.shape[0]
        df = pd.DataFrame({
            "x": world_pts[:, 0], "y": world_pts[:, 1], "z": world_pts[:, 2],
            "r": colors[:, 0], "g": colors[:, 1], "b": colors[:, 2],
            "frame_idx": np.full(n_pts, i),
            "view_angle": view_angle_grid[valid],
            "consistency": est.consistency[valid],
            "frame_quality": np.full(n_pts, frame_quality_scores[i]),
            "pose_confidence": np.full(n_pts, pose_confidences[i]),
        })
        rows.append(df)

    if not rows:
        empty = np.zeros((0,), dtype=np.float32)
        return PrototypePointRepresentation(np.zeros((0, 3)), np.zeros((0, 3)), empty, empty, empty,
                                             empty, empty, empty, empty, empty, voxel_size)

    all_pts = pd.concat(rows, ignore_index=True)
    all_pts["vx"] = np.floor(all_pts["x"] / voxel_size).astype(np.int64)
    all_pts["vy"] = np.floor(all_pts["y"] / voxel_size).astype(np.int64)
    all_pts["vz"] = np.floor(all_pts["z"] / voxel_size).astype(np.int64)

    g = all_pts.groupby(["vx", "vy", "vz"], sort=False)
    agg = g.agg(
        x=("x", "mean"), y=("y", "mean"), z=("z", "mean"),
        r=("r", "mean"), g_=("g", "mean"), b=("b", "mean"),
        observation_count=("frame_idx", "nunique"),
        view_angle_min=("view_angle", "min"), view_angle_max=("view_angle", "max"),
        mean_consistency=("consistency", "mean"),
        mean_frame_quality=("frame_quality", "mean"),
        mean_pose_confidence=("pose_confidence", "mean"),
        n_raw_observations=("frame_idx", "size"),
    ).reset_index(drop=True)

    positions = agg[["x", "y", "z"]].to_numpy(dtype=np.float32)
    colors = agg[["r", "g_", "b"]].to_numpy(dtype=np.float32)
    view_angle_spread = (agg["view_angle_max"] - agg["view_angle_min"]).to_numpy(dtype=np.float32)

    return PrototypePointRepresentation(
        positions=positions, colors=colors,
        observation_count=agg["observation_count"].to_numpy(dtype=np.int32),
        view_angle_min=agg["view_angle_min"].to_numpy(dtype=np.float32),
        view_angle_max=agg["view_angle_max"].to_numpy(dtype=np.float32),
        view_angle_spread=view_angle_spread,
        mean_consistency=agg["mean_consistency"].to_numpy(dtype=np.float32),
        mean_frame_quality=agg["mean_frame_quality"].to_numpy(dtype=np.float32),
        mean_pose_confidence=agg["mean_pose_confidence"].to_numpy(dtype=np.float32),
        n_raw_observations=agg["n_raw_observations"].to_numpy(dtype=np.int32),
        voxel_size=voxel_size,
    )
