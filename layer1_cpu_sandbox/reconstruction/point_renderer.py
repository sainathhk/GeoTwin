"""
point_renderer.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested (EVALUATION-SIDE UTILITY)

A minimal, NON-differentiable point-splat renderer used only so the
evaluation framework can compute image-space fidelity metrics (PSNR/SSIM)
for the reconstructed point cloud from a chosen viewpoint. It is not part
of the reconstruction pipeline and is not meant to resemble the real
renderer: in Layer 3, this entire concept is replaced by the differentiable
`diff-gaussian-rasterization` tile-based rasterizer (see
layer3_gpu/python/confidence_gaussian_model.py), which is what actually
produces publication-quality PSNR/SSIM/LPIPS numbers. Point-splat renders
from a sparse, several-meter-noisy point cloud are expected to look
blocky/holey -- that is the CPU prototype's representation limit, not a
bug, and is exactly why the report separates "Layer-1 sandbox validation
numbers" from "final system performance" throughout.
"""
from __future__ import annotations

import numpy as np
import cv2

from ..synthetic.camera_model import Intrinsics, project_points


def render_point_cloud(positions: np.ndarray, colors: np.ndarray, K: Intrinsics, pose,
                        splat_radius_px: int = 2):
    H, W = K.height, K.width
    if positions.shape[0] == 0:
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=bool)

    cam_pts = pose.world_to_cam(positions)
    uv, z = project_points(K.K(), cam_pts)
    valid = (z > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not np.any(valid):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=bool)

    uv_v, z_v, col_v = uv[valid], z[valid], colors[valid]
    order = np.argsort(z_v)  # nearest-first so first-write-wins == a proper z-buffer

    img = np.zeros((H, W, 3), dtype=np.float32)
    written = np.zeros((H, W), dtype=bool)
    us = uv_v[order, 0].astype(np.int32)
    vs = uv_v[order, 1].astype(np.int32)
    cols = col_v[order]

    for u, v, c in zip(us, vs, cols):
        if not written[v, u]:
            img[v, u] = c
            written[v, u] = True

    if splat_radius_px > 0:
        mask_u8 = (written.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * splat_radius_px + 1,) * 2)
        dilated_mask = cv2.dilate(mask_u8, kernel) > 0
        # Dilate color via a max-filter-like trick: dilate each channel weighted by the
        # original mask, giving nearest-written-pixel colour to newly covered neighbours.
        dilated_img = cv2.dilate((img * 255).astype(np.uint8), kernel)
        img_out = np.where(dilated_mask[..., None], dilated_img, 0).astype(np.uint8)
        return img_out, dilated_mask

    return (img * 255).clip(0, 255).astype(np.uint8), written
