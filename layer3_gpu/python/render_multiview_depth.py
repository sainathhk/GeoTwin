"""
render_multiview_depth.py
LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, NOT executed (needs
CUDA + `diff_gaussian_rasterization` + real camera poses; none were available in
the environment this was written in -- see docs/MESH_QUALITY_AUDIT.md). Run this
on Environment B and sanity-check a few output depth maps (e.g. as a matplotlib
colormap image, or by comparing against depth_estimation.py's plane-sweep output
for the same frame) BEFORE trusting it in a production run -- the same way every
other GPU-only module in this project is flagged "implemented, NOT GPU-executed"
until someone with a GPU confirms it, this one gets the same honest label.

WHAT THIS DOES: renders a per-pixel DEPTH map and ALPHA (accumulated-coverage)
map from the trained Gaussians, from a chosen set of `TrainingCamera`s -- the
"render depth maps, render alpha/visibility" step docs/MESH_QUALITY_AUDIT.md's
recommended pipeline calls for, upstream of fuse_tsdf.py.

HOW, WITHOUT A NEW CUDA KERNEL: reuses train_gpu.py's EXISTING
`_camera_to_rasterizer_settings` / `GaussianRasterizer` call, unmodified --
this is the same "don't reimplement the tile rasterizer" principle
mesh_export.py's docstring states explicitly. The trick (standard in the 3DGS
ecosystem -- see e.g. the reference implementation's own depth-visualization
scripts, and any paper that adds depth supervision to vanilla 3DGS without a
custom CUDA build): the rasterizer's `colors_precomp` argument is alpha-
composited through per-Gaussian visibility EXACTLY like SH color is, using
the same tile-based order-dependent compositing -- it has no idea whether the
values it's blending represent color or something else. Passing each
Gaussian's camera-space depth as `colors_precomp` (instead of SH color)
yields an alpha-weighted depth composite using the SAME correct-by-
construction blending the RGB image itself is rendered with. A second output
channel filled with a constant across all Gaussians, composited by that same
call, gives the accumulated alpha (coverage) needed to (a) turn the alpha-
weighted depth back into a true expected depth (divide one by the other) and
(b) tell fuse_tsdf.py which pixels were confidently observed at all.

WHY THIS MATTERS FOR THE OPEN-SCENE PROBLEM (see docs/MESH_QUALITY_AUDIT.md):
mesh_export.py's Poisson path has no notion of "this region was never
observed" -- it interpolates everywhere between confident points regardless.
The alpha map here is exactly that missing signal: near-zero alpha marks
"nothing was rendered here" (sky, occluded, or simply outside every trained
Gaussian's support), and fuse_tsdf.py is built to treat those pixels as
"don't know", not "smooth surface".
"""
from __future__ import annotations

import dataclasses
import os
from typing import List, Optional

import numpy as np
import torch

from gaussian_model import GaussianModel
from train_gpu import TrainingCamera, _camera_to_rasterizer_settings


@dataclasses.dataclass
class RenderedDepthView:
    frame_idx: int
    depth: np.ndarray   # (H,W) float32, world units, 0.0 = unresolved/no coverage
    alpha: np.ndarray   # (H,W) float32 in [0,1], accumulated Gaussian coverage
    color: np.ndarray   # (H,W,3) float32 in [0,1], the ordinary RGB render (for TSDF's color channel)
    R_wc: np.ndarray    # (3,3) camera pose at render time (for fuse_tsdf.py)
    position: np.ndarray  # (3,)
    fx: float
    fy: float
    cx: float
    cy: float


def render_gaussian_depth_alpha(model: GaussianModel, cam: TrainingCamera, device: str,
                                 active_sh_degree: Optional[int] = None, alpha_valid_thresh: float = 0.05):
    """
    Renders (depth, alpha, color) for one camera. depth/alpha are (H,W) torch
    tensors on `device`; color is the ordinary RGB render (separate rasterizer
    call, since it needs the real SH evaluation, not a colors_precomp override).

    depth is TRUE camera-space Z (distance along the optical axis, matching this
    project's `depth_estimation.py` convention), not straight-line distance to
    the camera -- consistent with what fuse_tsdf.py / Open3D's RGBD integration
    expect (pinhole-model "depth image" convention).

    Pixels with accumulated alpha below `alpha_valid_thresh` are zeroed in the
    returned depth (treated as unresolved) -- a pixel with almost no Gaussian
    coverage produced an almost-meaningless depth/alpha ratio (dividing two
    near-zero numbers), not a trustworthy thin/distant surface.
    """
    from diff_gaussian_rasterization import GaussianRasterizer

    settings = _camera_to_rasterizer_settings(cam, device, bg_color=(0.0, 0.0, 0.0), sh_degree=0)
    rasterizer = GaussianRasterizer(raster_settings=settings)
    screenspace_points = torch.zeros_like(model.xyz, requires_grad=False, device=device)

    R_wc = torch.from_numpy(cam.pose.R_wc).float().to(device)
    cam_pos = torch.from_numpy(cam.pose.position).float().to(device)
    rel = model.xyz.detach() - cam_pos
    # matches synthetic/camera_model.py's CameraPose.world_to_cam: `rel @ R_wc` -- see
    # that method's docstring for why this (not `rel @ R_wc.T`) is the world->camera map.
    cam_space = rel @ R_wc
    depth_per_gaussian = cam_space[:, 2].clamp(min=0.0)

    ones = torch.ones_like(depth_per_gaussian)
    colors_precomp = torch.stack([depth_per_gaussian, ones, ones], dim=1)  # [depth, 1, 1] per Gaussian

    rendered, radii = rasterizer(
        means3D=model.xyz, means2D=screenspace_points, shs=None, colors_precomp=colors_precomp,
        opacities=model.get_opacity(), scales=model.get_scaling(), rotations=model.get_rotation(),
        cov3D_precomp=None,
    )
    # channel 0 = alpha-weighted depth composite; channels 1,2 = accumulated alpha twice over
    # (both filled with the SAME per-Gaussian constant 1.0, so they're identical by construction --
    # this is not measuring two different things, it's using the compositing math to get one one
    # extra channel of "the same alpha" for free instead of a second rasterizer call).
    depth_composited = rendered[0]
    alpha_map = rendered[1]
    valid = alpha_map > alpha_valid_thresh
    depth_map = torch.where(valid, depth_composited / alpha_map.clamp(min=1e-6), torch.zeros_like(depth_composited))

    degree = model.cfg.sh_degree if active_sh_degree is None else min(active_sh_degree, model.cfg.sh_degree)
    color_settings = _camera_to_rasterizer_settings(cam, device, bg_color=(0.0, 0.0, 0.0), sh_degree=degree)
    color_rasterizer = GaussianRasterizer(raster_settings=color_settings)
    color_rendered, _ = color_rasterizer(
        means3D=model.xyz, means2D=screenspace_points, shs=model.get_features(), colors_precomp=None,
        opacities=model.get_opacity(), scales=model.get_scaling(), rotations=model.get_rotation(),
        cov3D_precomp=None,
    )

    return depth_map, alpha_map, color_rendered


def render_multiview_depth(model: GaussianModel, cameras: List[TrainingCamera], device: str,
                            active_sh_degree: Optional[int] = None, alpha_valid_thresh: float = 0.05,
                            save_dir: Optional[str] = None) -> List[RenderedDepthView]:
    """
    Renders depth+alpha+color for every camera in `cameras` (pass a SUBSET of your
    full training camera list if you don't need all of them -- e.g. every 3rd frame
    is often enough overlap for TSDF fusion and is 3x cheaper). If `save_dir` is
    given, writes `depth_{frame_idx:05d}.npy` / `alpha_{frame_idx:05d}.npy` /
    `color_{frame_idx:05d}.npy` per view plus a `manifest.json` with the camera
    poses/intrinsics -- exactly the "save intermediate outputs so I can debug"
    request this feature exists to satisfy.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    views = []
    manifest = []
    for cam in cameras:
        depth_t, alpha_t, color_t = render_gaussian_depth_alpha(
            model, cam, device, active_sh_degree=active_sh_degree, alpha_valid_thresh=alpha_valid_thresh)
        depth = depth_t.detach().cpu().numpy().astype(np.float32)
        alpha = alpha_t.detach().cpu().numpy().astype(np.float32)
        color = color_t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # (3,H,W) -> (H,W,3)

        view = RenderedDepthView(frame_idx=cam.frame_idx, depth=depth, alpha=alpha, color=color,
                                  R_wc=cam.pose.R_wc.copy(), position=cam.pose.position.copy(),
                                  fx=cam.K.fx, fy=cam.K.fy, cx=cam.K.cx, cy=cam.K.cy)
        views.append(view)

        if save_dir:
            tag = f"{cam.frame_idx:05d}"
            np.save(os.path.join(save_dir, f"depth_{tag}.npy"), depth)
            np.save(os.path.join(save_dir, f"alpha_{tag}.npy"), alpha)
            np.save(os.path.join(save_dir, f"color_{tag}.npy"), color)
            manifest.append(dict(frame_idx=cam.frame_idx, R_wc=cam.pose.R_wc.tolist(),
                                  position=cam.pose.position.tolist(), fx=cam.K.fx, fy=cam.K.fy,
                                  cx=cam.K.cx, cy=cam.K.cy, width=cam.K.width, height=cam.K.height,
                                  valid_fraction=float((alpha > alpha_valid_thresh).mean())))

    if save_dir:
        import json
        with open(os.path.join(save_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {len(views)} depth/alpha/color views + manifest.json to {save_dir}")

    return views
