"""
fuse_tsdf.py
LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, CPU-TESTED against a
synthetic analytic scene (tests/test_fuse_tsdf.py -- pure numpy ray-sphere depth,
no rasterizer needed), NOT yet run against real rendered-from-Gaussians depth
(that needs render_multiview_depth.py's CUDA rasterizer pass first -- see that
module's docstring for exactly what has and hasn't been executed).

WHY TSDF FUSION INSTEAD OF (OR ALONGSIDE) POISSON ON GAUSSIAN CENTERS: see
docs/MESH_QUALITY_AUDIT.md for the full reasoning; short version --
mesh_export.py's Poisson path treats every surviving Gaussian center as an
equally-real surface sample and tries to fit ONE smooth, closed-ish implicit
surface through them. A single-pass aerial flight observes a fundamentally
OPEN scene (no undersides of roofs, no far sides of buildings, genuine holes
wherever nothing was ever seen) -- forcing that into a closed-surface solver
is a mismatch independent of how good the input normals are (confirmed
empirically in the audit: fixing normals alone did not fix that checkpoint's
mesh). TSDF fusion instead asks, per voxel, "did multiple DIFFERENT views'
depth estimates agree this voxel is just past empty space, right at a
surface?" -- voxels no view ever looked at stay unknown and are never
meshed, so gaps show up as genuine holes instead of being smoothed over.
This is the same principle KinectFusion / most real photogrammetry depth-
fusion pipelines use, applied here to depth RENDERED FROM the trained
Gaussians (via render_multiview_depth.py) rather than depth sensed by a
physical depth camera.

This module only does the fusion + mesh extraction. It does not care where
the depth maps came from -- Gaussian-rendered (render_multiview_depth.py) or
the CPU sandbox's own plane-sweep MVS (depth_estimation.py) would both work,
as long as you supply consistent (depth, color, alpha/validity, camera pose,
intrinsics) per view.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np


@dataclasses.dataclass
class TSDFFusionStats:
    n_views_integrated: int
    n_views_skipped_empty: int
    voxel_size: float
    sdf_trunc: float
    n_vertices: int
    n_triangles: int
    bbox_min: tuple
    bbox_max: tuple


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError("fuse_tsdf needs the 'open3d' package -- see requirements.txt.") from e
    return o3d


def fuse_depth_maps_tsdf(depth_maps: List[np.ndarray], color_maps: List[np.ndarray],
                          R_wc_list: List[np.ndarray], position_list: List[np.ndarray],
                          fx: float, fy: float, cx: float, cy: float,
                          alpha_maps: Optional[List[np.ndarray]] = None, min_alpha: float = 0.5,
                          voxel_size: float = 0.05, sdf_trunc: Optional[float] = None,
                          depth_trunc: float = 200.0):
    """
    Fuses a set of per-view depth maps (world-metric, e.g. rendered by
    render_multiview_depth.py or produced by depth_estimation.py) into one
    mesh via TSDF integration. Returns (mesh, stats).

    depth_maps: list of (H,W) float arrays, world units, 0/NaN = invalid/unresolved.
    color_maps: list of (H,W,3) arrays, either [0,1] or [0,255] (auto-detected
        from the first frame).
    R_wc_list, position_list: per-view world_from_camera rotation (3,3) and camera
        position (3,) -- same convention as this project's CameraPose (R_wc rotates
        CAMERA-frame directions into WORLD frame; see camera_model.py).
    fx, fy, cx, cy: pinhole intrinsics. Each may be EITHER a single float, shared across every
        view (this project's original single-camera-per-flight assumption, e.g.
        RealDroneDataset.K), OR a sequence of length n_views, one value per view -- needed the
        moment any views come from a zoom lens with per-frame intrinsics (see
        real_dataset_builder.py's per-frame K / build_cameras_from_real). Mixing -- some scalar,
        some per-view -- is fine; each is broadcast independently.
    alpha_maps: optional list of (H,W) arrays in [0,1], coverage/accumulated-alpha
        per pixel (e.g. from render_multiview_depth.py). Pixels with alpha <
        min_alpha are treated as invalid and excluded from integration -- this is
        the "respect visibility, don't invent surface where nothing was
        confidently seen" mechanism the caller asked for. None = trust all
        depth>0 pixels.
    voxel_size: TSDF voxel edge length, WORLD UNITS. Start near the scene's
        typical Gaussian scale (see docs/MESH_QUALITY_AUDIT.md -- this project's
        real checkpoint had median Gaussian scale ~0.18); too fine relative to
        real depth-map noise reproduces the same over-fitting-to-noise problem
        raised in the audit for Poisson depth, just in a different algorithm.
    sdf_trunc: truncation distance for the signed distance function; default
        4*voxel_size (Open3D's own examples' usual ratio) if not given.
    depth_trunc: depth values beyond this are ignored (matches
        depth_estimation.py's depth_max semantics -- keep them consistent if
        you're fusing depth from that module).
    """
    o3d = _require_open3d()
    if sdf_trunc is None:
        sdf_trunc = 4.0 * voxel_size

    n_views = len(depth_maps)
    if not (len(color_maps) == len(R_wc_list) == len(position_list) == n_views):
        raise ValueError("depth_maps, color_maps, R_wc_list, position_list must all have the same length")
    if n_views == 0:
        raise ValueError("need at least one view to fuse")

    def _per_view(v, name):
        """Broadcast a scalar to n_views, or pass through a length-n_views sequence."""
        if np.isscalar(v):
            return [float(v)] * n_views
        v = list(v)
        if len(v) != n_views:
            raise ValueError(f"{name} has length {len(v)}, expected a scalar or a length-{n_views} sequence")
        return [float(x) for x in v]

    fx_list, fy_list, cx_list, cy_list = _per_view(fx, "fx"), _per_view(fy, "fy"), \
        _per_view(cx, "cx"), _per_view(cy, "cy")

    H, W = depth_maps[0].shape

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    color0 = np.asarray(color_maps[0])
    color_scale = 255.0 if color0.size and color0.max() <= 1.0001 else 1.0

    n_skipped = 0
    for i in range(n_views):
        depth = np.asarray(depth_maps[i], dtype=np.float32).copy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if alpha_maps is not None:
            depth[np.asarray(alpha_maps[i]) < min_alpha] = 0.0
        if not np.any(depth > 0):
            n_skipped += 1
            continue

        color = np.asarray(color_maps[i]) * color_scale
        color = np.clip(color, 0, 255).astype(np.uint8)

        color_img = o3d.geometry.Image(np.ascontiguousarray(color))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

        R_wc = np.asarray(R_wc_list[i], dtype=np.float64)
        position = np.asarray(position_list[i], dtype=np.float64)
        R_cw = R_wc.T
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_cw
        extrinsic[:3, 3] = -R_cw @ position

        # Built PER VIEW (not hoisted out of the loop) so a mix of zoom levels across views
        # integrates each one with its own correct intrinsics instead of silently reusing
        # frame 0's.
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx_list[i], fy_list[i], cx_list[i], cy_list[i])
        volume.integrate(rgbd, intrinsic, extrinsic)

    if n_views - n_skipped == 0:
        raise ValueError(f"all {n_views} views had zero valid depth pixels (after alpha_maps/min_alpha "
                          f"filtering) -- nothing to integrate. Lower min_alpha or check the depth maps.")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    V = np.asarray(mesh.vertices)
    bbox_min = tuple(V.min(0).tolist()) if len(V) else (0., 0., 0.)
    bbox_max = tuple(V.max(0).tolist()) if len(V) else (0., 0., 0.)
    stats = TSDFFusionStats(n_views_integrated=n_views - n_skipped, n_views_skipped_empty=n_skipped,
                             voxel_size=voxel_size, sdf_trunc=sdf_trunc, n_vertices=len(mesh.vertices),
                             n_triangles=len(mesh.triangles), bbox_min=bbox_min, bbox_max=bbox_max)
    return mesh, stats
