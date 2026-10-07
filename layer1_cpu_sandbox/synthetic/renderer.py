"""
renderer.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

A real (if deliberately simple) software Z-buffer rasterizer. It exists so
that "ground truth" in this sandbox is actually ground truth: RGB frames,
per-pixel depth, per-pixel semantics and per-triangle visibility/viewing-angle
statistics are all produced by rasterizing the *same* known scene geometry
the reconstruction pipeline is trying to recover -- nothing is hand-authored
or faked.

Known, explicitly accepted simplifications (documented, not hidden):
  * Flat (Lambertian) shading, no shadows/global illumination.
  * No near-plane clipping: a triangle with any vertex behind the near
    plane is dropped whole for that frame (locally reduces coverage right
    under the flight path; negligible at typical survey altitude).
  * Fully opaque triangles, standard painter's-algorithm-via-Z-buffer.
These do not affect the validity of the reconstruction/evaluation logic
that consumes the renderer's output -- they only affect visual realism.
"""
from __future__ import annotations

import dataclasses
from typing import List

import cv2
import numpy as np

from .camera_model import Intrinsics, CameraPose, project_points
from .scene_generator import SyntheticScene, Triangle

_LIGHT_DIR = np.array([0.4, -0.3, 0.85])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
_AMBIENT = 0.35


@dataclasses.dataclass
class RenderResult:
    rgb: np.ndarray            # (H,W,3) uint8
    depth: np.ndarray          # (H,W) float32, 0 = no geometry hit
    semantic: np.ndarray       # (H,W) int32, -1 = background
    triangle_id: np.ndarray    # (H,W) int32, -1 = background, else index into the combined triangle list
    dynamic_mask: np.ndarray   # (H,W) bool, True where a dynamic (moving) object is the visible surface
    visible_object_ids: np.ndarray   # sorted unique object ids that produced >=1 visible pixel
    per_triangle_view_angle: dict    # {tri_idx: angle_rad} for triangles with >=1 visible pixel
    per_triangle_visible_frac: dict  # {tri_idx: fraction of its projected area actually shown}


def render_frame(scene: SyntheticScene, K_obj: Intrinsics, pose: CameraPose,
                  near: float = 0.5, far: float = 400.0, extra_triangles=None) -> RenderResult:
    """
    extra_triangles: optional list[Triangle] rendered together with the
    static scene and z-tested against it (used to composite per-frame
    dynamic objects -- vehicles/pedestrians -- without mutating the scene).
    Indices >= len(scene.triangles) in the returned triangle_id map refer
    to this list, offset by len(scene.triangles).
    """
    H, W = K_obj.height, K_obj.width
    K = K_obj.K()

    depth_buf = np.full((H, W), np.inf, dtype=np.float32)
    rgb_buf = np.zeros((H, W, 3), dtype=np.uint8)
    sem_buf = np.full((H, W), -1, dtype=np.int32)
    tri_buf = np.full((H, W), -1, dtype=np.int32)
    dyn_buf = np.zeros((H, W), dtype=bool)

    all_triangles = scene.triangles if not extra_triangles else (scene.triangles + list(extra_triangles))
    n_static = len(scene.triangles)
    V = np.stack([t.verts for t in all_triangles], axis=0)
    C = np.stack([t.color for t in all_triangles], axis=0)
    Nrm = np.stack([t.normal for t in all_triangles], axis=0)
    Sem = np.array([t.semantic for t in all_triangles], dtype=np.int32)
    Obj = np.array([t.object_id for t in all_triangles], dtype=np.int32)
    Dyn = np.array([t.dynamic for t in all_triangles], dtype=bool)
    view_angle = {}
    visible_frac = {}

    # Camera-frame vertices for all triangles at once.
    Vc = pose.world_to_cam(V.reshape(-1, 3)).reshape(V.shape)   # (N,3,3)
    z = Vc[..., 2]
    in_front = np.all(z > near, axis=1) & np.all(z < far, axis=1)
    cand_idx = np.nonzero(in_front)[0]

    # Backface cull using camera-space normal (avoid drawing the inside of buildings).
    cam_dir_to_tri = Vc[cand_idx].mean(axis=1)
    Nc = Nrm[cand_idx] @ pose.R_wc  # normal into camera frame
    facing = np.einsum("ij,ij->i", Nc, -cam_dir_to_tri) > 0
    cand_idx = cand_idx[facing]

    uv, zc = project_points(K, Vc[cand_idx])   # uv:(M,3,2) zc:(M,3)

    for k, ti in enumerate(cand_idx):
        pts2d = uv[k]
        tri_z = zc[k]
        x0, y0 = pts2d.min(axis=0)
        x1, y1 = pts2d.max(axis=0)
        if x1 < 0 or y1 < 0 or x0 > W - 1 or y0 > H - 1:
            continue
        xi0, xi1 = int(max(0, np.floor(x0))), int(min(W - 1, np.ceil(x1)))
        yi0, yi1 = int(max(0, np.floor(y0))), int(min(H - 1, np.ceil(y1)))
        bw, bh = xi1 - xi0 + 1, yi1 - yi0 + 1
        if bw <= 0 or bh <= 0 or bw * bh > (W * H):
            continue

        mask = np.zeros((bh, bw), dtype=np.uint8)
        poly = (pts2d - np.array([xi0, yi0])).astype(np.int32)
        cv2.fillConvexPoly(mask, poly, 1)
        ys, xs = np.nonzero(mask)
        n_px = ys.shape[0]
        if n_px == 0:
            continue

        # Barycentric weights for interpolated depth + flat shading.
        p = np.stack([xs + xi0, ys + yi0], axis=1).astype(np.float64)
        a, b, c = pts2d[0], pts2d[1], pts2d[2]
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        denom = denom if abs(denom) > 1e-9 else 1e-9
        w0 = ((b[1] - c[1]) * (p[:, 0] - c[0]) + (c[0] - b[0]) * (p[:, 1] - c[1])) / denom
        w1 = ((c[1] - a[1]) * (p[:, 0] - c[0]) + (a[0] - c[0]) * (p[:, 1] - c[1])) / denom
        w2 = 1 - w0 - w1
        interp_z = (w0 * tri_z[0] + w1 * tri_z[1] + w2 * tri_z[2]).astype(np.float32)

        gy = ys + yi0
        gx = xs + xi0
        cur_depth = depth_buf[gy, gx]
        closer = interp_z < cur_depth
        visible_frac[int(ti)] = float(closer.sum()) / float(n_px)
        if not np.any(closer):
            continue

        sel_y, sel_x = gy[closer], gx[closer]
        depth_buf[sel_y, sel_x] = interp_z[closer]

        shade = _AMBIENT + (1 - _AMBIENT) * max(0.0, float(np.dot(Nrm[ti], _LIGHT_DIR)))
        color = np.clip(C[ti] * shade, 0, 1)
        rgb_buf[sel_y, sel_x] = (color * 255).astype(np.uint8)
        sem_buf[sel_y, sel_x] = Sem[ti]
        tri_buf[sel_y, sel_x] = ti
        if Dyn[ti]:
            dyn_buf[sel_y, sel_x] = True
        view_angle[int(ti)] = pose.viewing_ray_angle_to(all_triangles[ti].centroid)

    depth_out = np.where(np.isfinite(depth_buf), depth_buf, 0.0).astype(np.float32)
    static_keys = [k for k in view_angle.keys() if k < n_static]
    visible_ids = np.unique(Obj[np.array(static_keys, dtype=np.int64)]) if static_keys else np.array([], dtype=np.int32)

    return RenderResult(rgb=rgb_buf, depth=depth_out, semantic=sem_buf, triangle_id=tri_buf,
                         dynamic_mask=dyn_buf, visible_object_ids=visible_ids,
                         per_triangle_view_angle=view_angle, per_triangle_visible_frac=visible_frac)


if __name__ == "__main__":
    import time
    from .scene_generator import generate_scene
    from .trajectory_generator import generate_single_pass_trajectory

    scene = generate_scene(seed=0, extent=40.0)
    traj = generate_single_pass_trajectory(scene.scene_bounds, seed=0)
    K = Intrinsics.from_fov(160, 120, hfov_deg=70)

    t0 = time.perf_counter()
    res = render_frame(scene, K, traj.camera_pose(len(traj) // 2))
    t1 = time.perf_counter()
    print(f"Rendered 1 frame ({K.width}x{K.height}, {len(scene.triangles)} tris) in {t1 - t0:.3f}s")
    print("nonzero depth px:", int((res.depth > 0).sum()), "/", K.width * K.height)
    print("visible objects:", res.visible_object_ids)
    cv2.imwrite("/tmp/test_render.png", cv2.cvtColor(res.rgb, cv2.COLOR_RGB2BGR))
    print("saved /tmp/test_render.png")
