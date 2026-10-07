"""
ground_truth.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Everything the evaluation framework needs as "truth":
  * a dense GT surface point sample (for Chamfer / P2P / P2Plane / completeness)
  * a set of GT "control measurements" -- known building heights, footprint
    diagonals, and inter-building distances -- used to report a genuine
    metric-accuracy number ("a 20 m building reconstructs at 19.7 m") rather
    than only an image-similarity number.

Per-point *visibility* (which frames actually observed a given surface
point, at what angle) is NOT computed here -- it falls out for free from
the renderer's per-triangle visibility stats when the pipeline renders the
trajectory, and is joined back onto these points via `triangle_id`. Keeping
that logic in one place (the renderer) avoids a second, potentially
inconsistent visibility computation.
"""
from __future__ import annotations

import dataclasses
import itertools
import numpy as np

from .scene_generator import SyntheticScene


@dataclasses.dataclass
class GTPointCloud:
    points: np.ndarray        # (M,3)
    colors: np.ndarray        # (M,3) in [0,1]
    normals: np.ndarray       # (M,3)
    semantic: np.ndarray      # (M,)
    object_id: np.ndarray     # (M,)
    triangle_id: np.ndarray   # (M,) index into scene.triangles


def sample_surface_points(scene: SyntheticScene, points_per_sqm: float = 2.0,
                           seed: int = 0, max_points: int = 40000) -> GTPointCloud:
    """Uniformly sample points on every triangle's surface, density proportional to area."""
    rng = np.random.default_rng(seed)
    pts, cols, nrms, sems, objs, tids = [], [], [], [], [], []

    for ti, tri in enumerate(scene.triangles):
        v0, v1, v2 = tri.verts
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        n = max(1, int(round(area * points_per_sqm)))
        r1 = rng.random(n)
        r2 = rng.random(n)
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1 - u - v
        p = u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
        pts.append(p)
        cols.append(np.tile(tri.color, (n, 1)))
        nrms.append(np.tile(tri.normal, (n, 1)))
        sems.append(np.full(n, tri.semantic))
        objs.append(np.full(n, tri.object_id))
        tids.append(np.full(n, ti))

    points = np.concatenate(pts, axis=0)
    colors = np.concatenate(cols, axis=0)
    normals = np.concatenate(nrms, axis=0)
    semantic = np.concatenate(sems, axis=0)
    object_id = np.concatenate(objs, axis=0)
    triangle_id = np.concatenate(tids, axis=0)

    if points.shape[0] > max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points, colors, normals = points[idx], colors[idx], normals[idx]
        semantic, object_id, triangle_id = semantic[idx], object_id[idx], triangle_id[idx]

    return GTPointCloud(points, colors, normals, semantic, object_id, triangle_id)


def compute_control_measurements(scene: SyntheticScene) -> dict:
    """
    Known-truth distances used purely for METRIC accuracy evaluation
    (Section C of the evaluation framework) -- deliberately independent of
    the dense point cloud above, the way real GCP/control-point checks are
    independent of the photogrammetric point cloud they validate.
    """
    heights = {b["id"]: float(b["h"]) for b in scene.buildings}
    footprint_diag = {b["id"]: float(np.hypot(b["w"], b["d"])) for b in scene.buildings}

    inter_building = {}
    for (a, b) in itertools.combinations(scene.buildings, 2):
        d = float(np.hypot(a["center_xy"][0] - b["center_xy"][0],
                            a["center_xy"][1] - b["center_xy"][1]))
        inter_building[(a["id"], b["id"])] = d

    return {
        "building_heights_m": heights,
        "building_footprint_diagonals_m": footprint_diag,
        "inter_building_center_distances_m": inter_building,
    }


if __name__ == "__main__":
    from .scene_generator import generate_scene
    scene = generate_scene(seed=0, extent=40.0)
    gt = sample_surface_points(scene, points_per_sqm=1.0)
    print(f"GT point cloud: {gt.points.shape[0]} points")
    ctrl = compute_control_measurements(scene)
    print("building heights (m):", ctrl["building_heights_m"])
