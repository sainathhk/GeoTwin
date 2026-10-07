"""
scene_generator.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Generates a fully KNOWN synthetic 3D scene: buildings, terrain, roads and
vegetation, as an explicit triangle soup with per-triangle ground-truth
semantic label, color and (for planar city elements) a checkerboard
sub-tessellation so that classical stereo/MVS has texture to match against.

This is the ground-truth generator required by the evaluation framework:
every number the pipeline later reports is checked against the geometry
built here, so nothing about the "true" scene is ever guessed.

Coordinate convention: world frame is ENU-like (X=east, Y=north, Z=up),
units are meters. This is converted to a local drone-metric frame; the
georeferencing module is responsible for mapping this to lat/lon/alt.
"""
from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np

SemanticClass = {
    "terrain": 0,
    "building_facade": 1,
    "building_roof": 2,
    "road": 3,
    "vegetation": 4,
    "dynamic_object": 5,   # vehicles/people injected by degradation.py, never part of the static scene
}


def make_box_triangles(center_xy, cz0, w, d, h, color, object_id, dynamic=False):
    """Axis-aligned box (6 faces, 12 tris) -- used for vehicles/simple dynamic-object proxies."""
    x, y = center_xy
    x0, x1 = x - w / 2, x + w / 2
    y0, y1 = y - d / 2, y + d / 2
    z0, z1 = cz0, cz0 + h
    color = np.asarray(color, dtype=np.float32)

    corners_bot = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], dtype=np.float32)
    corners_top = corners_bot.copy()
    corners_top[:, 2] = z1

    faces = [
        (corners_bot[0], corners_bot[1], corners_bot[2], corners_bot[3]),  # bottom
        (corners_top[0], corners_top[3], corners_top[2], corners_top[1]),  # top
        (corners_bot[0], corners_top[0], corners_top[1], corners_bot[1]),  # y0 side
        (corners_bot[2], corners_top[2], corners_top[3], corners_bot[3]),  # y1 side
        (corners_bot[1], corners_top[1], corners_top[2], corners_bot[2]),  # x1 side
        (corners_bot[3], corners_top[3], corners_top[0], corners_bot[0]),  # x0 side
    ]
    tris = []
    for (a, b, c, d4) in faces:
        n1 = _face_normal(a, b, c)
        tris.append(Triangle(np.stack([a, b, c]), color, n1, SemanticClass["dynamic_object"],
                              object_id, dynamic=dynamic))
        n2 = _face_normal(a, c, d4)
        tris.append(Triangle(np.stack([a, c, d4]), color, n2, SemanticClass["dynamic_object"],
                              object_id, dynamic=dynamic))
    return tris


@dataclasses.dataclass
class Triangle:
    """A single ground-truth triangle: 3 vertices (3,3), flat color, normal, semantic id."""
    verts: np.ndarray          # (3,3) float32 world-space vertices
    color: np.ndarray          # (3,) float32 RGB in [0,1]
    normal: np.ndarray         # (3,) float32 unit normal
    semantic: int
    object_id: int             # which building/road/veg instance this triangle belongs to
    dynamic: bool = False       # true only for injected moving objects (added later by degradation)

    @property
    def centroid(self) -> np.ndarray:
        return self.verts.mean(axis=0)


@dataclasses.dataclass
class SyntheticScene:
    triangles: List[Triangle]
    buildings: List[dict]       # ground-truth building metadata (footprint, height, id)
    scene_bounds: Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    seed: int

    def triangle_array(self):
        """Stack all triangles into big numpy arrays for vectorised rendering."""
        V = np.stack([t.verts for t in self.triangles], axis=0)          # (N,3,3)
        C = np.stack([t.color for t in self.triangles], axis=0)          # (N,3)
        Nrm = np.stack([t.normal for t in self.triangles], axis=0)       # (N,3)
        Sem = np.array([t.semantic for t in self.triangles], dtype=np.int32)
        Obj = np.array([t.object_id for t in self.triangles], dtype=np.int32)
        return V, C, Nrm, Sem, Obj


def _face_normal(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (n / norm).astype(np.float32)


def _checker_quad(p00, p01, p11, p10, semantic, object_id, n_grid, color_a, color_b, rng):
    """
    Tessellate a planar quad (p00,p01,p11,p10 in order around the boundary)
    into an n_grid x n_grid checkerboard of triangles. Gives classical
    stereo/MVS matchers real texture to lock onto instead of blank facades.
    """
    tris = []
    for i in range(n_grid):
        for j in range(n_grid):
            u0, u1 = i / n_grid, (i + 1) / n_grid
            v0, v1 = j / n_grid, (j + 1) / n_grid

            def bilerp(u, v):
                a = p00 * (1 - u) + p10 * u
                b = p01 * (1 - u) + p11 * u
                return a * (1 - v) + b * v

            a = bilerp(u0, v0)
            b = bilerp(u1, v0)
            c = bilerp(u1, v1)
            d = bilerp(u0, v1)

            checker = (i + j) % 2 == 0
            base = color_a if checker else color_b
            # Strong per-cell jitter breaks the periodicity of a plain checkerboard,
            # which otherwise creates classic repeated-texture aliasing for stereo/MVS
            # matching (many equally-good false depth hypotheses). This keeps the
            # alternating pattern for visual readability while giving each cell a
            # near-unique appearance, closer to how real facades/ground texture behaves.
            jitter = (rng.random(3).astype(np.float32) - 0.5) * 0.5
            col = np.clip(base + jitter, 0, 1).astype(np.float32)

            n1 = _face_normal(a, b, c)
            tris.append(Triangle(np.stack([a, b, c]), col, n1, semantic, object_id))
            n2 = _face_normal(a, c, d)
            tris.append(Triangle(np.stack([a, c, d]), col, n2, semantic, object_id))
    return tris


def _make_building(x, y, w, d, h, object_id, rng, facade_grid=3):
    """Axis-aligned box building: 4 checkerboard facades + a flat roof."""
    tris = []
    corners = {
        "sw": np.array([x, y, 0.0], dtype=np.float32),
        "se": np.array([x + w, y, 0.0], dtype=np.float32),
        "ne": np.array([x + w, y + d, 0.0], dtype=np.float32),
        "nw": np.array([x, y + d, 0.0], dtype=np.float32),
    }
    top = {k: v + np.array([0, 0, h], dtype=np.float32) for k, v in corners.items()}

    facade_color = np.array([0.55, 0.55, 0.6], dtype=np.float32) + (rng.random(3) - 0.5) * 0.2
    facade_color = np.clip(facade_color, 0.2, 0.9).astype(np.float32)
    window_color = np.clip(facade_color * 0.55, 0.05, 0.9).astype(np.float32)

    # South facade (y = y), North facade (y = y+d), West (x=x), East (x=x+w)
    tris += _checker_quad(corners["sw"], top["sw"], top["se"], corners["se"],
                           SemanticClass["building_facade"], object_id, facade_grid,
                           facade_color, window_color, rng)
    tris += _checker_quad(corners["se"], top["se"], top["ne"], corners["ne"],
                           SemanticClass["building_facade"], object_id, facade_grid,
                           facade_color, window_color, rng)
    tris += _checker_quad(corners["ne"], top["ne"], top["nw"], corners["nw"],
                           SemanticClass["building_facade"], object_id, facade_grid,
                           facade_color, window_color, rng)
    tris += _checker_quad(corners["nw"], top["nw"], top["sw"], corners["sw"],
                           SemanticClass["building_facade"], object_id, facade_grid,
                           facade_color, window_color, rng)

    roof_color = np.clip(np.array([0.35, 0.3, 0.3], dtype=np.float32) + (rng.random(3) - 0.5) * 0.1, 0.05, 0.9)
    tris += _checker_quad(top["sw"], top["nw"], top["ne"], top["se"],
                           SemanticClass["building_roof"], object_id, 2,
                           roof_color, roof_color * 0.85, rng)

    meta = {"id": object_id, "x": x, "y": y, "w": w, "d": d, "h": h,
            "center_xy": (x + w / 2.0, y + d / 2.0), "footprint_z": 0.0, "top_z": h}
    return tris, meta


def _make_vegetation(x, y, radius, height, object_id, rng):
    """Simple 4-sided pyramid billboard proxy for a tree/bush, cheap but 3-D."""
    apex = np.array([x, y, height], dtype=np.float32)
    base_pts = []
    for k in range(4):
        ang = np.pi / 2 * k + np.pi / 4
        base_pts.append(np.array([x + radius * np.cos(ang), y + radius * np.sin(ang), 0.0], dtype=np.float32))
    color = np.clip(np.array([0.18, 0.42, 0.16], dtype=np.float32) + (rng.random(3) - 0.5) * 0.08, 0.02, 0.8)
    tris = []
    for k in range(4):
        b0 = base_pts[k]
        b1 = base_pts[(k + 1) % 4]
        n = _face_normal(b0, b1, apex)
        tris.append(Triangle(np.stack([b0, b1, apex]), color.astype(np.float32), n,
                              SemanticClass["vegetation"], object_id))
    return tris


def generate_scene(seed: int = 0, extent: float = 60.0) -> SyntheticScene:
    """
    Build a small mixed urban scene: ground terrain, a road, several
    buildings of varied footprint/height, and scattered vegetation.
    `extent` is the scene half-width in meters (scene spans [-extent, extent]^2).
    """
    rng = np.random.default_rng(seed)
    triangles: List[Triangle] = []
    buildings_meta = []
    obj_id = 0

    # --- Terrain: flat ground plane, mildly checkered for MVS texture ---
    n_grid = 8
    ground_color_a = np.array([0.42, 0.4, 0.32], dtype=np.float32)
    ground_color_b = np.array([0.38, 0.36, 0.29], dtype=np.float32)
    p00 = np.array([-extent, -extent, 0.0], dtype=np.float32)
    p10 = np.array([extent, -extent, 0.0], dtype=np.float32)
    p11 = np.array([extent, extent, 0.0], dtype=np.float32)
    p01 = np.array([-extent, extent, 0.0], dtype=np.float32)
    triangles += _checker_quad(p00, p01, p11, p10, SemanticClass["terrain"], obj_id, n_grid,
                                ground_color_a, ground_color_b, rng)
    obj_id += 1

    # --- Road: a straight strip running through the middle of the scene ---
    road_w = 6.0
    road_color_a = np.array([0.16, 0.16, 0.17], dtype=np.float32)
    road_color_b = np.array([0.2, 0.2, 0.21], dtype=np.float32)
    rp00 = np.array([-extent, -road_w / 2, 0.01], dtype=np.float32)
    rp10 = np.array([extent, -road_w / 2, 0.01], dtype=np.float32)
    rp11 = np.array([extent, road_w / 2, 0.01], dtype=np.float32)
    rp01 = np.array([-extent, road_w / 2, 0.01], dtype=np.float32)
    triangles += _checker_quad(rp00, rp01, rp11, rp10, SemanticClass["road"], obj_id, 16,
                                road_color_a, road_color_b, rng)
    obj_id += 1

    # --- Buildings: grid of plots, avoiding the road corridor ---
    plot_size = 14.0
    n_side = int((2 * extent) // (plot_size * 1.6))
    n_side = max(2, min(n_side, 4))
    for ix in range(n_side):
        for iy in range(n_side):
            px = -extent + 8 + ix * (2 * extent - 16) / max(1, n_side - 1)
            py = -extent + 8 + iy * (2 * extent - 16) / max(1, n_side - 1)
            if abs(py) < road_w * 1.5 and abs(px) < extent * 0.9:
                py += np.sign(py + 1e-6) * road_w * 2.0  # push off the road
            w = rng.uniform(6.0, 11.0)
            d = rng.uniform(6.0, 11.0)
            h = rng.uniform(8.0, 28.0)
            tris, meta = _make_building(px - w / 2, py - d / 2, w, d, h, obj_id, rng)
            triangles += tris
            buildings_meta.append(meta)
            obj_id += 1

    # --- Vegetation: scattered small trees away from buildings/road ---
    n_trees = 10
    for _ in range(n_trees):
        for _try in range(20):
            tx = rng.uniform(-extent + 3, extent - 3)
            ty = rng.uniform(-extent + 3, extent - 3)
            if abs(ty) < road_w:
                continue
            too_close = any(abs(tx - b["center_xy"][0]) < b["w"] and abs(ty - b["center_xy"][1]) < b["d"]
                             for b in buildings_meta)
            if too_close:
                continue
            r = rng.uniform(1.0, 2.2)
            hgt = rng.uniform(2.5, 6.0)
            triangles += _make_vegetation(tx, ty, r, hgt, obj_id, rng)
            obj_id += 1
            break

    return SyntheticScene(triangles=triangles, buildings=buildings_meta,
                           scene_bounds=(-extent, extent, -extent, extent), seed=seed)


if __name__ == "__main__":
    scene = generate_scene(seed=0, extent=40.0)
    print(f"Generated scene: {len(scene.triangles)} triangles, {len(scene.buildings)} buildings")
    for b in scene.buildings:
        print(f"  building {b['id']}: footprint {b['w']:.1f}x{b['d']:.1f} m, height {b['h']:.2f} m, "
              f"center {b['center_xy']}")
