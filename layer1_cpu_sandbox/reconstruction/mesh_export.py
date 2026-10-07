"""
mesh_export.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested (tests/test_mesh_export.py),
CPU-executed against a real checkpoint (docs/MESH_QUALITY_AUDIT.md, 2026-09).

Turns whatever positions+colors a layer already trusts -- Layer 1's
confidence-fused point cloud (`PrototypePointRepresentation`), or Layer 3's
trained Gaussian centers (`GaussianModel.xyz` + SH-DC color) -- into a
single self-contained `.obj` triangle mesh a person can open in Blender /
MeshLab / CloudCompare / a web viewer, without any project-specific
tooling. This is a DELIVERABLE-FORMAT step, not a change to the core
representation: docs/ARCHITECTURE.md's point-cloud-vs-mesh-vs-NeRF-vs-3DGS
table still holds (3DGS stays what's trained and rendered); this module
only runs once, after the fact, to produce an extra, easier-to-share export
of whatever positions the representation already has.

WHY OPEN3D, NOT A HAND-ROLLED RECONSTRUCTION (compare docs/ARCHITECTURE.md
section 3, "why we do not reimplement the tile rasterizer"): Poisson
surface reconstruction and ball-pivoting are solved, mature, well-optimized
problems with a public, well-tested implementation. Reimplementing either
from scratch would be hundreds of lines of numerical code that contributes
zero novelty to this project -- the actual novelty (observation-aware
confidence) is applied BEFORE reconstruction, as the `keep_mask` filter
below, not inside the meshing algorithm itself. This is the one module in
`layer1_cpu_sandbox/` with a heavy external dependency (everything else
here is numpy/opencv/scipy); that is a deliberate, scoped exception, not a
drift away from the CPU sandbox's usual minimal-dependency discipline.

CONFIDENCE / HONESTY DISCIPLINE (docs/STOP_CONDITIONS.md #1 applies here
too): this module does NOT do generative surface completion. Poisson
reconstruction is a smooth INTERPOLATING fit, and it will still bridge
small gaps between nearby confident points -- that is a known property of
the algorithm, not a claim that the bridge is observed geometry. Three
things keep that honest rather than silently papering over gaps: (1) the
caller is expected to pass `keep_mask` computed from the SAME
observation-confidence signal this project's core novelty is built on
(e.g. Gaussian opacity/confidence, or a fused point's confidence score) so
low-evidence primitives never reach the reconstructor at all; (2)
`density_trim_quantile` removes the lowest-support fraction of the
reconstructed surface using Poisson's OWN density output, so the parts of
the mesh it was least evidenced for get cut away rather than shipped as if
they were as reliable as the rest; (3) (2026-09 addition)
`min_component_vertices` / `keep_largest_n_components` drop small
disconnected debris -- islands Poisson invents around a handful of
outlier points -- which are legitimately part of "how good is this mesh"
but are NOT part of "does the reconstruction bridge unobserved gaps", so
they're a separate, additional cleanup step, not a replacement for (1)/(2).

2026-09 MESH QUALITY AUDIT (see docs/MESH_QUALITY_AUDIT.md for the full
investigation against a real full_state_iter8000.pt): this module's
original normal estimation (`pcd.estimate_normals()`, generic local-PCA
over bare point positions) ignored Gaussian scale/rotation entirely. This
version accepts a `normals` array so a caller can supply
`gaussian_geometry.compute_gaussian_geometry(...).normals` instead --
physically-grounded, free, and no worse than the old behavior. IMPORTANT,
because the audit's own ablation showed it clearly: supplying better
normals did NOT, by itself, fix a badly-seeded point cloud on that
checkpoint (the dominant fault was upstream, in depth-estimation
quantization -- see that module's docstring). Use the improvements below;
don't expect them alone to fix a checkpoint with the same upstream problem.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


class MeshExportError(ValueError):
    """Raised when there isn't enough surviving evidence to reconstruct a mesh --
    distinguished from a plain ValueError so callers (train_gpu.py in particular,
    which must NOT let a meshing problem discard hours of GPU training) can catch
    this specifically and degrade gracefully instead of crashing."""


@dataclasses.dataclass
class MeshExportStats:
    n_input_points: int
    n_points_after_filter: int
    n_points_after_outlier_removal: int
    n_vertices: int
    n_triangles: int
    method: str
    n_connected_components: int
    largest_component_fraction: float
    n_components_removed_as_debris: int
    n_degenerate_triangles: int
    n_nonmanifold_edges: int
    n_boundary_edges: int
    surface_area: float
    bbox_min: tuple
    bbox_max: tuple
    triangle_area_p50: float
    triangle_area_p99: float
    path: Optional[str] = None

    def format_report(self) -> str:
        bx = tuple(round(v, 3) for v in self.bbox_min)
        by = tuple(round(v, 3) for v in self.bbox_max)
        return (
            f"  input points:              {self.n_input_points}\n"
            f"  after opacity/confidence:  {self.n_points_after_filter} "
            f"({100*self.n_points_after_filter/max(self.n_input_points,1):.1f}%)\n"
            f"  after outlier removal:     {self.n_points_after_outlier_removal}\n"
            f"  method:                    {self.method}\n"
            f"  vertices / triangles:      {self.n_vertices} / {self.n_triangles}\n"
            f"  connected components:      {self.n_connected_components} "
            f"(largest holds {100*self.largest_component_fraction:.1f}% of vertices; "
            f"removed {self.n_components_removed_as_debris} small ones as debris)\n"
            f"  degenerate triangles:      {self.n_degenerate_triangles}\n"
            f"  non-manifold edges:        {self.n_nonmanifold_edges}\n"
            f"  boundary edges:            {self.n_boundary_edges}\n"
            f"  surface area:              {self.surface_area:.1f}\n"
            f"  bounding box:              {bx} .. {by}\n"
            f"  triangle area p50 / p99:   {self.triangle_area_p50:.5g} / {self.triangle_area_p99:.5g}\n"
        )


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "mesh_export needs the 'open3d' package (pip install open3d, or "
            "pip install -r requirements.txt -- it's listed there). It is a "
            "pure-CPU wheel with no CUDA dependency, safe in Environment A."
        ) from e
    return o3d


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    """Same [0,1]-vs-[0,255] auto-detection convention as utils.save_ply, so a
    caller never has to think about which one it's holding."""
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size and colors.max() > 1.0001:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _topology_stats(mesh) -> dict:
    """Every numerical mesh-quality signal this project's own audit process asked
    for: connected components + sizes, degenerate triangles, non-manifold /
    boundary edge counts, surface area, triangle-size distribution. Pure numpy
    + scipy over the mesh's own vertex/triangle arrays -- doesn't touch Open3D's
    (much more limited) built-in quality-check surface."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    n = len(V)
    if len(F) == 0 or n == 0:
        return dict(n_components=0, largest_frac=0.0, degenerate=0, nonmanifold_edges=0,
                     boundary_edges=0, area=0.0, bbox_min=(0., 0., 0.), bbox_max=(0., 0., 0.),
                     area_p50=0.0, area_p99=0.0, component_labels=np.zeros(0, dtype=int), component_sizes=[])

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    repeated = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    degenerate = int(((area < 1e-10) | repeated).sum())

    e0 = np.sort(F[:, [0, 1]], axis=1); e1 = np.sort(F[:, [1, 2]], axis=1); e2 = np.sort(F[:, [2, 0]], axis=1)
    edges = np.concatenate([e0, e1, e2], axis=0)
    key = edges[:, 0].astype(np.int64) * (n + 1) + edges[:, 1]
    _, counts = np.unique(key, return_counts=True)
    boundary_edges = int((counts == 1).sum())
    nonmanifold_edges = int((counts > 2).sum())

    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    adj = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    largest_frac = float(sizes.max() / n) if n_comp > 0 else 0.0

    return dict(n_components=int(n_comp), largest_frac=largest_frac, degenerate=degenerate,
                nonmanifold_edges=nonmanifold_edges, boundary_edges=boundary_edges,
                area=float(area.sum()), bbox_min=tuple(V.min(0).tolist()), bbox_max=tuple(V.max(0).tolist()),
                area_p50=float(np.median(area)), area_p99=float(np.percentile(area, 99)),
                component_labels=labels, component_sizes=sizes.tolist())


def remove_small_components(mesh, min_vertices: int = 0, keep_largest_n: Optional[int] = None):
    """Drops connected components below `min_vertices`, and/or keeps only the
    `keep_largest_n` biggest components. Either/both may be given; `None`/`0`
    disables that criterion. This is debris cleanup (see module docstring point
    3) -- islands Poisson invents around a handful of outlier points -- not a
    substitute for filtering the INPUT with keep_mask/outlier removal, which
    is what actually controls whether the surviving surface is trustworthy.
    Returns (mesh, n_components_removed)."""
    stats = _topology_stats(mesh)
    labels = stats["component_labels"]
    sizes = np.array(stats["component_sizes"])
    if len(sizes) == 0:
        return mesh, 0

    keep_component = np.ones(len(sizes), dtype=bool)
    if min_vertices and min_vertices > 0:
        keep_component &= sizes >= min_vertices
    if keep_largest_n:
        order = np.argsort(sizes)[::-1]
        drop = order[keep_largest_n:]
        keep_component[drop] = False

    n_removed = int((~keep_component).sum())
    if n_removed == 0:
        return mesh, 0

    vertex_keep_mask = keep_component[labels]
    mesh.remove_vertices_by_mask(~vertex_keep_mask)
    return mesh, n_removed


def points_to_mesh(positions: np.ndarray, colors: np.ndarray = None, keep_mask: np.ndarray = None,
                    method: str = "poisson", poisson_depth: int = 9, density_trim_quantile: float = 0.03,
                    normal_knn: int = 30, voxel_size: float = None, min_points: int = 50,
                    normals: np.ndarray = None, statistical_outlier_neighbors: int = 20,
                    statistical_outlier_std_ratio: float = 2.0, min_component_vertices: int = 0,
                    keep_largest_n_components: Optional[int] = None):
    """
    Reconstructs a triangle mesh from a point cloud. Returns (mesh, stats) where
    `mesh` is an open3d.geometry.TriangleMesh and `stats` is a MeshExportStats.

    positions: (N,3) float array.
    colors: optional (N,3) array, either [0,1] or [0,255] (auto-detected).
    keep_mask: optional (N,) bool array applied BEFORE reconstruction -- this is
        where a caller's own confidence/opacity gating belongs (see module
        docstring); this function itself has no opinion on what "confidence"
        means for a given representation.
    normals: optional (N,3) array of PRE-COMPUTED normal directions (e.g. from
        gaussian_geometry.compute_gaussian_geometry(...).normals), applied
        after `keep_mask` in the SAME order/indexing as `positions`. When given,
        `estimate_normals()` is skipped entirely -- the supplied direction is
        used as-is, then sign-corrected the same way either path is (see
        `orient_normals_consistent_tangent_plane` below; a covariance normal's
        SIGN is exactly as arbitrary as a PCA normal's, only its DIRECTION is
        better-grounded). None (default) = old behavior, estimate via local PCA.
    method: "poisson" (default -- smooth, watertight-seeking, needs oriented
        normals) or "ball_pivoting" (more literal to the input points, better
        at leaving genuine holes as holes, more sensitive to uneven density --
        see docs/MESH_QUALITY_AUDIT.md's ablation: this failed badly, 2.9% of
        vertices in its largest component, on a real non-uniform-density
        aerial checkpoint; keep as an option, don't default to it for this
        kind of data).
    poisson_depth: octree depth for Poisson reconstruction; higher = more
        detail and more compute/memory. 9 is a reasonable prototype default
        (Open3D's own examples typically use 8-10 for scene-scale data);
        raise this during the "quality" pass, not now.
    density_trim_quantile: fraction of reconstructed vertices with the LOWEST
        Poisson density estimate to discard (0 disables). Mild by default --
        see module docstring for why this exists.
    voxel_size: optional pre-reconstruction downsample; None = no downsampling.
        Homogenizing point density before Poisson reduces spurious high-
        frequency surface noise on very non-uniform-density inputs (Gaussian
        centers are NOT uniformly spaced -- densification concentrates them
        wherever training loss was highest, not wherever the true surface is
        simplest) -- see docs/MESH_QUALITY_AUDIT.md variant E.
    min_points: raise MeshExportError if fewer than this many points survive
        `keep_mask`, rather than silently attempting a degenerate reconstruction.
    statistical_outlier_neighbors / statistical_outlier_std_ratio: Open3D
        `remove_statistical_outlier` parameters, applied after `keep_mask` and
        before reconstruction. Set `statistical_outlier_neighbors=0` to disable.
    min_component_vertices / keep_largest_n_components: post-reconstruction
        debris cleanup -- see `remove_small_components`. Both default to "off"
        (0 / None) so existing callers get IDENTICAL behavior to before unless
        they opt in.
    """
    o3d = _require_open3d()

    positions = np.asarray(positions, dtype=np.float64)
    n_input = int(positions.shape[0])
    if colors is not None:
        colors = _normalize_colors(colors)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != positions.shape:
            raise ValueError(f"normals shape {normals.shape} must match positions shape {positions.shape}")

    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        positions = positions[keep_mask]
        colors = colors[keep_mask] if colors is not None else None
        normals = normals[keep_mask] if normals is not None else None
    n_after_filter = int(positions.shape[0])

    if n_after_filter < min_points:
        raise MeshExportError(
            f"Only {n_after_filter}/{n_input} points survive `keep_mask` filtering "
            f"(min_points={min_points}). Lower the confidence/opacity threshold that "
            f"produced keep_mask, collect more views, or lower min_points if this "
            f"scene is genuinely just sparse.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    if voxel_size:
        # NOTE: voxel_down_sample does not carry pre-set normals/colors through
        # index-for-index -- it re-averages per-voxel, so when both a voxel_size
        # AND supplied `normals` are given, direction is re-estimated after
        # downsampling rather than trusting the pre-downsample normal average
        # (which can cancel to near-zero for two oppositely-signed inputs in the
        # same voxel). Simpler and more robust than tracking which voxel each
        # supplied normal landed in.
        pcd = pcd.voxel_down_sample(voxel_size)
        normals = None

    n_before_outlier = len(pcd.points)
    if statistical_outlier_neighbors and statistical_outlier_neighbors > 0 and len(pcd.points) > statistical_outlier_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=statistical_outlier_neighbors,
                                                 std_ratio=statistical_outlier_std_ratio)
    n_after_outlier = len(pcd.points)
    if n_after_outlier < min_points:
        raise MeshExportError(
            f"Only {n_after_outlier}/{n_before_outlier} points survive statistical outlier "
            f"removal (nb_neighbors={statistical_outlier_neighbors}, "
            f"std_ratio={statistical_outlier_std_ratio}, min_points={min_points}). Raise "
            f"std_ratio (more permissive) or disable via statistical_outlier_neighbors=0.")

    if normals is None or len(pcd.normals) == 0:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))

    if method == "poisson":
        pcd.orient_normals_consistent_tangent_plane(k=normal_knn)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
        if density_trim_quantile and density_trim_quantile > 0:
            densities = np.asarray(densities)
            thresh = np.quantile(densities, density_trim_quantile)
            mesh.remove_vertices_by_mask(densities < thresh)
        if colors is None:
            # Open3D's Poisson reconstructor always allocates a vertex_colors buffer,
            # defaulting every entry to black (0,0,0) when the source point cloud had
            # none -- has_vertex_colors() then reports True and save_obj would write a
            # mesh that renders solid black instead of using the viewer's default
            # shading. Clear it back to genuinely absent. (ball_pivoting does not have
            # this quirk -- it leaves colors unset when the input had none -- but the
            # check is unconditional here so behavior can't silently depend on that.)
            mesh.vertex_colors = o3d.utility.Vector3dVector()
    elif method == "ball_pivoting":
        pcd.orient_normals_consistent_tangent_plane(k=normal_knn)
        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        avg_dist = float(dists.mean()) if dists.size else 0.0
        if avg_dist <= 0:
            raise MeshExportError("ball_pivoting needs a non-degenerate point cloud "
                                   "(all points coincide, or too few points to estimate spacing).")
        radii = o3d.utility.DoubleVector([avg_dist * m for m in (1.0, 2.0, 4.0)])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'poisson' or 'ball_pivoting'.")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) == 0:
        raise MeshExportError(
            f"Reconstruction ({method}) produced zero triangles from {n_after_outlier} points. "
            f"Try the other `method`, a lower `density_trim_quantile`, or check the points "
            f"aren't degenerate (coplanar/coincident).")

    n_components_removed = 0
    if min_component_vertices or keep_largest_n_components:
        mesh, n_components_removed = remove_small_components(
            mesh, min_vertices=min_component_vertices, keep_largest_n=keep_largest_n_components)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    topo = _topology_stats(mesh)

    stats = MeshExportStats(
        n_input_points=n_input, n_points_after_filter=n_after_filter,
        n_points_after_outlier_removal=n_after_outlier, n_vertices=len(mesh.vertices),
        n_triangles=len(mesh.triangles), method=method,
        n_connected_components=topo["n_components"], largest_component_fraction=topo["largest_frac"],
        n_components_removed_as_debris=n_components_removed, n_degenerate_triangles=topo["degenerate"],
        n_nonmanifold_edges=topo["nonmanifold_edges"], n_boundary_edges=topo["boundary_edges"],
        surface_area=topo["area"], bbox_min=topo["bbox_min"], bbox_max=topo["bbox_max"],
        triangle_area_p50=topo["area_p50"], triangle_area_p99=topo["area_p99"])
    return mesh, stats


def save_obj(mesh, path: str) -> str:
    """Writes `mesh` (an open3d.geometry.TriangleMesh) to `path` as Wavefront .obj.
    Vertex colors, if present, are written as the common (non-standard but
    widely-supported -- MeshLab/CloudCompare/three.js/Blender-with-import-options)
    `v x y z r g b` extension, so a single .obj is self-contained with no
    companion .mtl/texture needed. Baking colors into a proper UV texture is a
    reasonable "quality" pass follow-up, not needed for a first working export."""
    o3d = _require_open3d()
    ok = o3d.io.write_triangle_mesh(path, mesh, write_vertex_colors=True)
    if not ok:
        raise IOError(f"open3d failed to write mesh to {path}")
    return path


def save_glb(mesh, path: str) -> str:
    """Writes `mesh` to `path` as binary glTF (.glb) -- vertex colors carry over
    natively (glTF COLOR_0 attribute), no companion files, opens directly in most
    3D viewers/engines (three.js, Blender, web <model-viewer>). Requested as an
    optional output alongside .obj; same underlying mesh, different container."""
    o3d = _require_open3d()
    ok = o3d.io.write_triangle_mesh(path, mesh, write_vertex_colors=True)
    if not ok:
        raise IOError(f"open3d failed to write mesh to {path}")
    return path


def export_point_cloud_to_obj(path: str, positions: np.ndarray, colors: np.ndarray = None,
                               keep_mask: np.ndarray = None, method: str = "poisson",
                               poisson_depth: int = 9, density_trim_quantile: float = 0.03,
                               normal_knn: int = 30, voxel_size: float = None,
                               min_points: int = 50, normals: np.ndarray = None,
                               statistical_outlier_neighbors: int = 20,
                               statistical_outlier_std_ratio: float = 2.0,
                               min_component_vertices: int = 0,
                               keep_largest_n_components: Optional[int] = None,
                               also_save_glb: bool = False) -> MeshExportStats:
    """One call: reconstruct (see points_to_mesh) then write `path`. Returns the
    MeshExportStats with `.path` filled in. This is the function every entry
    point (run_experiment.py, run_real_experiment.py, train_gpu.py, export_obj.py)
    actually calls, so there is exactly one place that defines "point cloud -> .obj".
    New (2026-09) parameters all default to the pre-audit behavior unless passed
    explicitly -- see points_to_mesh for what each does."""
    mesh, stats = points_to_mesh(
        positions, colors=colors, keep_mask=keep_mask, method=method, poisson_depth=poisson_depth,
        density_trim_quantile=density_trim_quantile, normal_knn=normal_knn, voxel_size=voxel_size,
        min_points=min_points, normals=normals, statistical_outlier_neighbors=statistical_outlier_neighbors,
        statistical_outlier_std_ratio=statistical_outlier_std_ratio,
        min_component_vertices=min_component_vertices, keep_largest_n_components=keep_largest_n_components)
    save_obj(mesh, path)
    stats.path = path
    if also_save_glb:
        save_glb(mesh, path.rsplit(".", 1)[0] + ".glb")
    return stats
