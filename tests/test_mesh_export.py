"""
Tests for layer1_cpu_sandbox/reconstruction/mesh_export.py. Everything here runs
on plain numpy arrays -- no torch, no CUDA, no trained model needed -- since the
module's whole point is that "positions/colors/keep_mask -> mesh" is
representation-independent (see the module docstring). `open3d` is a real,
listed dependency (requirements.txt), not an optional extra, but we still skip
cleanly via importorskip so this file doesn't hard-crash pytest collection in an
environment that hasn't run `pip install -r requirements.txt` yet.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
o3d = pytest.importorskip("open3d")

from layer1_cpu_sandbox.reconstruction.mesh_export import (
    points_to_mesh, save_obj, export_point_cloud_to_obj, MeshExportError, MeshExportStats,
)


def _noisy_sphere(n=3000, radius=2.0, seed=0):
    """Fibonacci-sphere point sample with light noise -- cheap, deterministic,
    and (unlike a random blob) has a well-defined surface so we can sanity-check
    that the reconstructed mesh's extent actually tracks the input's extent."""
    rng = np.random.default_rng(seed)
    i = np.arange(0, n, dtype=np.float64)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden = np.pi * (1 + 5 ** 0.5)
    theta = golden * i
    x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    positions = np.stack([x, y, z], axis=1) * radius
    positions += rng.normal(scale=0.01, size=positions.shape)
    colors = np.clip((positions / radius + 1.0) / 2.0, 0.0, 1.0)
    return positions, colors


def test_poisson_reconstructs_nonempty_mesh_matching_input_extent():
    positions, colors = _noisy_sphere()
    mesh, stats = points_to_mesh(positions, colors, method="poisson", poisson_depth=8)

    assert isinstance(stats, MeshExportStats)
    assert stats.n_input_points == positions.shape[0]
    assert stats.n_points_after_filter == positions.shape[0]
    assert stats.n_vertices > 0 and stats.n_triangles > 0
    assert stats.method == "poisson"

    # reconstructed surface should sit close to the sphere it was built from,
    # not balloon out or collapse in -- loose bounds since Poisson smooths a bit.
    verts = np.asarray(mesh.vertices)
    radii = np.linalg.norm(verts, axis=1)
    assert 1.7 < radii.mean() < 2.3


def test_ball_pivoting_reconstructs_nonempty_mesh():
    positions, colors = _noisy_sphere(n=2000, seed=1)
    mesh, stats = points_to_mesh(positions, colors, method="ball_pivoting")
    assert stats.n_vertices > 0 and stats.n_triangles > 0
    assert stats.method == "ball_pivoting"


def test_unknown_method_raises_value_error():
    positions, colors = _noisy_sphere(n=200, seed=2)
    with pytest.raises(ValueError):
        points_to_mesh(positions, colors, method="not_a_real_method")


def test_keep_mask_filters_before_reconstruction():
    positions, colors = _noisy_sphere(n=3000, seed=3)
    rng = np.random.default_rng(3)
    confidence = rng.uniform(0, 1, size=positions.shape[0])
    keep_mask = confidence >= 0.5

    _, stats = points_to_mesh(positions, colors, keep_mask=keep_mask, method="poisson", poisson_depth=8)

    assert stats.n_input_points == positions.shape[0]
    assert stats.n_points_after_filter == int(keep_mask.sum())
    assert stats.n_points_after_filter < stats.n_input_points


def test_too_few_points_after_filtering_raises_mesh_export_error():
    positions, colors = _noisy_sphere(n=500, seed=4)
    keep_mask = np.zeros(positions.shape[0], dtype=bool)
    keep_mask[:10] = True  # far below the default min_points=50

    with pytest.raises(MeshExportError):
        points_to_mesh(positions, colors, keep_mask=keep_mask)


def test_min_points_is_configurable():
    # 10 points is comfortably enough for ball_pivoting to still find *some*
    # triangles once min_points itself stops blocking it -- the point of this
    # test is only that lowering min_points changes the outcome from "rejected
    # before reconstruction" to "actually attempted".
    positions, colors = _noisy_sphere(n=500, seed=5)
    keep_mask = np.zeros(positions.shape[0], dtype=bool)
    keep_mask[:10] = True

    with pytest.raises(MeshExportError):
        points_to_mesh(positions, colors, keep_mask=keep_mask, min_points=50)

    # same filter, min_points lowered below the surviving count -> no longer
    # rejected for point-count reasons (it may still legitimately fail to
    # produce triangles from 10 near-coplanar-looking points; we only assert
    # it gets *past* the min_points gate, via method="ball_pivoting" which
    # tolerates small clouds better than Poisson for this check).
    _, stats = points_to_mesh(positions, colors, keep_mask=keep_mask, min_points=5, method="ball_pivoting")
    assert stats.n_points_after_filter == 10


def test_colors_in_0_255_range_are_auto_normalized():
    positions, colors_unit = _noisy_sphere(n=1500, seed=6)
    colors_255 = colors_unit * 255.0

    mesh_unit, stats_unit = points_to_mesh(positions, colors_unit, method="poisson", poisson_depth=7)
    mesh_255, stats_255 = points_to_mesh(positions, colors_255, method="poisson", poisson_depth=7)

    # Same input positions + equivalent colors -> the SAME reconstructed geometry, to within
    # Open3D's own run-to-run nondeterminism. Open3D's Poisson solver is multithreaded and its
    # float accumulation order varies with thread scheduling, so vertex/triangle counts can
    # differ by a handful between two runs with identical input -- this was observed in the
    # field (7589 vs 7588 triangles on Colab) after `colors_unit * 255.0 / 255.0` round-tripped
    # to bit-patterns a few ULPs off the originals. Asserting exact equality here was testing
    # the solver's determinism, not this module's color handling, which is what the test is
    # actually for. A 0.5% tolerance still catches any real geometry change (a wrong
    # normalization would move colors by 255x and change the mesh far more than that).
    assert stats_unit.n_vertices == pytest.approx(stats_255.n_vertices, rel=0.005)
    assert stats_unit.n_triangles == pytest.approx(stats_255.n_triangles, rel=0.005)
    assert mesh_unit.has_vertex_colors() and mesh_255.has_vertex_colors()

    # the real point of this test: the [0,255] input must end up in [0,1], not 255x too bright
    colors_out = np.asarray(mesh_255.vertex_colors)
    assert colors_out.max() <= 1.0 + 1e-6, "0-255 colors were not normalized into [0,1]"


def test_mesh_without_colors_has_no_vertex_colors():
    positions, _ = _noisy_sphere(n=1000, seed=7)
    mesh, _ = points_to_mesh(positions, colors=None, method="poisson", poisson_depth=7)
    assert not mesh.has_vertex_colors()


def test_save_obj_writes_readable_file_with_vertex_colors(tmp_path):
    positions, colors = _noisy_sphere(n=2000, seed=8)
    mesh, stats = points_to_mesh(positions, colors, method="poisson", poisson_depth=8)

    out_path = str(tmp_path / "mesh.obj")
    save_obj(mesh, out_path)

    assert os.path.exists(out_path)
    reloaded = o3d.io.read_triangle_mesh(out_path)
    assert len(reloaded.vertices) == stats.n_vertices
    assert len(reloaded.triangles) == stats.n_triangles
    assert reloaded.has_vertex_colors()

    # the extended "v x y z r g b" line is what actually carries the color --
    # confirm it's really on disk, not just present in the in-memory object.
    with open(out_path) as f:
        vertex_lines = [line for line in f if line.startswith("v ")]
    assert len(vertex_lines) == stats.n_vertices
    assert all(len(line.split()) == 7 for line in vertex_lines[:5])  # "v x y z r g b"


def test_export_point_cloud_to_obj_end_to_end(tmp_path):
    positions, colors = _noisy_sphere(n=3000, seed=9)
    rng = np.random.default_rng(9)
    confidence = rng.uniform(0, 1, size=positions.shape[0])
    keep_mask = confidence >= 0.3

    out_path = str(tmp_path / "reconstruction.obj")
    stats = export_point_cloud_to_obj(out_path, positions, colors, keep_mask=keep_mask,
                                       method="poisson", poisson_depth=8)

    assert stats.path == out_path
    assert os.path.exists(out_path)
    assert stats.n_points_after_filter == int(keep_mask.sum())
    assert stats.n_vertices > 0 and stats.n_triangles > 0


# ---------------------------------------------------------------------------
# 2026-09 mesh quality audit additions (docs/MESH_QUALITY_AUDIT.md) -- covariance
# normals, outlier/debris cleanup, and the richer MeshExportStats fields.
# ---------------------------------------------------------------------------

def _sphere_gaussian_normals(positions, radius=2.0):
    """For a point ON a sphere of the given radius centered at the origin, the true
    outward surface normal is just the (unit) position vector -- lets us test the
    `normals=` plumbing without needing a real GaussianModel checkpoint."""
    return positions / np.clip(np.linalg.norm(positions, axis=1, keepdims=True), 1e-9, None)


def test_supplied_normals_are_used_instead_of_estimate_normals():
    positions, colors = _noisy_sphere(n=3000, seed=10)
    normals = _sphere_gaussian_normals(positions)
    mesh, stats = points_to_mesh(positions, colors, normals=normals, method="poisson", poisson_depth=8)
    assert stats.n_vertices > 0 and stats.n_triangles > 0
    # sanity: still reconstructs something sphere-shaped, same as the estimate_normals path
    verts = np.asarray(mesh.vertices)
    radii = np.linalg.norm(verts, axis=1)
    assert 1.7 < radii.mean() < 2.3


def test_supplied_normals_wrong_shape_raises():
    positions, colors = _noisy_sphere(n=500, seed=11)
    bad_normals = np.zeros((positions.shape[0] - 1, 3))
    with pytest.raises(ValueError):
        points_to_mesh(positions, colors, normals=bad_normals, method="poisson")


def test_supplied_normals_are_filtered_by_keep_mask_consistently():
    """normals must be indexed by the SAME keep_mask as positions/colors -- a
    misalignment here would silently attach the wrong normal to the wrong
    point, which would not raise but WOULD quietly wreck reconstruction quality."""
    positions, colors = _noisy_sphere(n=2000, seed=12)
    normals = _sphere_gaussian_normals(positions)
    rng = np.random.default_rng(12)
    keep_mask = rng.uniform(0, 1, positions.shape[0]) >= 0.4

    _, stats = points_to_mesh(positions, colors, keep_mask=keep_mask, normals=normals,
                               method="poisson", poisson_depth=7)
    assert stats.n_points_after_filter == int(keep_mask.sum())


def test_statistical_outlier_removal_drops_far_away_points():
    positions, colors = _noisy_sphere(n=2000, seed=13)
    rng = np.random.default_rng(13)
    far_outliers = rng.normal(scale=1.0, size=(50, 3)) + np.array([50.0, 50.0, 50.0])
    positions_with_outliers = np.concatenate([positions, far_outliers], axis=0)
    colors_with_outliers = np.concatenate([colors, np.ones((50, 3)) * 0.5], axis=0)

    _, stats_removed = points_to_mesh(positions_with_outliers, colors_with_outliers, method="poisson",
                                       poisson_depth=8, statistical_outlier_neighbors=20, statistical_outlier_std_ratio=1.0)
    _, stats_kept = points_to_mesh(positions_with_outliers, colors_with_outliers, method="poisson",
                                    poisson_depth=8, statistical_outlier_neighbors=0)

    assert stats_removed.n_points_after_outlier_removal < stats_kept.n_points_after_outlier_removal
    # with the far-away cluster gone, the reconstructed surface should sit near the
    # sphere again, not stretch out to include the outlier region
    assert max(stats_removed.bbox_max) < 40


def test_min_component_vertices_removes_debris():
    """A dense sphere plus a handful of small, spatially isolated 'floater' clusters --
    the isolated clusters should form their own tiny connected components that
    min_component_vertices removes, while the main sphere survives untouched."""
    positions, colors = _noisy_sphere(n=4000, radius=2.0, seed=14)
    rng = np.random.default_rng(14)
    floaters = []
    for center in ([20, 0, 0], [0, 20, 0], [-20, -20, 0]):
        cluster = rng.normal(scale=0.05, size=(15, 3)) + np.array(center, dtype=float)
        floaters.append(cluster)
    floaters = np.concatenate(floaters, axis=0)
    all_positions = np.concatenate([positions, floaters], axis=0)
    all_colors = np.concatenate([colors, np.ones((len(floaters), 3)) * 0.5], axis=0)

    _, stats_raw = points_to_mesh(all_positions, all_colors, method="poisson", poisson_depth=8,
                                   statistical_outlier_neighbors=0, density_trim_quantile=0.0)
    assert stats_raw.n_connected_components >= 2  # main sphere + at least one floater island

    # each 15-point floater is reconstructed by Poisson as a small but not tiny blob
    # (~70-80 vertices at depth=8 -- Poisson's octree tessellation isn't 1:1 with input
    # point count); the threshold below sits between that and the main sphere's (~2500).
    _, stats_cleaned = points_to_mesh(all_positions, all_colors, method="poisson", poisson_depth=8,
                                       statistical_outlier_neighbors=0, density_trim_quantile=0.0,
                                       min_component_vertices=150)
    assert stats_cleaned.n_components_removed_as_debris >= 1
    assert stats_cleaned.n_vertices < stats_raw.n_vertices
    assert stats_cleaned.largest_component_fraction >= stats_raw.largest_component_fraction


def test_keep_largest_n_components():
    positions, colors = _noisy_sphere(n=3000, radius=2.0, seed=15)
    rng = np.random.default_rng(15)
    floater = rng.normal(scale=0.05, size=(20, 3)) + np.array([30.0, 0.0, 0.0])
    all_positions = np.concatenate([positions, floater], axis=0)
    all_colors = np.concatenate([colors, np.ones((20, 3)) * 0.5], axis=0)

    _, stats = points_to_mesh(all_positions, all_colors, method="poisson", poisson_depth=8,
                               statistical_outlier_neighbors=0, density_trim_quantile=0.0,
                               keep_largest_n_components=1)
    assert stats.n_connected_components == 1
    assert stats.largest_component_fraction == 1.0


def test_topology_stats_populated_and_sane():
    positions, colors = _noisy_sphere(n=3000, seed=16)
    _, stats = points_to_mesh(positions, colors, method="poisson", poisson_depth=8)
    assert stats.n_connected_components >= 1
    assert 0.0 < stats.largest_component_fraction <= 1.0
    assert stats.surface_area > 0
    assert stats.triangle_area_p50 > 0
    assert stats.triangle_area_p50 <= stats.triangle_area_p99
    assert len(stats.bbox_min) == 3 and len(stats.bbox_max) == 3
    report = stats.format_report()
    assert "connected components" in report and "surface area" in report
