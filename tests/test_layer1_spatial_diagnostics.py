"""Unit tests for layer1_cpu_sandbox.evaluation.spatial_diagnostics."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_cpu_sandbox.evaluation.spatial_diagnostics import (
    compute_spatial_confidence_grid, summarize_weak_regions, BAND_ORDER,
)

HIGH, MOD, LOW = BAND_ORDER


def test_empty_input_returns_degenerate_diagnostic_not_crash():
    diag = compute_spatial_confidence_grid(np.zeros((0, 3)), np.zeros(0), np.array([], dtype=object))
    assert diag.cells == []
    assert np.isnan(diag.overall_mean_confidence)


def test_single_cluster_forms_one_cell():
    rng = np.random.default_rng(0)
    positions = rng.normal(loc=[5, 5, 0], scale=0.5, size=(50, 3))
    confidence = np.full(50, 0.8)
    band = np.array([HIGH] * 50, dtype=object)
    diag = compute_spatial_confidence_grid(positions, confidence, band, cell_size=10.0)
    assert len(diag.cells) == 1
    assert diag.cells[0].n_points == 50
    assert abs(diag.cells[0].mean_confidence - 0.8) < 1e-6


def test_two_well_separated_clusters_are_never_blended_into_one_cell():
    # The grid's origin is `x_min`/`y_min`, which comes from wherever the data happens to
    # start -- so a cluster's position relative to grid *lines* is only knowable once
    # x_min/y_min are known, not from the cluster's raw coordinates alone. Rather than
    # gamble on a coordinate being "safely" off a boundary, cluster b's center is placed
    # exactly at a cell CENTER derived from cluster a's actual x_min/y_min (as far from
    # any grid line as a point can be), so this test is deterministic, not lucky.
    cell_size = 5.0
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[1.5, 1.5, 0], scale=0.2, size=(30, 3))
    x_min, y_min = float(a[:, 0].min()), float(a[:, 1].min())
    b_center_x = x_min + 20.5 * cell_size   # cell-center, 20 cells over
    b_center_y = y_min + 19.5 * cell_size
    b = rng.normal(loc=[b_center_x, b_center_y, 0], scale=0.2, size=(30, 3))
    positions = np.vstack([a, b])
    confidence = np.concatenate([np.full(30, 0.9), np.full(30, 0.2)])
    band = np.array([HIGH] * 30 + [LOW] * 30, dtype=object)
    diag = compute_spatial_confidence_grid(positions, confidence, band, cell_size=cell_size)
    # However many cells the grid produced, no single cell may average the two clusters
    # together -- they are 20+ cells apart, far past any one cell's width.
    for cell in diag.cells:
        assert cell.mean_confidence > 0.85 or cell.mean_confidence < 0.25
    assert sum(c.n_points for c in diag.cells) == 60
    assert len(diag.cells) == 2  # centered placement leaves no room for boundary-straddling here


def test_band_counts_sum_to_n_points_per_cell():
    rng = np.random.default_rng(1)
    positions = rng.normal(loc=[0, 0, 0], scale=3.0, size=(200, 3))
    confidence = rng.uniform(0, 1, size=200)
    bands = rng.choice([HIGH, MOD, LOW], size=200)
    diag = compute_spatial_confidence_grid(positions, confidence, bands, cell_size=2.0)
    for cell in diag.cells:
        assert sum(cell.band_counts.values()) == cell.n_points
        assert set(cell.band_counts.keys()) == set(BAND_ORDER)


def test_low_confidence_fraction_matches_hand_count():
    positions = np.array([[0, 0, 0]] * 10, dtype=float)
    confidence = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    band = np.array([HIGH] * 7 + [LOW] * 3, dtype=object)
    diag = compute_spatial_confidence_grid(positions, confidence, band, cell_size=10.0)
    assert len(diag.cells) == 1
    assert abs(diag.cells[0].low_confidence_fraction - 0.3) < 1e-9


def test_summarize_weak_regions_excludes_tiny_cells():
    # Same "place at an explicit cell-center derived from the first cluster's own x_min/
    # y_min" approach as the two-cluster test above, so none of the three clusters can
    # straddle a grid boundary by bad luck.
    cell_size = 5.0
    rng = np.random.default_rng(2)
    strong = rng.normal(loc=[1.5, 1.5, 0], scale=0.3, size=(40, 3))
    x_min, y_min = float(strong[:, 0].min()), float(strong[:, 1].min())
    tiny_center = (x_min + 10.5 * cell_size, y_min + 0.5 * cell_size)
    real_center = (x_min + 0.5 * cell_size, y_min + 10.5 * cell_size)
    weak_but_tiny = rng.normal(loc=[tiny_center[0], tiny_center[1], 0], scale=0.3, size=(2, 3))
    weak_and_real = rng.normal(loc=[real_center[0], real_center[1], 0], scale=0.3, size=(20, 3))
    positions = np.vstack([strong, weak_but_tiny, weak_and_real])
    confidence = np.concatenate([np.full(40, 0.9), np.full(2, 0.05), np.full(20, 0.1)])
    band = np.array([HIGH] * 40 + [LOW] * 2 + [LOW] * 20, dtype=object)
    diag = compute_spatial_confidence_grid(positions, confidence, band, cell_size=cell_size)
    weak = summarize_weak_regions(diag, n=3, min_points_to_flag=5)
    # n=3 is requested but only one cell is actually weak -- the 40-point strong cluster
    # (low_confidence_fraction == 0.0) must never be padded in just to reach n.
    assert len(weak) == 1
    assert weak[0]["n_points"] == 20
    assert weak[0]["low_confidence_fraction"] == 1.0


def test_summarize_weak_regions_on_empty_diagnostic():
    diag = compute_spatial_confidence_grid(np.zeros((0, 3)), np.zeros(0), np.array([], dtype=object))
    assert summarize_weak_regions(diag) == []


def test_compass_directions_are_plausible():
    # Four clusters at the cardinal extremes of a shared centroid.
    rng = np.random.default_rng(3)
    n_pts, north, south, east, west = 25, rng.normal([0, 50, 0], 0.2, (25, 3)), None, None, None
    south = rng.normal([0, -50, 0], 0.2, (n_pts, 3))
    east = rng.normal([50, 0, 0], 0.2, (n_pts, 3))
    west = rng.normal([-50, 0, 0], 0.2, (n_pts, 3))
    center = rng.normal([0, 0, 0], 0.2, (n_pts, 3))
    positions = np.vstack([north, south, east, west, center])
    # Make the 4 outer clusters the "weak" ones so summarize_weak_regions surfaces them.
    confidence = np.concatenate([np.full(n_pts, 0.1)] * 4 + [np.full(n_pts, 0.95)])
    band = np.array([LOW] * (4 * n_pts) + [HIGH] * n_pts, dtype=object)
    diag = compute_spatial_confidence_grid(positions, confidence, band, cell_size=8.0)
    weak = summarize_weak_regions(diag, n=4, min_points_to_flag=5)
    directions = {w["direction"] for w in weak}
    assert directions.issubset({"north", "south", "east", "west", "northeast", "northwest",
                                  "southeast", "southwest", "central"})
    assert len(directions) >= 3  # they shouldn't all collapse onto the same label
