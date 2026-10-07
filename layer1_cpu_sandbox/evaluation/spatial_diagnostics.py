"""
spatial_diagnostics.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Turns the per-point confidence signal (`reconstruction/confidence.py`) into a
per-REGION summary: "this XY cell has N points, mean confidence C, and a
Low-confidence fraction of F". This is deliberately NOT the same thing as
`occlusion.py`'s coverage report -- `occlusion.py` needs the renderer's
ground-truth visibility trace and only ever runs on the synthetic dataset;
this module needs only `positions`/`confidence`/`band`, which exist for a
REAL reconstruction too (`real_pipeline.RealPipelineResult`), so it is the
one honest "where is this reconstruction weak" diagnostic available with no
ground truth at all.

Existing for one purpose: giving `layer4_agents`'s Reconstruction/Evaluation
agents something concrete to point at instead of a vague "quality is poor" --
e.g. turning a bare aggregate confidence number into "the SE region (12% of
kept points) is Low-confidence; the rest is Moderate-to-High", the same kind
of specific, falsifiable statement `occlusion.py`'s docstring argues for.

This module never invents geometry and never merges cells -- it is a
read-only summary over whatever points the pipeline already decided to
keep (`gate.keep_mask`-filtered positions/confidence), exactly like
`occlusion.py` is eval-only and `confidence.py`'s gating is the only thing
allowed to prune.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np
import pandas as pd

# Order matches confidence.py's BAND_* constants; kept as plain strings here so this
# module has no import-time dependency on reconstruction/confidence.py's internals.
BAND_ORDER = ["High", "Moderate", "Low / insufficient observation"]


@dataclasses.dataclass
class GridCell:
    cell_x: int
    cell_y: int
    center_x: float
    center_y: float
    n_points: int
    mean_confidence: float             # NaN if n_points == 0
    band_counts: dict                  # {band_name: count}
    low_confidence_fraction: float      # NaN if n_points == 0


@dataclasses.dataclass
class SpatialDiagnostic:
    cells: List[GridCell]
    cell_size: float
    x_min: float
    y_min: float
    n_empty_cells: int                  # cells inside the point-cloud's own bounding box with 0 points
    overall_mean_confidence: float


def compute_spatial_confidence_grid(positions: np.ndarray, confidence: np.ndarray,
                                      band: np.ndarray, cell_size: float = 10.0) -> SpatialDiagnostic:
    """positions: (M,3) world coords (only x,y used -- this is a plan-view grid, matching
    report_generator.py's existing plan-view trajectory/coverage figure).
    confidence: (M,) in [0,1]. band: (M,) dtype=object, values from confidence.py's BAND_*.
    An empty `positions` returns a single degenerate cell at the origin with n_points=0,
    the same "honestly report nothing rather than crash or fabricate" behavior
    `prototype_point_repr.fuse_point_cloud` uses for zero input points.
    """
    if positions.shape[0] == 0:
        return SpatialDiagnostic(cells=[], cell_size=cell_size, x_min=0.0, y_min=0.0,
                                   n_empty_cells=0, overall_mean_confidence=float("nan"))

    xs, ys = positions[:, 0], positions[:, 1]
    x_min, y_min = float(xs.min()), float(ys.min())
    cell_x = np.floor((xs - x_min) / cell_size).astype(np.int64)
    cell_y = np.floor((ys - y_min) / cell_size).astype(np.int64)

    # groupby, not a gx/gy double loop over every possible cell: same reason
    # prototype_point_repr.fuse_point_cloud voxelizes via pandas groupby rather than a
    # dense 3D loop -- cost tracks the number of OCCUPIED cells, not the bounding grid's
    # full extent, which matters once a flight path spans a wide, sparsely-covered area.
    df = pd.DataFrame({"cell_x": cell_x, "cell_y": cell_y, "confidence": confidence, "band": band})
    g = df.groupby(["cell_x", "cell_y"], sort=True)
    agg = g.agg(n_points=("confidence", "size"), mean_confidence=("confidence", "mean")).reset_index()
    band_counts_df = g["band"].value_counts().unstack(fill_value=0)
    for b in BAND_ORDER:
        if b not in band_counts_df.columns:
            band_counts_df[b] = 0

    cells: List[GridCell] = []
    for _, row in agg.iterrows():
        gx, gy = int(row["cell_x"]), int(row["cell_y"])
        n_pts = int(row["n_points"])
        band_counts = {b: int(band_counts_df.loc[(gx, gy), b]) for b in BAND_ORDER}
        low_frac = band_counts.get(BAND_ORDER[-1], 0) / n_pts
        cells.append(GridCell(
            cell_x=gx, cell_y=gy,
            center_x=x_min + (gx + 0.5) * cell_size, center_y=y_min + (gy + 0.5) * cell_size,
            n_points=n_pts, mean_confidence=float(row["mean_confidence"]),
            band_counts=band_counts, low_confidence_fraction=float(low_frac),
        ))

    # cell_x/cell_y both provably hit 0 (the point achieving x_min/y_min maps to cell 0),
    # so max()+1 is the true grid span -- this is only a count of never-visited cells
    # within that bounding span, not a claim about anything outside the observed extent.
    n_cx = int(cell_x.max()) + 1
    n_cy = int(cell_y.max()) + 1
    n_empty = (n_cx * n_cy) - len(cells)

    return SpatialDiagnostic(cells=cells, cell_size=cell_size, x_min=x_min, y_min=y_min,
                               n_empty_cells=n_empty, overall_mean_confidence=float(confidence.mean()))


def _compass_label(center_x: float, center_y: float, mean_x: float, mean_y: float) -> str:
    """Coarse compass direction of a cell relative to the point cloud's own centroid --
    intentionally relative (there is no true-north guarantee without a georeferencing
    step this repo doesn't claim yet), used only to make an agent's text diagnosis
    readable ("the southern part of the reconstruction"), never as a metric-accuracy claim."""
    dx, dy = center_x - mean_x, center_y - mean_y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "central"
    ns = "north" if dy > 0 else "south"
    ew = "east" if dx > 0 else "west"
    if abs(dy) < 0.4 * abs(dx):
        return ew
    if abs(dx) < 0.4 * abs(dy):
        return ns
    return f"{ns}{ew}"


def summarize_weak_regions(diag: SpatialDiagnostic, n: int = 3,
                             min_points_to_flag: int = 5,
                             min_low_confidence_fraction: float = 0.2) -> List[dict]:
    """Top-`n` weakest cells by low_confidence_fraction (ties broken by fewer points),
    restricted to cells with at least `min_points_to_flag` points -- a cell with 1-2
    points and a bad mean isn't "a weak region", it's just a small sample, and
    reporting it as a region would be exactly the kind of overclaim `occlusion.py`'s
    docstring warns against.

    Also requires `low_confidence_fraction > min_low_confidence_fraction`: a healthy cell
    must never be reported just to pad the list out to `n` entries. Without this a scene
    with only one truly weak cell and `n=3` requested would silently report two perfectly
    fine cells alongside it as if they too needed attention.
    """
    if not diag.cells:
        return []
    mean_x = float(np.mean([c.center_x for c in diag.cells]))
    mean_y = float(np.mean([c.center_y for c in diag.cells]))
    candidates = [c for c in diag.cells
                  if c.n_points >= min_points_to_flag
                  and c.low_confidence_fraction > min_low_confidence_fraction]
    ranked = sorted(candidates, key=lambda c: (-c.low_confidence_fraction, c.n_points))
    out = []
    for c in ranked[:n]:
        out.append({
            "direction": _compass_label(c.center_x, c.center_y, mean_x, mean_y),
            "n_points": c.n_points,
            "mean_confidence": round(c.mean_confidence, 3),
            "low_confidence_fraction": round(c.low_confidence_fraction, 3),
            "band_counts": c.band_counts,
        })
    return out
