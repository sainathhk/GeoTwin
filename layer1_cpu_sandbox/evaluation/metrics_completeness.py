"""
metrics_completeness.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Section D. A single-pass flight can NEVER achieve 100% surface
completeness by construction (module 9's whole point); the number that
matters is not "how close to 100%" but "does completeness match the
theoretical single-pass visibility ceiling from occlusion.py, or fall
short of even that?" -- the gap between the two is diagnostic, not just
descriptive.
"""
from __future__ import annotations

import dataclasses
import numpy as np
from scipy.spatial import cKDTree


@dataclasses.dataclass
class CompletenessResult:
    surface_completeness: float          # fraction of GT surface points with a reconstructed point nearby
    surface_completeness_high_conf: float  # same, but only counting HIGH-confidence reconstructed points
    completeness_threshold_m: float
    fraction_of_theoretical_ceiling: float  # completeness / (fraction of scene that was even observable)


def evaluate_completeness(recon_positions: np.ndarray, gt_points: np.ndarray,
                           observable_fraction: float, threshold_m: float = 1.5,
                           recon_confidence: np.ndarray = None, high_conf_mask: np.ndarray = None) -> CompletenessResult:
    if recon_positions.shape[0] == 0 or gt_points.shape[0] == 0:
        return CompletenessResult(0.0, 0.0, threshold_m, 0.0)

    tree = cKDTree(recon_positions)
    d, _ = tree.query(gt_points, k=1)
    completeness = float((d <= threshold_m).mean())

    completeness_hi = 0.0
    if high_conf_mask is not None and high_conf_mask.any():
        tree_hi = cKDTree(recon_positions[high_conf_mask])
        d_hi, _ = tree_hi.query(gt_points, k=1)
        completeness_hi = float((d_hi <= threshold_m).mean())

    ceiling = max(observable_fraction, 1e-6)
    return CompletenessResult(surface_completeness=completeness, surface_completeness_high_conf=completeness_hi,
                               completeness_threshold_m=threshold_m,
                               fraction_of_theoretical_ceiling=float(completeness / ceiling))
