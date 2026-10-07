"""
metrics_metric_accuracy.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Section C of the evaluation framework. Because this pipeline fuses GPS
directly into the pose estimate (sensor_fusion.py), the reconstruction
already lives in the SAME approximately-georeferenced coordinate frame as
the ground truth -- unlike vision-only SfM, which needs a separate
similarity-transform alignment step before any metric claim is possible.
That is a real, structural side-benefit of doing GPS/IMU fusion up front,
and it is what makes the numbers below a direct "reconstructed vs true"
comparison rather than a comparison after best-fit alignment (which can
hide systematic bias).
"""
from __future__ import annotations

import dataclasses
import numpy as np
from scipy.spatial import cKDTree


@dataclasses.dataclass
class BuildingMeasurement:
    building_id: int
    true_height_m: float
    measured_height_m: float
    height_error_m: float
    height_error_pct: float
    n_points_used: int


@dataclasses.dataclass
class MetricAccuracyResult:
    building_measurements: list
    mean_abs_height_error_m: float
    mean_pct_height_error: float
    n_buildings_measurable: int
    n_buildings_total: int
    inter_building_distance_errors_m: list   # [(id_a, id_b, true_d, measured_d, error_m), ...]
    mean_abs_distance_error_m: float
    control_point_rmse_m: float              # RMSE of matched control-point (roof-corner-ish) positions
    control_point_mean_m: float
    n_control_points_matched: int
    n_control_points_total: int


def _robust_height(z_values: np.ndarray) -> float:
    if z_values.shape[0] < 3:
        return float(z_values.max() - z_values.min()) if z_values.shape[0] else float("nan")
    return float(np.percentile(z_values, 95) - np.percentile(z_values, 5))


def measure_building_heights(recon_positions: np.ndarray, buildings_meta: list,
                              margin_m: float = 1.5, min_points: int = 5) -> list:
    results = []
    for b in buildings_meta:
        x0, x1 = b["x"] - margin_m, b["x"] + b["w"] + margin_m
        y0, y1 = b["y"] - margin_m, b["y"] + b["d"] + margin_m
        m = (recon_positions[:, 0] >= x0) & (recon_positions[:, 0] <= x1) & \
            (recon_positions[:, 1] >= y0) & (recon_positions[:, 1] <= y1)
        n = int(m.sum())
        if n < min_points:
            results.append(BuildingMeasurement(b["id"], b["h"], float("nan"), float("nan"), float("nan"), n))
            continue
        measured = _robust_height(recon_positions[m, 2])
        err = measured - b["h"]
        results.append(BuildingMeasurement(b["id"], b["h"], measured, err,
                                            100.0 * err / max(b["h"], 1e-6), n))
    return results


def measure_inter_building_distances(recon_positions: np.ndarray, buildings_meta: list,
                                      control: dict, margin_m: float = 1.5, min_points: int = 5):
    centers = {}
    for b in buildings_meta:
        x0, x1 = b["x"] - margin_m, b["x"] + b["w"] + margin_m
        y0, y1 = b["y"] - margin_m, b["y"] + b["d"] + margin_m
        m = (recon_positions[:, 0] >= x0) & (recon_positions[:, 0] <= x1) & \
            (recon_positions[:, 1] >= y0) & (recon_positions[:, 1] <= y1)
        if m.sum() >= min_points:
            centers[b["id"]] = recon_positions[m, :2].mean(axis=0)

    out = []
    for (ida, idb), true_d in control["inter_building_center_distances_m"].items():
        if ida in centers and idb in centers:
            measured_d = float(np.linalg.norm(centers[ida] - centers[idb]))
            out.append((ida, idb, true_d, measured_d, measured_d - true_d))
    return out


def measure_control_points(recon_positions: np.ndarray, buildings_meta: list,
                            match_radius_m: float = 4.0):
    """Treats each building's roof-center apex as a synthetic 'control point' (the kind of
    known survey marker a real deployment would use RTK/GCPs for), and measures how far the
    nearest reconstructed point falls from its true 3-D location."""
    if recon_positions.shape[0] == 0:
        return [], 0
    tree = cKDTree(recon_positions)
    errors = []
    for b in buildings_meta:
        cp = np.array([b["center_xy"][0], b["center_xy"][1], b["h"]])
        d, idx = tree.query(cp, k=1)
        if d <= match_radius_m:
            errors.append(float(d))
    return errors, len(buildings_meta)


def evaluate_metric_accuracy(recon_positions: np.ndarray, buildings_meta: list, control: dict) -> MetricAccuracyResult:
    heights = measure_building_heights(recon_positions, buildings_meta)
    valid_h = [h for h in heights if not np.isnan(h.height_error_m)]
    mean_abs_h = float(np.mean([abs(h.height_error_m) for h in valid_h])) if valid_h else float("nan")
    mean_pct_h = float(np.mean([abs(h.height_error_pct) for h in valid_h])) if valid_h else float("nan")

    dist_errors = measure_inter_building_distances(recon_positions, buildings_meta, control)
    mean_abs_d = float(np.mean([abs(e[4]) for e in dist_errors])) if dist_errors else float("nan")

    cp_errors, n_cp_total = measure_control_points(recon_positions, buildings_meta)
    cp_rmse = float(np.sqrt(np.mean(np.square(cp_errors)))) if cp_errors else float("nan")
    cp_mean = float(np.mean(cp_errors)) if cp_errors else float("nan")

    return MetricAccuracyResult(
        building_measurements=heights, mean_abs_height_error_m=mean_abs_h, mean_pct_height_error=mean_pct_h,
        n_buildings_measurable=len(valid_h), n_buildings_total=len(buildings_meta),
        inter_building_distance_errors_m=dist_errors, mean_abs_distance_error_m=mean_abs_d,
        control_point_rmse_m=cp_rmse, control_point_mean_m=cp_mean,
        n_control_points_matched=len(cp_errors), n_control_points_total=n_cp_total,
    )
