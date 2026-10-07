"""
ablation_framework.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Runs a list of `PipelineConfig`s against one dataset, evaluates every one
with the exact same `evaluate_pipeline_result` code path, and returns a
flat pandas DataFrame -- one row per config, every metric family as
columns. This is the one function report_generator.py and
run_experiment.py both consume.
"""
from __future__ import annotations

import pandas as pd

from ..synthetic.dataset_builder import SyntheticDroneDataset
from ..pipeline import run_pipeline, PipelineConfig
from ..evaluate_result import evaluate_pipeline_result


def _flatten(cfg: PipelineConfig, ev, n_frames_total: int) -> dict:
    eff = ev.efficiency
    return {
        "run_name": ev.run_name,
        "n_frames_total": n_frames_total,
        "n_frames_kept": eff["counts"].get("n_frames_kept"),
        "n_points_raw_fused": eff["counts"].get("n_points_raw_fused"),
        "n_points_final": eff["counts"].get("n_points_after_gating"),
        "total_time_s": eff["total_seconds"],
        # --- A: visual (image-space) fidelity ---
        "psnr_train_view": ev.visual_train_view.psnr,
        "ssim_train_view": ev.visual_train_view.ssim,
        "ms_ssim_proxy_train_view": ev.visual_train_view.ms_ssim_proxy,
        "psnr_novel_view": ev.visual_novel_view.psnr,
        "ssim_novel_view": ev.visual_novel_view.ssim,
        "ms_ssim_proxy_novel_view": ev.visual_novel_view.ms_ssim_proxy,
        # --- B: geometric accuracy ---
        "chamfer_distance_m": ev.geometry.chamfer_distance,
        "point_to_point_rmse_m": ev.geometry.point_to_point_rmse,
        "point_to_plane_rmse_m": ev.geometry.point_to_plane_rmse,
        # --- C: metric / georeferencing accuracy ---
        "mean_abs_height_error_m": ev.metric_accuracy.mean_abs_height_error_m,
        "mean_pct_height_error": ev.metric_accuracy.mean_pct_height_error,
        "mean_abs_distance_error_m": ev.metric_accuracy.mean_abs_distance_error_m,
        "control_point_rmse_m": ev.metric_accuracy.control_point_rmse_m,
        "control_points_matched": f"{ev.metric_accuracy.n_control_points_matched}/{ev.metric_accuracy.n_control_points_total}",
        # --- D: completeness ---
        "surface_completeness": ev.completeness.surface_completeness,
        "surface_completeness_high_conf": ev.completeness.surface_completeness_high_conf,
        "completeness_vs_theoretical_ceiling": ev.completeness.fraction_of_theoretical_ceiling,
        "coverage_never_observable": ev.coverage_never_observable,
        "coverage_observable_but_dropped": ev.coverage_observable_but_dropped,
        "coverage_observed": ev.coverage_observed,
    }


def run_configs(dataset: SyntheticDroneDataset, configs: list) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        result = run_pipeline(dataset, cfg)
        ev = evaluate_pipeline_result(dataset, result, run_name=cfg.name)
        rows.append(_flatten(cfg, ev, n_frames_total=len(dataset.frames)))
    return pd.DataFrame(rows)
