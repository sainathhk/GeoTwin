"""
run_experiment.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested, CPU-EXECUTED (see /outputs of a run)

Single command that builds a synthetic single-pass drone dataset, runs the
full pipeline, runs the ablation sweep, evaluates everything against
ground truth, and writes every artifact (CSV/markdown tables, PNG figures,
a PLY point cloud) to --out_dir. This is what "Layer 1 must produce REAL
computed results" means executed end to end.

Usage:
    python3 -m layer1_cpu_sandbox.run_experiment --out_dir outputs/run1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .synthetic.dataset_builder import build_dataset
from .pipeline import run_pipeline, PipelineConfig
from .evaluate_result import evaluate_pipeline_result
from .evaluation.baseline_framework import get_ablation_configs, get_baseline_configs
from .evaluation.ablation_framework import run_configs
from .evaluation.report_generator import (save_ablation_table, plot_confidence_vs_error,
                                            plot_trajectory_and_coverage, plot_qualitative_comparison)
from .reconstruction.occlusion import analyze_coverage
from .reconstruction.mesh_export import export_point_cloud_to_obj, MeshExportError
from .utils import save_ply, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/home/claude/sih2026-3d-recon/layer1_cpu_sandbox/outputs/run1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extent", type=float, default=40.0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    t_start = time.perf_counter()

    print(f"[1/6] Building synthetic single-pass drone dataset (seed={args.seed})...")
    dataset = build_dataset(seed=args.seed, extent=args.extent)
    print(f"      {len(dataset.frames)} frames, {len(dataset.scene.triangles)} scene triangles, "
          f"{len(dataset.scene.buildings)} buildings, {dataset.gt_points.points.shape[0]} GT surface points")

    print("[2/6] Running full method pipeline...")
    full_cfg = PipelineConfig(name="full_method")
    full_result = run_pipeline(dataset, full_cfg)
    full_eval = evaluate_pipeline_result(dataset, full_result, run_name="full_method")
    print(f"      kept {len(full_result.kept_frame_indices)}/{len(dataset.frames)} frames, "
          f"{full_result.final_positions.shape[0]} final points after confidence gating")

    print("[3/6] Running ablation sweep (cumulative module build-up)...")
    ablation_df = run_configs(dataset, get_ablation_configs())
    ablation_csv, ablation_md = save_ablation_table(ablation_df, args.out_dir, filename="ablation_table")
    print(f"      saved {ablation_csv}")

    print("[4/6] Running baseline sweep (single-factor comparisons)...")
    baseline_df = run_configs(dataset, get_baseline_configs())
    baseline_csv, baseline_md = save_ablation_table(baseline_df, args.out_dir, filename="baseline_table")
    print(f"      saved {baseline_csv}")

    print("[5/6] Generating figures (confidence validation, coverage, qualitative)...")
    corr, decile_table = plot_confidence_vs_error(dataset, full_result,
                                                    os.path.join(args.out_dir, "confidence_vs_error.png"))
    print(f"      confidence-vs-error Pearson r = {corr:.4f} (see confidence_vs_error.png)")

    kept_view_angle = [dataset.per_triangle_view_angle_by_frame[i] for i in full_result.kept_frame_indices]
    coverage = analyze_coverage(dataset.scene, dataset.per_triangle_view_angle_by_frame, kept_view_angle)
    plot_trajectory_and_coverage(dataset, coverage.per_triangle_status,
                                  os.path.join(args.out_dir, "trajectory_coverage.png"))

    mid_frame = full_result.kept_frame_indices[len(full_result.kept_frame_indices) // 2]
    plot_qualitative_comparison(dataset, full_result, mid_frame,
                                 os.path.join(args.out_dir, "qualitative_comparison.png"))

    print("[6/6] Saving point cloud (.ply/.obj) and summary JSON...")
    save_ply(os.path.join(args.out_dir, "reconstruction_full_method.ply"),
              full_result.final_positions, full_result.final_colors)
    save_ply(os.path.join(args.out_dir, "ground_truth_surface.ply"),
              dataset.gt_points.points, dataset.gt_points.colors)

    # Same points as reconstruction_full_method.ply, turned into a triangle mesh
    # (see layer1_cpu_sandbox/reconstruction/mesh_export.py for why/how) -- an
    # extra, easier-to-open deliverable, not a replacement for the .ply. Confidence
    # already gated which points reached `final_positions` (see pipeline.py), so
    # there's no separate keep_mask to apply here.
    mesh_stats = None
    try:
        mesh_stats = export_point_cloud_to_obj(
            os.path.join(args.out_dir, "reconstruction_full_method.obj"),
            full_result.final_positions, full_result.final_colors)
        print(f"      mesh: {mesh_stats.n_vertices} vertices, {mesh_stats.n_triangles} triangles "
              f"from {mesh_stats.n_points_after_filter} points -> reconstruction_full_method.obj")
    except MeshExportError as e:
        print(f"      WARNING: mesh export skipped ({e}); reconstruction_full_method.ply is still available.")

    summary = {
        "seed": args.seed,
        "n_frames_total": len(dataset.frames),
        "n_frames_kept": len(full_result.kept_frame_indices),
        "n_scene_triangles": len(dataset.scene.triangles),
        "n_buildings": len(dataset.scene.buildings),
        "n_gt_surface_points": int(dataset.gt_points.points.shape[0]),
        "n_final_reconstructed_points": int(full_result.final_positions.shape[0]),
        "mesh_n_vertices": mesh_stats.n_vertices if mesh_stats else None,
        "mesh_n_triangles": mesh_stats.n_triangles if mesh_stats else None,
        "confidence_vs_error_pearson_r": corr,
        "coverage_never_observable": coverage.fraction_never_observable,
        "coverage_observable_but_dropped": coverage.fraction_observable_but_dropped,
        "coverage_observed": coverage.fraction_observed,
        "visual_novel_view_psnr": full_eval.visual_novel_view.psnr,
        "visual_novel_view_ssim": full_eval.visual_novel_view.ssim,
        "visual_novel_view_ms_ssim": full_eval.visual_novel_view.ms_ssim_proxy,
        "visual_novel_view_lpips": full_eval.visual_novel_view.lpips,  # NaN in Env A, real in Env B (if lpips installed)
        "geometry_chamfer_distance_m": full_eval.geometry.chamfer_distance,
        "metric_mean_abs_height_error_m": full_eval.metric_accuracy.mean_abs_height_error_m,
        "metric_control_point_rmse_m": full_eval.metric_accuracy.control_point_rmse_m,
        "completeness_surface": full_eval.completeness.surface_completeness,
        "total_pipeline_time_s": full_result.efficiency["total_seconds"],
        "total_experiment_wall_time_s": round(time.perf_counter() - t_start, 2),
        "NOTE": "All numbers above are real, computed on a Layer-1 CPU-sandbox synthetic dataset. "
                "They validate pipeline logic and the confidence-error relationship; they are NOT "
                "representative of final GPU/learned-model performance on real drone footage.",
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s. Artifacts written to: {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
