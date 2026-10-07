"""
evaluate_result.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Takes one `PipelineResult` (see pipeline.py) plus the `SyntheticDroneDataset`
it was computed from, and produces every metric family from the brief in
one call. This is the single function baseline_framework.py and
ablation_framework.py both call, so every row of every table in the report
is guaranteed to come from the same evaluation code path.
"""
from __future__ import annotations

import dataclasses
import numpy as np

from .synthetic.dataset_builder import SyntheticDroneDataset
from .pipeline import PipelineResult
from .reconstruction.occlusion import analyze_coverage
from .reconstruction.point_renderer import render_point_cloud
from .evaluation.metrics_visual import evaluate_visual_fidelity, VisualFidelityResult
from .evaluation.metrics_geometry import evaluate_point_cloud_geometry, GeometryAccuracyResult
from .evaluation.metrics_metric_accuracy import evaluate_metric_accuracy, MetricAccuracyResult
from .evaluation.metrics_completeness import evaluate_completeness, CompletenessResult


@dataclasses.dataclass
class FullEvaluation:
    run_name: str
    visual_train_view: VisualFidelityResult    # rendered at a KEPT (reconstruction-contributing) pose
    visual_novel_view: VisualFidelityResult    # rendered at an interpolated, never-used pose
    geometry: GeometryAccuracyResult
    metric_accuracy: MetricAccuracyResult
    completeness: CompletenessResult
    coverage_never_observable: float
    coverage_observable_but_dropped: float
    coverage_observed: float
    efficiency: dict


def _interpolated_novel_pose(dataset: SyntheticDroneDataset, kept: list, fused_traj):
    """A pose strictly between two consecutive KEPT frames -- never used by any pipeline
    stage -- for a genuine held-out view-synthesis check."""
    if len(kept) < 2:
        i = kept[0]
        return dataset.frames[i].true_pose, dataset.frames[i].rgb_clean
    mid = len(kept) // 2
    ia, ib = kept[mid - 1], kept[mid]
    pa, pb = dataset.trajectory.camera_pose(ia), dataset.trajectory.camera_pose(ib)
    pos = 0.5 * (pa.position + pb.position)
    from .synthetic.camera_model import CameraPose
    # slerp-free small-angle blend of the two flight attitudes (adequate for closely-spaced samples)
    R = pa.R_wc if np.random.default_rng(0).random() < 0.5 else pb.R_wc
    novel_pose = CameraPose(position=pos, R_wc=0.5 * pa.R_wc + 0.5 * pb.R_wc)
    # re-orthonormalize via SVD (an average of rotations is not itself a rotation)
    U, _, Vt = np.linalg.svd(novel_pose.R_wc)
    novel_pose.R_wc = U @ Vt
    from .synthetic.renderer import render_frame
    from .synthetic.camera_model import Intrinsics
    gt_render = render_frame(dataset.scene, dataset.K, novel_pose)
    return novel_pose, gt_render.rgb


def evaluate_pipeline_result(dataset: SyntheticDroneDataset, result: PipelineResult,
                              run_name: str = "run") -> FullEvaluation:
    kept = result.kept_frame_indices

    # --- visual fidelity: train-view (a kept pose) ---
    train_idx = kept[len(kept) // 2]
    train_pose = dataset.frames[train_idx].true_pose
    rendered_train, mask_train = render_point_cloud(result.final_positions, result.final_colors, dataset.K, train_pose)
    visual_train = evaluate_visual_fidelity(rendered_train, dataset.frames[train_idx].rgb_clean, mask_train)

    # --- visual fidelity: novel view (interpolated, never used anywhere in the pipeline) ---
    novel_pose, novel_gt_rgb = _interpolated_novel_pose(dataset, kept, result.fused_traj)
    rendered_novel, mask_novel = render_point_cloud(result.final_positions, result.final_colors, dataset.K, novel_pose)
    visual_novel = evaluate_visual_fidelity(rendered_novel, novel_gt_rgb, mask_novel)

    # --- geometry ---
    geometry = evaluate_point_cloud_geometry(result.final_positions, dataset.gt_points.points,
                                              dataset.gt_points.normals)

    # --- metric / georeferencing accuracy ---
    metric_acc = evaluate_metric_accuracy(result.final_positions, dataset.scene.buildings,
                                           dataset.control_measurements)

    # --- coverage (theoretical single-pass ceiling) ---
    kept_view_angle = [dataset.per_triangle_view_angle_by_frame[i] for i in kept]
    coverage = analyze_coverage(dataset.scene, dataset.per_triangle_view_angle_by_frame, kept_view_angle)

    # --- completeness (against that ceiling) ---
    hi_mask = result.final_confidence >= 0.60
    completeness = evaluate_completeness(result.final_positions, dataset.gt_points.points,
                                          observable_fraction=coverage.fraction_observed,
                                          recon_confidence=result.final_confidence, high_conf_mask=hi_mask)

    return FullEvaluation(run_name=run_name, visual_train_view=visual_train, visual_novel_view=visual_novel,
                           geometry=geometry, metric_accuracy=metric_acc, completeness=completeness,
                           coverage_never_observable=coverage.fraction_never_observable,
                           coverage_observable_but_dropped=coverage.fraction_observable_but_dropped,
                           coverage_observed=coverage.fraction_observed, efficiency=result.efficiency)
