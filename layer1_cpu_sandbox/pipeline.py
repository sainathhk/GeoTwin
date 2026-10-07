"""
pipeline.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Wires every module above (frame quality -> sensor fusion -> depth ->
dynamic filtering -> point fusion -> confidence -> gating) into one
configurable run. Every toggle in `PipelineConfig` corresponds to exactly
one row of the ablation table requested in the brief: flipping it off
reproduces "baseline without module X" without duplicating pipeline logic.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np
import cv2

from .synthetic.dataset_builder import SyntheticDroneDataset
from .perception.frame_quality import compute_frame_quality, select_frames
from .perception.sensor_fusion import fuse_gps_imu, naive_pose_from_raw_gps, FusedTrajectory
from .perception.dynamic_object_filter import detect_dynamic_pixels
from .reconstruction.depth_estimation import estimate_depth_for_sequence
from .reconstruction.prototype_point_repr import fuse_point_cloud, PrototypePointRepresentation
from .reconstruction.confidence import compute_confidence, gate_by_confidence, ConfidenceResult, ConfidenceGatedDecision
from .evaluation.metrics_efficiency import EfficiencyTracker


@dataclasses.dataclass
class PipelineConfig:
    name: str = "full_method"
    use_quality_filter: bool = True
    keep_fraction: float = 0.75
    min_overall: float = 0.15  # added for layer4_agents: was select_frames()'s hardcoded default;
                                # same value, now a real knob instead of buried in the call site.
                                # See layer4_agents/orchestrator.py's Frame Agent, which sets this
                                # via dataclasses.replace(initial_config, ..., min_overall=...).
    use_sensor_fusion: bool = True
    use_dynamic_filter: bool = True
    use_confidence_gating: bool = True
    voxel_size: float = 2.0
    min_consistency: float = 0.3
    depth_window: int = 2
    n_depths: int = 32
    depth_min: float = 8.0
    depth_max: float = 140.0
    prune_below: float = 0.12
    densify_above: float = 0.60


@dataclasses.dataclass
class PipelineResult:
    config: PipelineConfig
    points: PrototypePointRepresentation
    confidence: ConfidenceResult
    gate: ConfidenceGatedDecision
    kept_frame_indices: List[int]
    rejected_frame_indices: List[int]
    fused_traj: FusedTrajectory
    frame_quality_scores: list
    efficiency: dict
    final_positions: np.ndarray
    final_colors: np.ndarray
    final_confidence: np.ndarray


def run_pipeline(dataset: SyntheticDroneDataset, config: PipelineConfig = None) -> PipelineResult:
    cfg = config or PipelineConfig()
    tracker = EfficiencyTracker()
    n_total = len(dataset.frames)

    with tracker.track("frame_quality_scoring"):
        quality = [compute_frame_quality(f.rgb) for f in dataset.frames]
    if cfg.use_quality_filter:
        kept, rejected = select_frames(quality, keep_fraction=cfg.keep_fraction, min_overall=cfg.min_overall)
    else:
        kept, rejected = list(range(n_total)), []

    with tracker.track("sensor_fusion"):
        if cfg.use_sensor_fusion:
            fused_traj = fuse_gps_imu(dataset.sensor_trace, nominal_pitch_deg=dataset.nominal_pitch_deg,
                                       nominal_roll_deg=dataset.nominal_roll_deg)
        else:
            fused_traj = naive_pose_from_raw_gps(dataset.sensor_trace, nominal_pitch_deg=dataset.nominal_pitch_deg,
                                                  nominal_roll_deg=dataset.nominal_roll_deg)

    kept_grays = [cv2.cvtColor(dataset.frames[i].rgb, cv2.COLOR_RGB2GRAY) for i in kept]
    kept_poses = [fused_traj.camera_pose(i) for i in kept]
    kept_rgb = [dataset.frames[i].rgb for i in kept]
    kept_quality = [quality[i].overall for i in kept]
    kept_pose_conf = [fused_traj.pose_confidence(i) for i in kept]

    with tracker.track("depth_estimation_mvs"):
        depth_ests = estimate_depth_for_sequence(kept_grays, kept_poses, dataset.K, window=cfg.depth_window,
                                                   n_depths=cfg.n_depths, depth_min=cfg.depth_min,
                                                   depth_max=cfg.depth_max)

    dynamic_masks: Optional[List] = None
    with tracker.track("dynamic_object_filtering"):
        if cfg.use_dynamic_filter:
            dynamic_masks = []
            for li in range(len(kept)):
                nb = li + 2 if li + 2 < len(kept) else (li + 1 if li + 1 < len(kept) else None)
                if nb is None:
                    dynamic_masks.append(None)
                    continue
                res = detect_dynamic_pixels(kept_grays[li], kept_grays[nb], depth_ests[li].depth,
                                             kept_poses[li], kept_poses[nb], dataset.K)
                dynamic_masks.append(res)

    with tracker.track("point_fusion"):
        pts = fuse_point_cloud(depth_ests, dynamic_masks, kept_poses, kept_rgb, kept_quality, kept_pose_conf,
                                dataset.K, voxel_size=cfg.voxel_size, min_consistency=cfg.min_consistency)

    with tracker.track("confidence_estimation"):
        conf = compute_confidence(pts)

    with tracker.track("confidence_gating"):
        if cfg.use_confidence_gating:
            gate = gate_by_confidence(conf, prune_below=cfg.prune_below, densify_above=cfg.densify_above)
        else:
            keep_all = np.ones(pts.positions.shape[0], dtype=bool)
            gate = ConfidenceGatedDecision(keep_mask=keep_all,
                                            densify_mask=conf.confidence >= cfg.densify_above,
                                            flag_uncertain_mask=np.zeros_like(keep_all))

    tracker.set_count("n_frames_total", n_total)
    tracker.set_count("n_frames_kept", len(kept))
    tracker.set_count("n_frames_rejected", len(rejected))
    tracker.set_count("n_points_raw_fused", int(pts.positions.shape[0]))
    tracker.set_count("n_points_after_gating", int(gate.keep_mask.sum()))

    return PipelineResult(
        config=cfg, points=pts, confidence=conf, gate=gate,
        kept_frame_indices=kept, rejected_frame_indices=rejected, fused_traj=fused_traj,
        frame_quality_scores=quality, efficiency=tracker.report(),
        final_positions=pts.positions[gate.keep_mask], final_colors=pts.colors[gate.keep_mask],
        final_confidence=conf.confidence[gate.keep_mask],
    )
