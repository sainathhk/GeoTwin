"""
real_pipeline.py
REAL-DATA INGESTION -- STATUS: implemented, CPU-tested against synthetic
mock data (tests/test_real_data.py). NOT yet run against a real dataset in
this environment.

Runs the SAME representation-agnostic modules `pipeline.py` uses (frame
quality, plane-sweep depth, dynamic-object filtering, point fusion,
confidence) against a `RealDroneDataset` instead of a
`SyntheticDroneDataset`. The only things that differ from `pipeline.py`:
  * No sensor-fusion step -- `RealDroneDataset` already carries poses built
    directly from DJI's onboard gimbal telemetry (see
    real_dataset_builder.py's module docstring for why that's used as-is
    rather than re-derived).
  * No ground-truth-dependent evaluation -- see `evaluate_real_result`
    below for what IS honestly checkable without GT.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np
import cv2

from .real_dataset_builder import RealDroneDataset
from ..perception.frame_quality import compute_frame_quality, select_frames
from ..perception.dynamic_object_filter import detect_dynamic_pixels
from ..reconstruction.depth_estimation import estimate_depth_for_sequence, check_depth_range_coverage
from ..reconstruction.frustum_check import check_frustum_shape, vfov_from_hfov
from ..reconstruction.prototype_point_repr import fuse_point_cloud, PrototypePointRepresentation
from ..reconstruction.confidence import compute_confidence, gate_by_confidence, ConfidenceResult, ConfidenceGatedDecision
from ..reconstruction.point_renderer import render_point_cloud
from ..evaluation.metrics_visual import evaluate_visual_fidelity, VisualFidelityResult
from ..evaluation.metrics_efficiency import EfficiencyTracker


@dataclasses.dataclass
class RealPipelineConfig:
    keep_fraction: float = 0.75
    min_overall: float = 0.15  # added for layer4_agents -- see PipelineConfig's matching field
    use_dynamic_filter: bool = True
    voxel_size: float = 2.0
    min_consistency: float = 0.3
    depth_window: int = 2
    # 2026-09 mesh quality audit (docs/MESH_QUALITY_AUDIT.md) changed n_depths/depth_spacing's
    # defaults -- see depth_estimation.py's module docstring for the full finding. depth_min/
    # depth_max are UNCHANGED here (still the pre-audit 3.0/100.0) because the right values are
    # scene-specific and were not re-derived against real footage in this environment -- run_real_pipeline
    # below now prints an automatic coverage diagnostic every run specifically so you can tell,
    # on YOUR footage, whether these two still need to change. Don't assume they're fine just
    # because the type/quantization fix is applied.
    n_depths: int = 128
    depth_spacing: str = "log"
    depth_min: float = 3.0
    depth_max: float = 100.0
    # Opt-in: discard pixels whose best depth sits at either end of the sweep (their true
    # depth is probably outside [depth_min, depth_max], making the reported value arbitrary).
    # Off by default because it also discards surfaces that legitimately sit near depth_max --
    # see plane_sweep_depth's docstring for the measured tradeoff. Turn this ON once the
    # boundary-fraction figure printed below tells you the range is genuinely too small.
    depth_boundary_reject: bool = False
    prune_below: float = 0.12
    densify_above: float = 0.60
    pose_confidence_scale_m: float = 3.0  # position_residual_m at which pose_confidence -> 0


@dataclasses.dataclass
class RealPipelineResult:
    points: PrototypePointRepresentation
    confidence: ConfidenceResult
    gate: ConfidenceGatedDecision
    kept_frame_indices: List[int]
    rejected_frame_indices: List[int]
    final_positions: np.ndarray
    final_colors: np.ndarray
    final_confidence: np.ndarray
    efficiency: dict
    depth_coverage: object = None  # DepthRangeCoverageReport, see depth_estimation.py (2026-09)
    frustum_check: object = None   # FrustumCheckResult, see frustum_check.py (2026-09)


def run_real_pipeline(dataset: RealDroneDataset, config: RealPipelineConfig = None) -> RealPipelineResult:
    cfg = config or RealPipelineConfig()
    tracker = EfficiencyTracker()
    n_total = len(dataset.frames)

    with tracker.track("frame_quality_scoring"):
        quality = [compute_frame_quality(f.rgb) for f in dataset.frames]
    kept, rejected = select_frames(quality, keep_fraction=cfg.keep_fraction, min_overall=cfg.min_overall)

    kept_grays = [cv2.cvtColor(dataset.frames[i].rgb, cv2.COLOR_RGB2GRAY) for i in kept]
    kept_poses = [dataset.frames[i].pose for i in kept]
    kept_rgb = [dataset.frames[i].rgb for i in kept]
    kept_quality = [quality[i].overall for i in kept]
    kept_pose_conf = [float(np.clip(1.0 - dataset.frames[i].position_residual_m / cfg.pose_confidence_scale_m, 0, 1))
                       for i in kept]
    # Each kept frame's OWN intrinsics when it has one (a zoom lens -- see
    # real_dataset_builder.py's per-frame K), else the shared dataset.K. 2026-09-21: this used
    # to be `dataset.K` inline at every call site below, which is what caused that day's
    # regression -- depth estimation kept using dataset.K for every frame even after frames
    # started carrying their own, so any pixel whose plane-sweep window crossed a focal-length
    # boundary got warped with the wrong intrinsics and its whole cost volume was garbage,
    # silently (still an ordinary-looking depth number, just a wrong one). One list, defined
    # once here, so every consumer in this function reads the same values.
    kept_Ks = [getattr(dataset.frames[i], "K", None) or dataset.K for i in kept]

    with tracker.track("depth_estimation_mvs"):
        depth_ests = estimate_depth_for_sequence(kept_grays, kept_poses, kept_Ks, window=cfg.depth_window,
                                                   n_depths=cfg.n_depths, depth_min=cfg.depth_min,
                                                   depth_max=cfg.depth_max, depth_spacing=cfg.depth_spacing,
                                                   boundary_reject=cfg.depth_boundary_reject)
    coverage = check_depth_range_coverage(depth_ests, cfg.depth_min, cfg.depth_max)
    if coverage.likely_range_too_small:
        print(f"WARNING [depth range coverage]: {coverage.warning}")
    bf = [e.boundary_fraction for e in depth_ests]
    if bf:
        mean_bf = sum(bf) / len(bf)
        print(f"[depth] pixels rejected for hitting a depth-sweep BOUNDARY (true depth outside "
              f"[{cfg.depth_min}, {cfg.depth_max}]): mean {mean_bf*100:.1f}% per frame "
              f"(max {max(bf)*100:.1f}%)")
        if mean_bf > 0.35:
            print(f"WARNING [depth range]: {mean_bf*100:.0f}% of pixels have no valid depth hypothesis "
                  f"in [{cfg.depth_min}, {cfg.depth_max}]. Most of this scene is OUTSIDE the tested "
                  f"range -- raise depth_max (and/or crop sky/horizon out of the frames). Proceeding "
                  f"would seed training from mostly-arbitrary depths; see docs/MESH_QUALITY_AUDIT.md.")

    dynamic_masks: Optional[List] = None
    with tracker.track("dynamic_object_filtering"):
        if cfg.use_dynamic_filter:
            dynamic_masks = []
            frac_flagged = []
            for li in range(len(kept)):
                nb = li + 2 if li + 2 < len(kept) else (li + 1 if li + 1 < len(kept) else None)
                if nb is None:
                    dynamic_masks.append(None)
                    continue
                res = detect_dynamic_pixels(kept_grays[li], kept_grays[nb], depth_ests[li].depth,
                                             kept_poses[li], kept_poses[nb], kept_Ks[li], K_next=kept_Ks[nb])
                dynamic_masks.append(res)
                frac_flagged.append(float(res.mask.mean()))
            # 2026-09: this stage previously printed nothing at all -- STOP_CONDITIONS.md item 3
            # explicitly flags the classical residual-flow filter's precision as "genuinely open,
            # untested" at real resolution, and there was no way to tell from a real run's log
            # whether it was doing anything. It runs (use_dynamic_filter defaults True) but was
            # silent either way.
            if frac_flagged:
                print(f"[dynamic object filter] {len(frac_flagged)} frame pairs checked; fraction of "
                      f"pixels flagged dynamic: mean={100*sum(frac_flagged)/len(frac_flagged):.2f}% "
                      f"max={100*max(frac_flagged):.2f}% (0 pairs skipped, no neighbor available: "
                      f"{sum(1 for m in dynamic_masks if m is None)})")
                if max(frac_flagged) < 0.001:
                    print("      NOTE: essentially nothing was flagged on any frame pair. If this scene "
                          "has visible moving traffic, that's a sign the classical residual-flow filter "
                          "isn't catching it at this resolution/threshold (see "
                          "layer1_cpu_sandbox/perception/dynamic_object_filter.py's residual_thresh_px / "
                          "min_area_px) rather than a sign there's nothing to catch -- spot-check a mask "
                          "against the actual frame before trusting either conclusion.")

    with tracker.track("point_fusion"):
        pts = fuse_point_cloud(depth_ests, dynamic_masks, kept_poses, kept_rgb, kept_quality, kept_pose_conf,
                                kept_Ks, voxel_size=cfg.voxel_size, min_consistency=cfg.min_consistency)

    with tracker.track("confidence_estimation"):
        conf = compute_confidence(pts)
    with tracker.track("confidence_gating"):
        gate = gate_by_confidence(conf, prune_below=cfg.prune_below, densify_above=cfg.densify_above)

    tracker.set_count("n_frames_total", n_total)
    tracker.set_count("n_frames_kept", len(kept))
    tracker.set_count("n_frames_rejected", len(rejected))
    tracker.set_count("n_points_raw_fused", int(pts.positions.shape[0]))
    tracker.set_count("n_points_after_gating", int(gate.keep_mask.sum()))

    # Shape sanity check on the SEED cloud, before any GPU training is spent on it. This is
    # the cheapest possible place to catch depth estimates that carry no real information --
    # see frustum_check.py's module docstring for the 2026-09 failure that motivated it.
    #
    # 2026-09-21: the expected-FOV comparison below was silently dead code before this fix --
    # `Intrinsics` has no `hfov_deg` attribute, so `hasattr(dataset.K, "hfov_deg")` was always
    # False and both `expected_*fov_deg` args were always None, meaning this check only ever
    # used the cone-shape (R^2/apex) signature, never the (stronger, decisive) "opening angle
    # matches a real camera's FOV" one -- see _export_mesh's post-training check for what that
    # signature catches when it's actually wired up. Deriving HFOV from fx/width (the inverse of
    # Intrinsics.from_fov) fixes that. With per-frame K now real, "the" camera FOV is anchored to
    # the FIRST kept frame specifically (kept_Ks[0]) -- a reasoned choice, not the only one, since
    # a scene with real zoom has no single answer.
    frustum = None
    if pts.positions.shape[0] >= 200:
        _ref_K = kept_Ks[0] if kept_Ks else dataset.K
        _ref_hfov_deg = float(np.degrees(2 * np.arctan(_ref_K.width / (2 * _ref_K.fx))))
        _vfov = vfov_from_hfov(_ref_hfov_deg, _ref_K.width, _ref_K.height)
        frustum = check_frustum_shape(pts.positions, expected_vfov_deg=_vfov, expected_hfov_deg=_ref_hfov_deg)
        print(f"[geometry sanity] {frustum.format_report()}")
        if frustum.is_frustum_shaped:
            print("WARNING [geometry sanity]: the fused seed cloud is shaped like the CAMERA FRUSTUM, "
                  "not like a scene -- the depth estimates are probably not carrying real information. "
                  "Training on this will still produce good-looking renders but geometrically "
                  "meaningless 3D. Fix depth estimation before spending GPU time; see "
                  "docs/MESH_QUALITY_AUDIT.md.")

    return RealPipelineResult(points=pts, confidence=conf, gate=gate, kept_frame_indices=kept,
                               rejected_frame_indices=rejected, final_positions=pts.positions[gate.keep_mask],
                               final_colors=pts.colors[gate.keep_mask], final_confidence=conf.confidence[gate.keep_mask],
                               efficiency=tracker.report(), depth_coverage=coverage,
                               frustum_check=frustum)


@dataclasses.dataclass
class RealEvaluation:
    self_consistency_views: List[dict]   # [{"frame_idx": i, "visual": VisualFidelityResult, "coverage_frac": f}, ...]
    confidence_mean: float
    confidence_band_counts: dict
    n_final_points: int
    NOTE: str = ("No ground truth exists for real footage. `self_consistency_views` compares the "
                 "reconstruction, RENDERED FROM A POSE THAT WAS ACTUALLY CAPTURED, against the REAL "
                 "frame the camera captured at that pose -- a genuine check ('does the reconstruction "
                 "reproduce what the camera saw'), but NOT a geometric/metric accuracy claim, and NOT "
                 "a novel-view test (there is no known-correct image for an unobserved pose here).")


def evaluate_real_result(dataset: RealDroneDataset, result: RealPipelineResult,
                          n_self_consistency_views: int = 3) -> RealEvaluation:
    kept = result.kept_frame_indices
    check_indices = np.linspace(0, len(kept) - 1, min(n_self_consistency_views, len(kept)), dtype=int)
    views = []
    for li in check_indices:
        frame_idx = kept[int(li)]
        frame = dataset.frames[frame_idx]
        pose = frame.pose
        actual_rgb = frame.rgb
        # That frame's own K when it has one (a zoom lens -- see real_dataset_builder.py), else
        # dataset.K -- rendering this comparison with the WRONG intrinsics for this specific frame
        # would misalign it against actual_rgb for a reason that has nothing to do with
        # reconstruction quality. Same class of bug as fuse_point_cloud's, fixed 2026-09-21.
        frame_K = getattr(frame, "K", None) or dataset.K
        rendered, mask = render_point_cloud(result.final_positions, result.final_colors, frame_K, pose)
        visual = evaluate_visual_fidelity(rendered, actual_rgb, mask)
        views.append({"frame_idx": frame_idx, "visual": visual, "coverage_frac": float(mask.mean())})

    band_counts = {b: int((result.confidence.band == b).sum()) for b in set(result.confidence.band)}
    return RealEvaluation(self_consistency_views=views, confidence_mean=float(result.confidence.confidence.mean()),
                           confidence_band_counts=band_counts, n_final_points=int(result.final_positions.shape[0]))
