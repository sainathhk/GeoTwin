"""
context_builders.py
LAYER 4 (AGENTS) -- STATUS: implemented, CPU-tested

Pure functions: existing Layer-1 dataclasses IN, small JSON-safe dict OUT.
No decisions, no LLM calls, nothing stochastic -- these are as deterministic
and unit-testable as `report_generator.py`'s own summary tables, because
that's exactly the role they play for the agent layer.

Two things every builder here guarantees:
  1. NaN/inf never reach the output. `evaluate_result.py` can legitimately
     produce NaN (e.g. a novel-view PSNR when too few points survived
     gating to render anything) and that is a REAL, meaningful result, not
     a bug -- but `json.dumps(nan)` produces invalid JSON, and an agent
     reading a literal NaN token in a prompt has no idea whether that's
     "no value" or a formatting error. Every numeric goes through
     `_safe(x)`, which turns non-finite values into `None` and leaves
     everything else untouched.
  2. Only scalars, small dicts/lists of scalars, and short strings ever go
     in. No positions arrays, no images, no raw point clouds -- an agent
     reading this context physically cannot see geometry, only numbers
     ABOUT geometry that other, deterministic code already computed.
"""
from __future__ import annotations

import math
from typing import List, Optional, Dict, Any

import numpy as np

from layer1_cpu_sandbox.perception.frame_overlap import OverlapSummary
from layer1_cpu_sandbox.evaluation.spatial_diagnostics import SpatialDiagnostic, summarize_weak_regions

RECON_TUNABLE_FIELDS = (
    "keep_fraction", "min_overall", "voxel_size", "min_consistency",
    "depth_window", "n_depths", "depth_min", "depth_max", "prune_below", "densify_above",
)


def _safe(x) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _round_dict(d: dict, ndigits: int = 4) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = None if v is None or not math.isfinite(v) else round(v, ndigits)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Frame Agent context
# ---------------------------------------------------------------------------

def build_frame_agent_context(quality_scores: list, raw_overlap: OverlapSummary,
                                current_keep_fraction: float = 0.75,
                                current_min_overall: float = 0.15) -> Dict[str, Any]:
    """`quality_scores`: List[FrameQualityScore] for the FULL raw sequence (pre-selection).
    `raw_overlap`: from frame_overlap.compute_consecutive_overlap on the same raw sequence.
    """
    overall = np.array([s.overall for s in quality_scores], dtype=float)
    blur = np.array([s.blur_score for s in quality_scores], dtype=float)
    exposure = np.array([s.exposure_score for s in quality_scores], dtype=float)
    coverage = np.array([s.coverage_score for s in quality_scores], dtype=float)

    return {
        "n_frames": len(quality_scores),
        "current_keep_fraction": current_keep_fraction,
        "current_min_overall": current_min_overall,
        "quality": _round_dict({
            "mean_overall": float(overall.mean()) if len(overall) else None,
            "min_overall_score": float(overall.min()) if len(overall) else None,
            "p10_overall": float(np.percentile(overall, 10)) if len(overall) else None,
            "mean_blur": float(blur.mean()) if len(blur) else None,
            "mean_exposure": float(exposure.mean()) if len(exposure) else None,
            "mean_coverage": float(coverage.mean()) if len(coverage) else None,
            "fraction_below_0.15": float((overall < 0.15).mean()) if len(overall) else None,
        }),
        "raw_capture_overlap": {
            "n_pairs": len(raw_overlap.pairs),
            "n_ok": raw_overlap.n_ok,
            "n_duplicate_risk": raw_overlap.n_duplicate_risk,
            "n_gap_risk": raw_overlap.n_gap_risk,
            "n_indeterminate": raw_overlap.n_indeterminate,
            "mean_match_ratio": _safe(raw_overlap.mean_match_ratio),
        },
    }


# ---------------------------------------------------------------------------
# Reconstruction Agent context
# ---------------------------------------------------------------------------

def config_tunable_subset(config) -> Dict[str, Any]:
    return {f: getattr(config, f) for f in RECON_TUNABLE_FIELDS if hasattr(config, f)}


def build_reconstruction_agent_context(result, config, spatial_diag: SpatialDiagnostic,
                                         attempt_history: Optional[List[dict]] = None,
                                         n_total_frames: Optional[int] = None) -> Dict[str, Any]:
    """`result`: a PipelineResult or RealPipelineResult (both expose the same field names
    used below). `config`: the PipelineConfig/RealPipelineConfig THIS result was produced
    with (its tunable subset is echoed back so the agent knows what it's adjusting from).
    `attempt_history`: prior iterations this orchestrator run has already tried, oldest
    first -- lets the agent avoid repeating something that already didn't help.
    """
    kept_mask = result.gate.keep_mask
    conf_kept = result.confidence.confidence[kept_mask]
    band_kept = result.confidence.band[kept_mask]
    band_counts = {b: int((band_kept == b).sum()) for b in
                   ["High", "Moderate", "Low / insufficient observation"]}

    weak_regions = summarize_weak_regions(spatial_diag, n=3)

    n_kept_frames = len(result.kept_frame_indices)
    n_rejected_frames = len(result.rejected_frame_indices)

    return {
        "current_config": _round_dict(config_tunable_subset(config)),
        "frames": {
            "n_total": n_total_frames if n_total_frames is not None else n_kept_frames + n_rejected_frames,
            "n_kept": n_kept_frames,
            "n_rejected": n_rejected_frames,
        },
        "points": {
            "n_raw_fused": int(result.points.positions.shape[0]),
            "n_after_gating": int(result.final_positions.shape[0]),
            "mean_confidence_kept": _safe(conf_kept.mean()) if conf_kept.size else None,
            "band_counts_kept": band_counts,
            "low_confidence_fraction_kept": (
                _safe(band_counts["Low / insufficient observation"] / max(1, kept_mask.sum()))
            ),
        },
        "spatial_diagnostic": {
            "n_occupied_cells": len(spatial_diag.cells),
            "n_empty_cells_in_bbox": spatial_diag.n_empty_cells,
            "weak_regions": weak_regions,  # already compact dicts; [] if nothing meets the bar
        },
        "efficiency_seconds": _round_dict({k: _safe(v) for k, v in (result.efficiency or {}).items()}),
        "attempt_history": attempt_history or [],
    }


# ---------------------------------------------------------------------------
# Evaluation Agent context
# ---------------------------------------------------------------------------

def build_evaluation_agent_context_synthetic(evaluation) -> Dict[str, Any]:
    """`evaluation`: a `FullEvaluation` (has ground truth -- synthetic dataset only)."""
    ev = evaluation
    key_numbers = _round_dict({
        "coverage_observed": _safe(ev.coverage_observed),
        "coverage_observable_but_dropped": _safe(ev.coverage_observable_but_dropped),
        "coverage_never_observable": _safe(ev.coverage_never_observable),
        "novel_view_psnr": _safe(ev.visual_novel_view.psnr),
        "novel_view_ssim": _safe(ev.visual_novel_view.ssim),
        "novel_view_valid_pixel_fraction": _safe(ev.visual_novel_view.valid_pixel_fraction),
        "train_view_psnr": _safe(ev.visual_train_view.psnr),
        "chamfer_distance_m": _safe(ev.geometry.chamfer_distance),
        "point_to_point_rmse_m": _safe(ev.geometry.point_to_point_rmse),
        "mean_abs_height_error_m": _safe(ev.metric_accuracy.mean_abs_height_error_m),
        "control_point_rmse_m": _safe(ev.metric_accuracy.control_point_rmse_m),
        "n_control_points_matched": ev.metric_accuracy.n_control_points_matched,
        "surface_completeness": _safe(ev.completeness.surface_completeness),
        "surface_completeness_high_conf": _safe(ev.completeness.surface_completeness_high_conf),
        "fraction_of_theoretical_ceiling": _safe(ev.completeness.fraction_of_theoretical_ceiling),
        "total_seconds": _safe((ev.efficiency or {}).get("total_seconds")),
    })
    return {"has_ground_truth": True, "key_numbers": key_numbers}


def build_evaluation_agent_context_real(evaluation) -> Dict[str, Any]:
    """`evaluation`: a `RealEvaluation` (NO ground truth -- self-consistency only)."""
    ev = evaluation
    views = ev.self_consistency_views or []
    psnrs = [_safe(v["visual"].psnr) for v in views]
    psnrs = [p for p in psnrs if p is not None]
    ssims = [_safe(v["visual"].ssim) for v in views]
    ssims = [s for s in ssims if s is not None]
    key_numbers = _round_dict({
        "mean_self_consistency_psnr": float(np.mean(psnrs)) if psnrs else None,
        "mean_self_consistency_ssim": float(np.mean(ssims)) if ssims else None,
        "n_self_consistency_views": len(views),
        "confidence_mean": _safe(ev.confidence_mean),
        "n_final_points": ev.n_final_points,
    })
    key_numbers["confidence_band_counts"] = dict(ev.confidence_band_counts or {})
    return {"has_ground_truth": False, "key_numbers": key_numbers,
            "note": "no ground truth for real footage -- these are self-consistency and "
                     "confidence-distribution numbers only, not geometric accuracy"}


# ---------------------------------------------------------------------------
# Training Control Agent context (Layer 3 GPU training loop, NOT Layer 1)
# ---------------------------------------------------------------------------

def build_training_agent_context(held_out_history: List[dict], best_held_out: dict,
                                    current_iter: int, max_iterations: int, checkpoint_every: int,
                                    densifying_active: bool, n_gaussians: int, max_gaussians: int,
                                    history_window: int = 10) -> Dict[str, Any]:
    """`held_out_history`: list of {"iter","psnr","ssim","lpips","n_gaussians"} dicts, one per
    held-out checkpoint eval, oldest first (train_gpu.py's `train()` builds this -- see
    docs/AGENTIC_ARCHITECTURE.md's "Training Control Agent" section). `best_held_out`:
    train_gpu.py's existing {"psnr": {"value","iter"}, "ssim": {...}, "lpips": {...}} dict --
    passed through as-is, not recomputed here. Only the last `history_window` checkpoints are
    included (an LLM doesn't need all 40 checkpoints of a 20000-iteration run to see a trend,
    and this keeps prompts small the same way spatial_diagnostic's top-N weak regions does)."""
    recent = held_out_history[-history_window:]
    return {
        "current_iter": current_iter,
        "max_iterations": max_iterations,
        "checkpoint_every": checkpoint_every,
        "densifying_active": densifying_active,
        "n_gaussians": n_gaussians,
        "max_gaussians": max_gaussians,
        "held_out_history": [
            _round_dict({"iter": h["iter"], "psnr": _safe(h.get("psnr")), "ssim": _safe(h.get("ssim")),
                          "lpips": _safe(h.get("lpips")), "n_gaussians": h.get("n_gaussians")})
            for h in recent
        ],
        "best_held_out": {
            metric: _round_dict({"value": _safe(best_held_out.get(metric, {}).get("value")),
                                   "iter": best_held_out.get(metric, {}).get("iter")})
            for metric in ("psnr", "ssim", "lpips")
        },
    }
