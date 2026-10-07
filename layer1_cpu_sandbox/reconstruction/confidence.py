"""
confidence.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested
CORE NOVELTY MODULE -- "Observation-Aware Reconstruction Confidence"

For every reconstructed primitive (here: a fused point; in Layer 3, a
Gaussian) this module computes a single confidence score in [0,1] from
five independently-measurable evidence signals, all of which are already
produced by earlier CPU-sandbox modules with NO access to ground truth:

  1. obs_score      -- how many DISTINCT frames independently produced a
                        point landing in this same location (multi-view
                        redundancy: the strongest evidence there is).
  2. angle_score     -- angular spread between the extreme observing
                        viewpoints (a wide baseline triangulates far more
                        reliably than a narrow one; near-zero spread means
                        "confirmed" only by frames that were barely
                        different viewpoints).
  3. consistency_score -- mean plane-sweep cost-curve sharpness of the
                        contributing pixels (a flat cost curve = ambiguous
                        match even if it nominally "won").
  4. quality_score   -- mean classical frame-quality score of the
                        contributing frames (blur/exposure/compression).
  5. pose_score      -- mean camera-pose confidence (inverse GPS/IMU-EKF
                        position uncertainty) of the contributing frames.

Combination: a WEIGHTED GEOMETRIC MEAN, not a weighted arithmetic mean.
This is a deliberate choice, not a default: an arithmetic mean lets one
excellent signal compensate for another catastrophically bad one (e.g. a
"perfect" cost-curve match from a single, barely-moved viewpoint pair
still averaging out to "moderate" confidence); a geometric mean forces
every factor to be simultaneously reasonable, which better matches how
photogrammetry practitioners actually reason about triangulation
reliability ("no amount of sharpness fixes a near-zero baseline").

SINGLE-OBSERVATION CAP (explicit, and important for scientific honesty):
a point confirmed by only one frame -- i.e. never independently
re-observed by a second, sufficiently different viewpoint -- is hard-capped
at the "Moderate" band, however good its individual signals look. This
directly implements the report's requirement that the system must be able
to say "insufficient observation" rather than silently presenting an
unconfirmed single-view guess as reliable geometry.

Default weights below are a reasoned starting point for the CPU sandbox,
not a claimed optimum -- layer2_interfaces/i_confidence_estimator.py notes
learning these weights (e.g. via logistic regression against real
GT/RTK-checkpoint error) as a concrete Layer-3 research extension.
"""
from __future__ import annotations

import dataclasses
import numpy as np

from .prototype_point_repr import PrototypePointRepresentation

DEFAULT_WEIGHTS = dict(obs=0.30, angle=0.20, consistency=0.25, quality=0.15, pose=0.10)

BAND_HIGH = "High"
BAND_MODERATE = "Moderate"
BAND_LOW = "Low / insufficient observation"


@dataclasses.dataclass
class ConfidenceResult:
    confidence: np.ndarray         # (M,) in [0,1]
    band: np.ndarray               # (M,) dtype=object, one of BAND_HIGH/MODERATE/LOW
    obs_score: np.ndarray
    angle_score: np.ndarray
    consistency_score: np.ndarray
    quality_score: np.ndarray
    pose_score: np.ndarray
    single_view_capped: np.ndarray  # (M,) bool -- True where the hard single-observation cap applied
    weights: dict


def compute_confidence(pts: PrototypePointRepresentation, weights: dict = None,
                        obs_tau: float = 1.5, target_angle_spread_rad: float = np.deg2rad(6.0),
                        high_thresh: float = 0.60, moderate_thresh: float = 0.35,
                        single_view_cap: float = 0.55) -> ConfidenceResult:
    w = weights or DEFAULT_WEIGHTS
    eps = 1e-6

    obs_score = 1.0 - np.exp(-(pts.observation_count - 1) / obs_tau)
    angle_score = np.clip(pts.view_angle_spread / target_angle_spread_rad, 0.0, 1.0)
    consistency_score = np.clip(pts.mean_consistency, 0.0, 1.0)
    quality_score = np.clip(pts.mean_frame_quality, 0.0, 1.0)
    pose_score = np.clip(pts.mean_pose_confidence, 0.0, 1.0)

    log_conf = (w["obs"] * np.log(obs_score + eps) + w["angle"] * np.log(angle_score + eps) +
                w["consistency"] * np.log(consistency_score + eps) + w["quality"] * np.log(quality_score + eps) +
                w["pose"] * np.log(pose_score + eps))
    confidence = np.exp(log_conf)

    single_view = pts.observation_count <= 1
    capped = single_view & (confidence > single_view_cap)
    confidence = np.where(capped, np.minimum(confidence, single_view_cap), confidence)

    band = np.where(confidence >= high_thresh, BAND_HIGH,
                     np.where(confidence >= moderate_thresh, BAND_MODERATE, BAND_LOW))

    return ConfidenceResult(confidence=confidence.astype(np.float32), band=band,
                             obs_score=obs_score.astype(np.float32), angle_score=angle_score.astype(np.float32),
                             consistency_score=consistency_score.astype(np.float32),
                             quality_score=quality_score.astype(np.float32),
                             pose_score=pose_score.astype(np.float32),
                             single_view_capped=capped, weights=w)


@dataclasses.dataclass
class ConfidenceGatedDecision:
    keep_mask: np.ndarray       # (M,) bool -- False = pruned outright (report as a gap, don't render)
    densify_mask: np.ndarray    # (M,) bool -- True = candidate for extra sampling / Gaussian splitting
    flag_uncertain_mask: np.ndarray  # (M,) bool -- kept but must be visually flagged low-confidence


def gate_by_confidence(conf: ConfidenceResult, prune_below: float = 0.12,
                        densify_above: float = 0.60) -> ConfidenceGatedDecision:
    """
    THE central control loop this module exists to justify: confidence does not
    just get reported after the fact, it actively controls the representation.
      * prune_below: points this unreliable are DROPPED, not rendered as if real
        (this is what keeps the system from "pretending unseen geometry is
        accurate" -- see occlusion.py for the complementary coverage-gap report).
      * densify_above: points this reliable are marked as safe anchors for
        additional sampling/refinement budget (in Layer 3: safe anchors for
        Gaussian splitting/densification).
      * everything in between is kept but flagged uncertain in the visualization.
    """
    keep = conf.confidence >= prune_below
    densify = conf.confidence >= densify_above
    flag_uncertain = keep & (conf.band != BAND_HIGH)
    return ConfidenceGatedDecision(keep_mask=keep, densify_mask=densify, flag_uncertain_mask=flag_uncertain)
