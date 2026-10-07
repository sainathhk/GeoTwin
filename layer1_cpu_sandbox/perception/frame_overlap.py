"""
frame_overlap.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested
LAYER 2 INTERFACE: (extends i_frame_quality.py's contract -- see module note)

Answers a question frame_quality.py deliberately does NOT: not "is this frame
sharp/exposed/clean" but "how much does this frame actually share with its
neighbor". Classical, explainable, ORB-feature-match based -- same discipline
as frame_quality.py's blur/exposure/compression scores.

WHY THIS MODULE EXISTS (a specific, code-level gap, not a generic worry):
`reconstruction/depth_estimation.py::estimate_depth_for_sequence` picks MVS
neighbors by *position in the KEPT list* (`window` on either side of the
kept-list index), not by measured visual overlap. `pipeline.py` builds that
kept list by calling `select_frames()`, which ranks frames by quality score
ALONE and can drop a consecutive run of frames with no regard for what that
does to the survivors' effective baseline. Concretely: if frames 5-7 are
dropped for being blurry, frame 4 and frame 8 become "adjacent" in the kept
list and plane-sweep MVS will treat them as a `window=1` neighbor pair --
but 4 and 8 may share far less overlap than two genuinely consecutive raw
frames would. `depth_estimation.py`'s own consistency signal can catch some
of this (a flat cost curve if the match fails) but only AFTER the expensive
plane-sweep has run, and with no explanation of WHY. This module makes the
overlap risk visible BEFORE that -- cheaply, on raw frames -- so a decision
layer can act on it directly.

This module does not change frame selection itself: `select_frames()` in
frame_quality.py is untouched (its numbers stay reproducible). It only
produces a new, independent signal that a decision layer (layer4_agents/)
can read alongside frame_quality's scores.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np
import cv2

STATUS_DUPLICATE_RISK = "duplicate_risk"   # near-static pair: almost every feature matches with ~0 displacement
STATUS_OK = "ok"                            # healthy overlap: enough matches, real but moderate displacement
STATUS_GAP_RISK = "gap_risk"                # too few confident matches: baseline may be too wide / scene changed
STATUS_INDETERMINATE = "indeterminate"      # not enough texture in one or both frames to say anything (e.g. sky/water)

# Tuned against 160-320px-wide synthetic/mock frames (this repo's Environment A scale).
# Re-check these against real 1080p/4K footage before trusting them there -- see
# docs/AGENTIC_ARCHITECTURE.md "what must be re-validated on real footage".
DEFAULT_MIN_KEYPOINTS = 12          # below this in EITHER frame, the pair is indeterminate, not "gap_risk"
DEFAULT_DUPLICATE_MATCH_RATIO = 0.85
DEFAULT_DUPLICATE_MAX_DISP_PX = 1.5
DEFAULT_GAP_MATCH_RATIO = 0.12


@dataclasses.dataclass
class FramePairOverlap:
    i: int                        # index of the first frame IN WHATEVER LIST WAS PASSED IN (caller maps back)
    j: int                        # index of the second frame
    n_keypoints_i: int
    n_keypoints_j: int
    n_matches: int
    match_ratio: float            # n_matches / max(1, min(n_keypoints_i, n_keypoints_j)), in [0,1]
    median_disp_px: float         # median matched-keypoint displacement in pixels; NaN if n_matches == 0
    status: str                   # one of the STATUS_* constants above


@dataclasses.dataclass
class OverlapSummary:
    pairs: List[FramePairOverlap]
    n_duplicate_risk: int
    n_gap_risk: int
    n_indeterminate: int
    n_ok: int
    mean_match_ratio: float       # over non-indeterminate pairs only; NaN if all indeterminate
    worst_gap_pairs: List[FramePairOverlap]   # up to 3, lowest match_ratio first, excludes indeterminate


_orb = None  # lazily constructed -- cv2.ORB_create() has a small fixed cost, share one instance


def _get_orb():
    global _orb
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=500)
    return _orb


def pairwise_overlap(gray_a: np.ndarray, gray_b: np.ndarray, i: int = 0, j: int = 1,
                      min_keypoints: int = DEFAULT_MIN_KEYPOINTS,
                      duplicate_match_ratio: float = DEFAULT_DUPLICATE_MATCH_RATIO,
                      duplicate_max_disp_px: float = DEFAULT_DUPLICATE_MAX_DISP_PX,
                      gap_match_ratio: float = DEFAULT_GAP_MATCH_RATIO) -> FramePairOverlap:
    """ORB-feature overlap between two grayscale frames. Classification is deliberately
    conservative: a pair only gets a definite status if there was enough texture in BOTH
    frames to measure it (see STATUS_INDETERMINATE) -- a low-texture pair (sky, water,
    a blank wall) is not evidence of a coverage gap, it's just unmeasurable, and treating
    it as `gap_risk` would be a false alarm a downstream agent could act on wrongly.
    """
    orb = _get_orb()
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    n_kp_a, n_kp_b = len(kp_a), len(kp_b)

    if n_kp_a < min_keypoints or n_kp_b < min_keypoints or des_a is None or des_b is None:
        return FramePairOverlap(i=i, j=j, n_keypoints_i=n_kp_a, n_keypoints_j=n_kp_b,
                                 n_matches=0, match_ratio=0.0, median_disp_px=float("nan"),
                                 status=STATUS_INDETERMINATE)

    # Lowe's ratio test (knnMatch, k=2), NOT plain crossCheck: crossCheck only enforces
    # mutual-nearest-neighbour agreement, which still accepts a match when BOTH sides'
    # descriptor sets are effectively random (e.g. two unrelated noise/low-structure
    # frames) -- every descriptor has SOME nearest neighbour, mutual or not, so crossCheck
    # alone measured ~0.46 match_ratio between two independent pure-noise frames in testing.
    # The ratio test instead asks "is the best match meaningfully better than the second
    # best", which is what actually distinguishes a genuine correspondence from a
    # coincidental nearest neighbour among noise.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des_a, des_b, k=2)
    # Defensive: knnMatch can return a pair shorter than k=2 when the "other" descriptor
    # set is very small; only apply the ratio test where a genuine second-best exists.
    matches = [pair[0] for pair in raw_matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    n_matches = len(matches)
    match_ratio = float(n_matches / max(1, min(n_kp_a, n_kp_b)))

    if n_matches == 0:
        median_disp = float("nan")
    else:
        pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], dtype=np.float32)
        disp = np.linalg.norm(pts_a - pts_b, axis=1)
        median_disp = float(np.median(disp))

    if match_ratio >= duplicate_match_ratio and n_matches > 0 and median_disp <= duplicate_max_disp_px:
        status = STATUS_DUPLICATE_RISK
    elif match_ratio < gap_match_ratio:
        status = STATUS_GAP_RISK
    else:
        status = STATUS_OK

    return FramePairOverlap(i=i, j=j, n_keypoints_i=n_kp_a, n_keypoints_j=n_kp_b,
                             n_matches=n_matches, match_ratio=match_ratio,
                             median_disp_px=median_disp, status=status)


def _summarize(pairs: List[FramePairOverlap]) -> OverlapSummary:
    n_dup = sum(1 for p in pairs if p.status == STATUS_DUPLICATE_RISK)
    n_gap = sum(1 for p in pairs if p.status == STATUS_GAP_RISK)
    n_indet = sum(1 for p in pairs if p.status == STATUS_INDETERMINATE)
    n_ok = sum(1 for p in pairs if p.status == STATUS_OK)
    measurable = [p for p in pairs if p.status != STATUS_INDETERMINATE]
    mean_ratio = float(np.mean([p.match_ratio for p in measurable])) if measurable else float("nan")
    gap_sorted = sorted([p for p in pairs if p.status == STATUS_GAP_RISK], key=lambda p: p.match_ratio)
    return OverlapSummary(pairs=pairs, n_duplicate_risk=n_dup, n_gap_risk=n_gap,
                           n_indeterminate=n_indet, n_ok=n_ok, mean_match_ratio=mean_ratio,
                           worst_gap_pairs=gap_sorted[:3])


def compute_consecutive_overlap(gray_frames: List[np.ndarray], **kwargs) -> OverlapSummary:
    """Raw-capture diagnostic: overlap between every (i, i+1) pair in the FULL sequence,
    before any quality filtering. Answers "what did the flight/encode actually look like"
    -- e.g. a hover shows up as a run of duplicate_risk pairs, a dropped/corrupted frame
    or a fast turn shows up as gap_risk.
    """
    pairs = [pairwise_overlap(gray_frames[k], gray_frames[k + 1], i=k, j=k + 1, **kwargs)
              for k in range(len(gray_frames) - 1)]
    return _summarize(pairs)


def compute_kept_neighbor_overlap(gray_frames: List[np.ndarray], kept_indices: List[int],
                                    **kwargs) -> OverlapSummary:
    """Post-selection diagnostic: DIRECT overlap between every pair of consecutive entries
    in `kept_indices` (indices into `gray_frames`) -- i.e. exactly the pairs
    `depth_estimation.estimate_depth_for_sequence` will treat as `window`-neighbors once
    `pipeline.py` re-indexes the kept list. This is what actually validates the "adjacent
    in the kept list means adjacent in the world" assumption for a GIVEN selection, whatever
    that selection turned out to be.
    """
    pairs = [pairwise_overlap(gray_frames[kept_indices[k]], gray_frames[kept_indices[k + 1]],
                                i=kept_indices[k], j=kept_indices[k + 1], **kwargs)
              for k in range(len(kept_indices) - 1)]
    return _summarize(pairs)
