"""
frame_quality.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested
LAYER 2 INTERFACE: layer2_interfaces/i_frame_quality.py

Answers "which frames contain useful information / which are unreliable"
(module 3 of the required pipeline) using classical, fully-explainable
signals: blur (Laplacian variance), exposure (histogram clipping),
compression artifacting (JPEG blockiness via 8x8 DCT-grid gradient), and
coverage (fraction of frame that is actual scene vs. sky/empty).

This is deliberately classical -- see layer2_interfaces/i_frame_quality.py
for the learned-CNN replacement this is designed to be swapped for on GPU.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import cv2


@dataclasses.dataclass
class FrameQualityScore:
    blur_score: float          # 0 (very blurred) .. 1 (sharp)
    exposure_score: float      # 0 (over/under-exposed) .. 1 (well exposed)
    compression_score: float   # 0 (heavy blocking artifacts) .. 1 (clean)
    coverage_score: float      # 0 (mostly empty) .. 1 (fully covered by scene content)
    overall: float             # weighted combination, used for frame selection


def _blur_score(gray: np.ndarray) -> float:
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Empirically, "acceptably sharp" small aerial frames sit well above ~40-60
    # Laplacian variance; saturate the score smoothly rather than with a hard cutoff.
    return float(np.clip(lap_var / 150.0, 0.0, 1.0))


def _exposure_score(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist /= max(hist.sum(), 1.0)
    clipped_low = hist[:5].sum()
    clipped_high = hist[-5:].sum()
    clipping_penalty = clipped_low + clipped_high
    # Reward histograms that actually spread across the range (contrast).
    nonzero_bins = np.count_nonzero(hist > 1e-4)
    spread = nonzero_bins / 256.0
    return float(np.clip(1.0 - 2.0 * clipping_penalty, 0, 1) * 0.6 + spread * 0.4)


def _compression_score(gray: np.ndarray) -> float:
    """Cheap JPEG-blockiness estimate: energy of the gradient exactly on 8x8 block boundaries."""
    h, w = gray.shape
    if h < 16 or w < 16:
        return 1.0
    gx = np.abs(np.diff(gray.astype(np.float32), axis=1))
    block_cols = np.arange(7, w - 1, 8)
    if block_cols.size == 0:
        return 1.0
    boundary_energy = gx[:, block_cols].mean()
    overall_energy = gx.mean() + 1e-6
    ratio = boundary_energy / overall_energy
    # ratio ~1 => no extra energy at block boundaries (clean); higher => blocky.
    return float(np.clip(2.0 - ratio, 0.0, 1.0))


def _coverage_score(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    non_black = (gray > 3).mean()
    return float(non_black)


def compute_frame_quality(rgb: np.ndarray,
                           weights=(0.40, 0.20, 0.20, 0.20)) -> FrameQualityScore:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    b = _blur_score(gray)
    e = _exposure_score(gray)
    c = _compression_score(gray)
    cov = _coverage_score(rgb)
    wb, we, wc, wcov = weights
    overall = wb * b + we * e + wc * c + wcov * cov
    return FrameQualityScore(blur_score=b, exposure_score=e, compression_score=c,
                              coverage_score=cov, overall=float(np.clip(overall, 0, 1)))


def select_frames(scores: list, keep_fraction: float = 0.7, min_overall: float = 0.15):
    """
    Frame-selection policy: drop frames that fail an absolute quality floor,
    then keep the best `keep_fraction` of what remains. Returns
    (kept_indices, rejected_indices) both sorted ascending.
    """
    n = len(scores)
    idx_scores = [(i, s.overall) for i, s in enumerate(scores)]
    survivors = [i for i, s in idx_scores if s >= min_overall]
    survivors_sorted = sorted(survivors, key=lambda i: idx_scores[i][1], reverse=True)
    n_keep = max(3, int(round(len(survivors_sorted) * keep_fraction)))
    kept = sorted(survivors_sorted[:n_keep])
    rejected = sorted(set(range(n)) - set(kept))
    return kept, rejected
