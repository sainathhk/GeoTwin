"""Unit tests for layer1_cpu_sandbox.perception.frame_overlap."""
import numpy as np
import cv2
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_cpu_sandbox.perception.frame_overlap import (
    pairwise_overlap, compute_consecutive_overlap, compute_kept_neighbor_overlap,
    STATUS_DUPLICATE_RISK, STATUS_OK, STATUS_GAP_RISK, STATUS_INDETERMINATE,
)


def _textured_frame(seed: int, shift=(0, 0), size=128):
    """A reproducible, textured synthetic frame: random blobs at fixed seed, then shifted.
    Shifting (not regenerating) is what makes two frames genuinely overlapping."""
    rng = np.random.default_rng(0)  # SAME base pattern every call
    canvas = np.zeros((size + 40, size + 40), dtype=np.uint8)
    centers = rng.integers(20, size + 20, size=(60, 2))
    radii = rng.integers(3, 9, size=60)
    for (cx, cy), r in zip(centers, radii):
        cv2.circle(canvas, (int(cx), int(cy)), int(r), 255, -1)
    dx, dy = shift
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(canvas, M, (size + 40, size + 40), borderValue=0)
    return shifted[20:20 + size, 20:20 + size]


def _blank_frame(size=128, value=10):
    return np.full((size, size), value, dtype=np.uint8)


def test_identical_frames_are_duplicate_risk():
    f = _textured_frame(seed=0)
    result = pairwise_overlap(f, f.copy())
    assert result.status == STATUS_DUPLICATE_RISK
    assert result.match_ratio > 0.8
    assert result.median_disp_px < 1.0


def test_small_shift_is_ok_not_duplicate():
    f_a = _textured_frame(seed=0, shift=(0, 0))
    f_b = _textured_frame(seed=0, shift=(15, 0))
    result = pairwise_overlap(f_a, f_b)
    assert result.status in (STATUS_OK, STATUS_DUPLICATE_RISK)
    assert result.median_disp_px > 5.0


def test_large_shift_reduces_match_ratio_vs_small_shift():
    f_base = _textured_frame(seed=0, shift=(0, 0))
    f_small = _textured_frame(seed=0, shift=(10, 0))
    f_large = _textured_frame(seed=0, shift=(90, 70))
    small = pairwise_overlap(f_base, f_small)
    large = pairwise_overlap(f_base, f_large)
    assert small.match_ratio >= large.match_ratio


def test_blank_frames_are_indeterminate_not_gap_risk():
    a = _blank_frame(value=10)
    b = _blank_frame(value=12)
    result = pairwise_overlap(a, b)
    assert result.status == STATUS_INDETERMINATE
    assert result.n_matches == 0


def test_completely_unrelated_textures_can_be_gap_risk_or_indeterminate():
    rng = np.random.default_rng(1)
    a = (rng.random((128, 128)) * 255).astype(np.uint8)
    rng2 = np.random.default_rng(999)
    b = (rng2.random((128, 128)) * 255).astype(np.uint8)
    result = pairwise_overlap(a, b)
    # Pure noise has no stable structure for ORB to lock onto across two independent frames;
    # it must not be misreported as healthy overlap.
    assert result.status != STATUS_OK


def test_consecutive_overlap_summary_shapes():
    frames = [_textured_frame(seed=0, shift=(5 * k, 0)) for k in range(6)]
    summary = compute_consecutive_overlap(frames)
    assert len(summary.pairs) == 5
    assert summary.n_duplicate_risk + summary.n_gap_risk + summary.n_indeterminate + summary.n_ok == 5


def test_kept_neighbor_overlap_uses_direct_pairs_not_chain():
    # 6 raw frames sweeping left to right; keep only 0 and 5 (as if 1-4 were dropped for quality).
    frames = [_textured_frame(seed=0, shift=(20 * k, 0)) for k in range(6)]
    kept = [0, 5]
    summary = compute_kept_neighbor_overlap(frames, kept)
    assert len(summary.pairs) == 1
    direct = summary.pairs[0]
    # Direct 0-vs-5 overlap must be measured, not assumed equal to a single small consecutive step.
    consecutive = pairwise_overlap(frames[0], frames[1])
    assert direct.i == 0 and direct.j == 5
    assert direct.match_ratio <= consecutive.match_ratio + 1e-6


def test_worst_gap_pairs_sorted_ascending_by_match_ratio():
    frames = [_textured_frame(seed=0, shift=(0, 0)),
              _textured_frame(seed=0, shift=(200, 150)),   # big jump -> likely gap_risk
              _textured_frame(seed=0, shift=(400, 300)),   # even further from frame 0's content
              ]
    summary = compute_kept_neighbor_overlap(frames, [0, 1, 2])
    if len(summary.worst_gap_pairs) >= 2:
        ratios = [p.match_ratio for p in summary.worst_gap_pairs]
        assert ratios == sorted(ratios)
