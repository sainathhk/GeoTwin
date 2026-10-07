"""
Tests for the 2026-09 mesh-quality-audit additions to
layer1_cpu_sandbox/reconstruction/depth_estimation.py: `depth_spacing` on
plane_sweep_depth/estimate_depth_for_sequence, and the new
check_depth_range_coverage diagnostic. See that module's docstring for why
these exist -- docs/MESH_QUALITY_AUDIT.md has the full investigation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from layer1_cpu_sandbox.reconstruction.depth_estimation import (
    plane_sweep_depth, estimate_depth_for_sequence, check_depth_range_coverage, DepthEstimate,
)
from layer1_cpu_sandbox.synthetic.camera_model import Intrinsics, CameraPose


def _checkerboard(size=64, square=8):
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    board = (((xs // square) + (ys // square)) % 2).astype(np.uint8) * 255
    return board


def _two_camera_fronto_scene(true_depth=20.0):
    """A small textured fronto-parallel 'wall' at a known depth, seen by a reference
    camera and one neighbor shifted sideways -- just enough parallax for plane-sweep
    to have a real, checkable answer, at a size small enough to run in well under a
    second per call."""
    K = Intrinsics.from_fov(64, 64, hfov_deg=50.0)
    ref_pose = CameraPose.from_flight_attitude(position=[0.0, 0.0, 0.0], roll=0.0, pitch=0.0, yaw=0.0)
    nb_pose = CameraPose.from_flight_attitude(position=[1.5, 0.0, 0.0], roll=0.0, pitch=0.0, yaw=0.0)
    # Both cameras look along the SAME +Z body axis (approx) -- reuse one synthetic
    # texture as "the wall" for both views; plane-sweep's own homography warp (not
    # this fixture) is what has to find the depth, so a flat shared texture plus a
    # small x-shift between camera centers is sufficient signal for that.
    ref_gray = _checkerboard()
    nb_gray = _checkerboard()
    return ref_gray, nb_gray, ref_pose, nb_pose, K, true_depth


def test_invalid_depth_spacing_raises():
    ref_gray, nb_gray, ref_pose, nb_pose, K, _ = _two_camera_fronto_scene()
    with pytest.raises(ValueError):
        plane_sweep_depth(ref_gray, ref_pose, [nb_gray], [nb_pose], K, depth_spacing="not_a_real_option")


def test_log_spacing_requires_positive_depth_min():
    ref_gray, nb_gray, ref_pose, nb_pose, K, _ = _two_camera_fronto_scene()
    with pytest.raises(ValueError):
        plane_sweep_depth(ref_gray, ref_pose, [nb_gray], [nb_pose], K,
                           depth_min=0.0, depth_max=50.0, depth_spacing="log")


def test_log_and_linear_spacing_both_run_and_resolve_pixels():
    ref_gray, nb_gray, ref_pose, nb_pose, K, true_depth = _two_camera_fronto_scene(true_depth=20.0)
    est_log = plane_sweep_depth(ref_gray, ref_pose, [nb_gray], [nb_pose], K,
                                 depth_min=2.0, depth_max=80.0, n_depths=48, depth_spacing="log")
    est_lin = plane_sweep_depth(ref_gray, ref_pose, [nb_gray], [nb_pose], K,
                                 depth_min=2.0, depth_max=80.0, n_depths=48, depth_spacing="linear")
    assert est_log.depth.shape == (64, 64)
    assert (est_log.depth > 0).mean() > 0.5, "expected most pixels to resolve on a textured scene"
    assert (est_lin.depth > 0).mean() > 0.5

    # the RESOLVED depth values are drawn from the respective hypothesis grid --
    # every resolved log-spacing depth should be (approximately) a geomspace member,
    # not a linspace member, and vice versa. This is what actually distinguishes
    # "the code took the log branch" from "the code silently ignored depth_spacing".
    log_grid = np.geomspace(2.0, 80.0, 48)
    lin_grid = np.linspace(2.0, 80.0, 48)
    resolved_log = np.unique(est_log.depth[est_log.depth > 0])
    resolved_lin = np.unique(est_lin.depth[est_lin.depth > 0])
    assert all(np.min(np.abs(log_grid - d)) < 1e-3 for d in resolved_log)
    assert all(np.min(np.abs(lin_grid - d)) < 1e-3 for d in resolved_lin)
    # and the two grids genuinely differ from each other in the interior (not just at
    # the shared endpoints), so this test would actually fail if log silently fell
    # back to linear
    assert np.abs(log_grid[24] - lin_grid[24]) > 0.5


def test_estimate_depth_for_sequence_passes_depth_spacing_through():
    ref_gray, nb_gray, ref_pose, nb_pose, K, _ = _two_camera_fronto_scene()
    results = estimate_depth_for_sequence([ref_gray, nb_gray], [ref_pose, nb_pose], K, window=1,
                                           depth_min=2.0, depth_max=80.0, n_depths=32, depth_spacing="log")
    assert len(results) == 2
    log_grid = np.geomspace(2.0, 80.0, 32)
    for est in results:
        resolved = np.unique(est.depth[est.depth > 0])
        if resolved.size:
            assert all(np.min(np.abs(log_grid - d)) < 1e-3 for d in resolved)


def _fake_depth_estimate(depth_values, frame_idx=0):
    return DepthEstimate(frame_idx=frame_idx, depth=np.asarray(depth_values, dtype=np.float32),
                          consistency=np.ones_like(depth_values, dtype=np.float32),
                          min_cost=np.zeros_like(depth_values, dtype=np.float32), n_neighbors_used=1)


def test_coverage_flags_undersized_range():
    # 80% of resolved depths piled right at the far boundary -- exactly the aliasing
    # symptom this function exists to catch.
    depths = np.concatenate([np.full(800, 99.0), np.full(200, 30.0)]).reshape(1, -1)
    report = check_depth_range_coverage([_fake_depth_estimate(depths)], depth_min=3.0, depth_max=100.0)
    assert report.likely_range_too_small
    assert report.frac_near_max_boundary > 0.15
    assert "depth_max" in report.warning


def test_coverage_ok_when_depths_are_well_distributed():
    rng = np.random.default_rng(0)
    depths = rng.uniform(10.0, 60.0, size=(1, 1000)).astype(np.float32)  # nowhere near either boundary
    report = check_depth_range_coverage([_fake_depth_estimate(depths)], depth_min=3.0, depth_max=100.0)
    assert not report.likely_range_too_small
    assert report.warning == ""


def test_coverage_handles_no_resolved_pixels():
    depths = np.zeros((1, 100), dtype=np.float32)  # nothing resolved
    report = check_depth_range_coverage([_fake_depth_estimate(depths)], depth_min=3.0, depth_max=100.0)
    assert report.frac_resolved == 0.0
    assert not report.likely_range_too_small  # no data isn't the same claim as "range too small"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
