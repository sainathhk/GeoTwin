"""
Tests for layer1_cpu_sandbox/reconstruction/gaussian_geometry.py. Pure numpy --
no torch, no open3d, no CUDA -- matching this module's own no-heavy-deps design.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from layer1_cpu_sandbox.reconstruction.gaussian_geometry import (
    quat_to_rotmat, compute_gaussian_geometry, suggest_threshold_from_valley, floater_mask,
)


def test_identity_quat_gives_identity_rotation():
    q = np.array([[1.0, 0.0, 0.0, 0.0]])  # (w,x,y,z) identity
    R = quat_to_rotmat(q)
    np.testing.assert_allclose(R[0], np.eye(3), atol=1e-10)


def test_normal_points_along_true_shortest_axis_after_rotation():
    """A flat disc-like Gaussian (thin in local Z) rotated 90deg about the world X axis
    should end up thin along world Y -- i.e. its recovered normal should be +/- world Y,
    NOT world Z. This is the check that would catch a wrong quaternion-order convention
    (e.g. (x,y,z,w) instead of (w,x,y,z)): a wrong convention still returns *a* unit
    vector of the right shape, just pointing the wrong way, so a shape-only test would
    pass while the actual geometry was silently incorrect."""
    scale = np.array([[1.0, 1.0, 0.05]])  # thin along LOCAL z
    half = np.pi / 4  # 90 degrees about X: quaternion (cos(theta/2), sin(theta/2), 0, 0)
    q = np.array([[np.cos(half), np.sin(half), 0.0, 0.0]])
    geom = compute_gaussian_geometry(scale, q)
    normal = geom.normals[0]
    # local z (0,0,1) rotated 90deg about world X -> world (0,-1,0) -- accept either sign.
    assert abs(abs(normal[1]) - 1.0) < 1e-6, f"expected normal ~= (0,+/-1,0), got {normal}"
    assert abs(normal[0]) < 1e-6 and abs(normal[2]) < 1e-6


def test_anisotropy_and_scale_mean():
    scale = np.array([[1.0, 2.0, 4.0], [0.5, 0.5, 0.5]])
    q = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    geom = compute_gaussian_geometry(scale, q)
    np.testing.assert_allclose(geom.anisotropy, [4.0, 1.0])
    np.testing.assert_allclose(geom.scale_mean, [7.0 / 3.0, 0.5])
    np.testing.assert_allclose(geom.scale_min, [1.0, 0.5])


def test_rejects_wrong_last_dim():
    import pytest
    with pytest.raises(ValueError):
        compute_gaussian_geometry(np.zeros((5, 2)), np.zeros((5, 4)))


def test_valley_threshold_finds_real_gap():
    """Synthetic bimodal distribution with a clear, deliberately-placed gap around
    0.5 -- mirrors the real checkpoint's confidence histogram shape (see
    docs/MESH_QUALITY_AUDIT.md): a low cluster, then a real gap, then a high cluster."""
    rng = np.random.default_rng(0)
    low = rng.normal(0.35, 0.05, 2000)
    high = rng.normal(0.85, 0.05, 8000)
    values = np.clip(np.concatenate([low, high]), 0, 1)
    thresh = suggest_threshold_from_valley(values)
    assert 0.5 <= thresh <= 0.7, f"expected the valley threshold near 0.5-0.7, got {thresh}"


def test_valley_threshold_falls_back_on_monotonic_distribution():
    """A smooth, monotonically-decaying distribution (like the real checkpoint's OPACITY
    histogram, see audit doc) has no interior valley -- must fall back to a percentile,
    not raise or return something nonsensical like 0 or 1."""
    rng = np.random.default_rng(1)
    values = rng.exponential(0.2, 20000)
    values = np.clip(values, 0, 1)
    thresh = suggest_threshold_from_valley(values)
    assert 0.0 < thresh < 0.5


def test_floater_mask_flags_big_round_gaussians_only():
    # 3 small flat (normal surface Gaussians), 1 big round (floater-like)
    scale = np.array([[1.0, 1.0, 0.02], [1.0, 1.0, 0.02], [1.0, 1.0, 0.02], [5.0, 5.0, 5.0]])
    q = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    geom = compute_gaussian_geometry(scale, q)
    mask = floater_mask(geom, anisotropy_below=3.0, scale_above_percentile=50.0)
    assert mask.tolist() == [False, False, False, True]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
