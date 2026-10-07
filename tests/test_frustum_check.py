"""
Tests for layer1_cpu_sandbox/reconstruction/frustum_check.py.

The critical property is BOTH directions: it must fire on a frustum-filled cloud
AND stay quiet on a genuine scene. A detector that always says "frustum" would
have caught the 2026-09 failure by accident while being useless as a gate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from layer1_cpu_sandbox.reconstruction.frustum_check import (
    check_frustum_shape, vfov_from_hfov, FrustumCheckResult,
)


def _frustum_filled_cloud(n=40000, hfov_deg=84.0, vfov_deg=53.7, depth_min=3.0, depth_max=100.0, seed=0):
    """Back-project random pixels to random depths inside the tested range -- i.e. exactly
    what a depth estimator that carries no information produces. Camera at the origin
    looking down -Z, so the cone's apex is at Z max."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(depth_min, depth_max, n)
    tx = np.tan(np.radians(hfov_deg) / 2)
    ty = np.tan(np.radians(vfov_deg) / 2)
    x = rng.uniform(-tx, tx, n) * d
    y = rng.uniform(-ty, ty, n) * d
    z = -d
    return np.stack([x, y, z], axis=1)


def _genuine_city_scene(n=40000, seed=0):
    """A real-ish aerial scene: a broad flat ground plane with box-shaped buildings rising
    from it. Wide and flat, NOT tapering to a point -- the shape a working reconstruction
    of this kind of footage should have."""
    rng = np.random.default_rng(seed)
    n_ground = int(n * 0.6)
    ground = np.stack([rng.uniform(-120, 120, n_ground), rng.uniform(-120, 120, n_ground),
                        rng.normal(0, 0.3, n_ground)], axis=1)
    blocks = []
    n_left = n - n_ground
    for _ in range(25):
        cx, cy = rng.uniform(-110, 110), rng.uniform(-110, 110)
        w, dpt, h = rng.uniform(6, 18), rng.uniform(6, 18), rng.uniform(8, 35)
        k = n_left // 25
        blocks.append(np.stack([cx + rng.uniform(-w/2, w/2, k), cy + rng.uniform(-dpt/2, dpt/2, k),
                                 rng.uniform(0, h, k)], axis=1))
    return np.concatenate([ground] + blocks, axis=0)


def test_fires_on_frustum_filled_cloud():
    pts = _frustum_filled_cloud()
    res = check_frustum_shape(pts, expected_hfov_deg=84.0, expected_vfov_deg=53.7)
    assert res.is_frustum_shaped, res.format_report()
    assert res.r_squared > 0.95
    assert res.apex_intercept_relative < 0.08
    assert res.fov_match_deg is not None and res.fov_match_deg < 8.0


def test_recovers_the_correct_opening_angles():
    """The measured opening angles should match the frustum the cloud was built from --
    this is what makes the 'matches the camera FOV' claim meaningful rather than circular."""
    pts = _frustum_filled_cloud(hfov_deg=84.0, vfov_deg=53.7)
    res = check_frustum_shape(pts, expected_hfov_deg=84.0, expected_vfov_deg=53.7)
    assert res.implied_fov_deg_wide == pytest.approx(84.0, abs=5.0), res.format_report()
    assert res.implied_fov_deg_narrow == pytest.approx(53.7, abs=5.0), res.format_report()


def test_does_not_fire_on_a_genuine_city_scene():
    """The important negative case. A flat ground plane with buildings must NOT be flagged."""
    pts = _genuine_city_scene()
    res = check_frustum_shape(pts, expected_hfov_deg=84.0, expected_vfov_deg=53.7)
    assert not res.is_frustum_shaped, res.format_report()


def test_does_not_fire_on_a_sphere():
    """Another negative case: a closed, non-conical object."""
    rng = np.random.default_rng(3)
    v = rng.normal(size=(20000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = v * 50.0 + rng.normal(0, 0.5, (20000, 3))
    res = check_frustum_shape(pts, expected_hfov_deg=84.0, expected_vfov_deg=53.7)
    assert not res.is_frustum_shaped, res.format_report()


def test_conical_but_wrong_angle_is_not_flagged_when_fov_is_known():
    """A genuinely cone-shaped SCENE (e.g. a quarry) whose opening angle does not match the
    camera should not be flagged as a frustum -- the FOV match is what distinguishes the two."""
    pts = _frustum_filled_cloud(hfov_deg=20.0, vfov_deg=14.0)  # far narrower than the camera
    res = check_frustum_shape(pts, expected_hfov_deg=84.0, expected_vfov_deg=53.7)
    assert not res.is_frustum_shaped
    assert res.r_squared > 0.9  # still conical...
    assert res.fov_match_deg > 8.0  # ...but the angle exonerates it


def test_without_expected_fov_it_reports_the_weaker_test():
    pts = _frustum_filled_cloud()
    res = check_frustum_shape(pts)
    assert res.fov_match_deg is None
    assert any("weaker conical-shape test" in r for r in res.reasons)


def test_degenerate_inputs_do_not_raise():
    assert not check_frustum_shape(np.zeros((10, 3))).is_frustum_shaped      # too few points
    assert not check_frustum_shape(np.zeros((500, 3))).is_frustum_shaped     # zero extent
    with pytest.raises(ValueError):
        check_frustum_shape(np.zeros((500, 2)))                               # wrong shape


def test_vfov_from_hfov_matches_known_case():
    # 1280x720 at 84 deg horizontal -> ~53.7 deg vertical
    assert vfov_from_hfov(84.0, 1280, 720) == pytest.approx(53.7, abs=0.5)


def test_report_is_human_readable():
    res = check_frustum_shape(_frustum_filled_cloud(), expected_hfov_deg=84.0, expected_vfov_deg=53.7)
    report = res.format_report()
    assert "FRUSTUM-SHAPED" in report and "R^2" in report and "opening angles" in report


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
