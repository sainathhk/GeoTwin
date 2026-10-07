"""
Regression tests for per-frame camera intrinsics (a zoom lens mid-clip -- see
real_dataset_builder.py's per-frame K, added 2026-09 for the DJI focal_len issue).

WHY THIS FILE EXISTS: the 2026-09-21 fix that first threaded per-frame K through
TrainingCamera/visual_odometry.py/dynamic_object_filter.py did NOT catch that
depth_estimation.py's plane_sweep_depth/estimate_depth_for_sequence and
prototype_point_repr.py's fuse_point_cloud ALSO built one shared ray-direction grid
from a single K, applied to every frame regardless of its own real focal length. Both
bugs were completely silent -- no exception, no obviously-wrong shape, just a plausible-
looking wrong depth/position for any frame whose real K differed from the one passed in.
The only real-world symptom was a downstream one three steps removed from the actual
bug (79-96% of pixels flagged "dynamic" by a filter that trusted the resulting depth,
and the post-training frustum check on the exported Gaussians failing outright) --
exactly the kind of thing that is expensive to diagnose from a training log and cheap
to catch here. If you touch camera-intrinsics plumbing in either module, run this file.

Every test below builds TWO cameras at DIFFERENT focal lengths (a "wide" and a "tele",
3x apart -- matching this project's real DJI Air 3 zoom range), places a synthetic
scene point or textured plane at a KNOWN 3D location/depth, and checks that the
function under test recovers something close to ground truth. Where useful, a test also
runs the OLD single-shared-K call style (no neighbor_Ks / a plain K instead of a list)
to confirm it's still measurably wrong -- proof the test is actually discriminating,
not just permissive.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from layer1_cpu_sandbox.synthetic.camera_model import Intrinsics, CameraPose
from layer1_cpu_sandbox.reconstruction.depth_estimation import plane_sweep_depth, estimate_depth_for_sequence
from layer1_cpu_sandbox.reconstruction.prototype_point_repr import fuse_point_cloud, PrototypePointRepresentation
from layer1_cpu_sandbox.perception.dynamic_object_filter import _expected_rigid_flow, detect_dynamic_pixels
from layer1_cpu_sandbox.reconstruction.depth_estimation import DepthEstimate


WIDE = Intrinsics(width=200, height=200, fx=100.0, fy=100.0, cx=100.0, cy=100.0)   # 24mm-like
TELE = Intrinsics(width=200, height=200, fx=300.0, fy=300.0, cx=100.0, cy=100.0)   # 70mm-like (3x)


def _texture(world_xy):
    x, y = world_xy[..., 0], world_xy[..., 1]
    return (128 + 100 * np.sin(x * 0.8) * np.cos(y * 0.6)).astype(np.uint8)


def _fronto_wall_pair(true_depth=25.0, baseline=2.0):
    """A textured fronto-parallel wall at world Z=true_depth. ref camera is WIDE at the
    origin; neighbor camera is TELE, shifted `baseline` metres in +X. Both images are
    built on THEIR OWN pixel grid by construction (not by remapping one into the
    other's grid), so there is no risk of a remap-direction bug in the fixture itself."""
    ref_pose = CameraPose(position=np.array([0., 0., 0.]), R_wc=np.eye(3))
    nb_pose = CameraPose(position=np.array([baseline, 0., 0.]), R_wc=np.eye(3))

    us, vs = np.meshgrid(np.arange(WIDE.width), np.arange(WIDE.height))
    rx, ry = (us - WIDE.cx) / WIDE.fx, (vs - WIDE.cy) / WIDE.fy
    world = np.stack([rx * true_depth, ry * true_depth, np.full_like(rx, true_depth, dtype=np.float64)], axis=-1)
    world = world @ ref_pose.R_cw() + ref_pose.position
    ref_gray = _texture(world[..., :2])

    us_n, vs_n = np.meshgrid(np.arange(TELE.width), np.arange(TELE.height))
    rx_n, ry_n = (us_n - TELE.cx) / TELE.fx, (vs_n - TELE.cy) / TELE.fy
    d_n = np.full_like(rx_n, true_depth, dtype=np.float64)  # solved for world Z == true_depth given nb_pose
    world_n = np.stack([rx_n * d_n, ry_n * d_n, d_n], axis=-1) @ nb_pose.R_cw() + nb_pose.position
    nb_gray = _texture(world_n[..., :2])

    return ref_gray, nb_gray, ref_pose, nb_pose


def test_plane_sweep_depth_wide_ref_tele_neighbor_recovers_true_depth():
    ref_gray, nb_gray, ref_pose, nb_pose = _fronto_wall_pair(true_depth=25.0)
    est = plane_sweep_depth(ref_gray, ref_pose, [nb_gray], [nb_pose], WIDE,
                             depth_min=5.0, depth_max=60.0, n_depths=60, match_window=5,
                             neighbor_Ks=[TELE])
    # Central region only: TELE's much narrower FOV genuinely can't see the WIDE frame's
    # edges (a real, expected coverage limit -- not a bug), so restrict the check to
    # where both cameras' view cones actually overlap.
    region = np.s_[85:115, 85:115]
    resolved = est.depth[region]
    assert (resolved > 0).mean() > 0.9, "expected near-complete coverage in the guaranteed-overlap region"
    assert abs(np.median(resolved[resolved > 0]) - 25.0) < 1.0, "recovered depth should be close to the true 25.0"


def test_plane_sweep_depth_without_neighbor_Ks_is_measurably_wrong():
    """Confirms the test above is actually discriminating: reusing WIDE for the TELE
    neighbor (the pre-2026-09-21 behavior) should NOT recover the true depth, even
    though it will still confidently report *a* depth for every pixel."""
    ref_gray, nb_gray, ref_pose, nb_pose = _fronto_wall_pair(true_depth=25.0)
    est_bug = plane_sweep_depth(ref_gray, ref_pose, [nb_gray], [nb_pose], WIDE,
                                 depth_min=5.0, depth_max=60.0, n_depths=60, match_window=5)
    region = np.s_[85:115, 85:115]
    resolved = est_bug.depth[region]
    assert (resolved > 0).mean() > 0.9  # still "confidently" resolves almost everywhere...
    assert abs(np.median(resolved[resolved > 0]) - 25.0) > 3.0  # ...just to the wrong depth


def test_estimate_depth_for_sequence_accepts_per_frame_K_list():
    ref_gray, nb_gray, ref_pose, nb_pose = _fronto_wall_pair(true_depth=25.0)
    ests = estimate_depth_for_sequence([ref_gray, nb_gray], [ref_pose, nb_pose], [WIDE, TELE],
                                        window=1, depth_min=5.0, depth_max=60.0, n_depths=60)
    assert len(ests) == 2
    region = np.s_[85:115, 85:115]
    resolved = ests[0].depth[region]
    assert abs(np.median(resolved[resolved > 0]) - 25.0) < 1.0

    with pytest.raises(ValueError):
        estimate_depth_for_sequence([ref_gray, nb_gray], [ref_pose, nb_pose], [WIDE],  # wrong length
                                     window=1, depth_min=5.0, depth_max=60.0, n_depths=60)


def test_fuse_point_cloud_unprojects_each_frame_with_its_own_K():
    pose = CameraPose(position=np.array([0., 0., 0.]), R_wc=np.eye(3))
    true_point = np.array([2.0, 1.0, 20.0])

    def project(K, pt):
        cam = (pt - pose.position) @ pose.R_wc
        return K.fx * cam[0] / cam[2] + K.cx, K.fy * cam[1] / cam[2] + K.cy, cam[2]

    u_w, v_w, z = project(WIDE, true_point)
    u_t, v_t, _ = project(TELE, true_point)

    def make_est(u, v):
        d = np.zeros((200, 200), dtype=np.float32)
        d[int(round(v)), int(round(u))] = z
        c = np.ones((200, 200), dtype=np.float32)
        return DepthEstimate(frame_idx=0, depth=d, consistency=c,
                              min_cost=np.zeros((200, 200), np.float32), n_neighbors_used=1, boundary_fraction=0.0)

    est_wide, est_tele = make_est(u_w, v_w), make_est(u_t, v_t)
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)

    result = fuse_point_cloud([est_wide, est_tele], None, [pose, pose], [rgb, rgb],
                               [1.0, 1.0], [1.0, 1.0], [WIDE, TELE], voxel_size=10.0)
    # Both observations describe the SAME real point -- with correct per-frame K they
    # must fuse into (approximately) ONE voxel at true_point, not two different ones.
    assert result.positions.shape[0] == 1
    assert np.allclose(result.positions[0], true_point, atol=0.1)


def test_fuse_point_cloud_single_shared_K_does_not_collapse_to_one_point():
    """Same setup, but the pre-fix call style (one K for every frame) -- the two
    observations of the SAME real point should NOT land in the same place."""
    pose = CameraPose(position=np.array([0., 0., 0.]), R_wc=np.eye(3))
    true_point = np.array([2.0, 1.0, 20.0])

    def project(K, pt):
        cam = (pt - pose.position) @ pose.R_wc
        return K.fx * cam[0] / cam[2] + K.cx, K.fy * cam[1] / cam[2] + K.cy, cam[2]

    u_w, v_w, z = project(WIDE, true_point)
    u_t, v_t, _ = project(TELE, true_point)

    def make_est(u, v):
        d = np.zeros((200, 200), dtype=np.float32)
        d[int(round(v)), int(round(u))] = z
        c = np.ones((200, 200), dtype=np.float32)
        return DepthEstimate(frame_idx=0, depth=d, consistency=c,
                              min_cost=np.zeros((200, 200), np.float32), n_neighbors_used=1, boundary_fraction=0.0)

    est_wide, est_tele = make_est(u_w, v_w), make_est(u_t, v_t)
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)

    result = fuse_point_cloud([est_wide, est_tele], None, [pose, pose], [rgb, rgb],
                               [1.0, 1.0], [1.0, 1.0], WIDE, voxel_size=10.0)  # single shared K
    assert not (result.positions.shape[0] == 1 and np.allclose(result.positions[0], true_point, atol=0.1))


def test_expected_rigid_flow_matches_hand_computed_projection_across_zoom_levels():
    """Direct check of the dynamic-object-filter's expected-flow math in isolation from
    Farneback optical flow (which needs real image content and is not what this test is
    about) -- already fixed in the 2026-09-21 session that started this file, kept here
    so a future refactor can't quietly re-break it alongside the other two."""
    pose_a = CameraPose(position=np.array([0., 0., 0.]), R_wc=np.eye(3))
    pose_b = CameraPose(position=np.array([1.0, 0., 0.]), R_wc=np.eye(3))
    u0, v0, d0 = 130, 115, 20.0
    depth_a = np.zeros((200, 200), dtype=np.float32)
    depth_a[v0, u0] = d0

    rx, ry = (u0 - WIDE.cx) / WIDE.fx, (v0 - WIDE.cy) / WIDE.fy
    world = np.array([rx * d0, ry * d0, d0]) @ pose_a.R_cw() + pose_a.position
    cam_b = (world - pose_b.position) @ pose_b.R_wc
    expected_u = TELE.fx * cam_b[0] / cam_b[2] + TELE.cx
    expected_v = TELE.fy * cam_b[1] / cam_b[2] + TELE.cy

    exp_u_field, exp_v_field, valid = _expected_rigid_flow(depth_a, pose_a, pose_b, WIDE, TELE)
    assert np.isclose(u0 + exp_u_field[v0, u0], expected_u, atol=1e-2)
    assert np.isclose(v0 + exp_v_field[v0, u0], expected_v, atol=1e-2)
