"""
test_vggt_integration.py -- proves vggt_pose_depth_source.py's glue code is
correct WITHOUT needing a GPU or a real VGGT forward pass.

Method: build a small synthetic flat-ground scene with KNOWN true (telemetry)
camera poses and KNOWN true per-pixel ground-plane depth. Apply a randomly
chosen but KNOWN Sim(3) "corruption" transform to emulate VGGT's arbitrary
relative-scale coordinate frame (see the derivation in
vggt_pose_depth_source.py's module docstring for why depth scales by the same
factor as position). Feed the corrupted (pose, depth) pairs through
build_pose_depth_source() exactly as real VGGT output would arrive, then
through THIS REPO'S REAL fuse_point_cloud()/compute_confidence() (not a
mock), and check the recovered scale/poses/points match the known truth.

This test can and should be run before spending Colab GPU time -- if this
fails, the real run will fail too, for the same reason.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_cpu_sandbox.synthetic.camera_model import CameraPose, Intrinsics
from layer1_cpu_sandbox.real_data.real_dataset_builder import RealFrame
from layer1_cpu_sandbox.real_data.vggt_pose_depth_source import (
    umeyama_alignment, vggt_extrinsic_to_camera_pose, build_pose_depth_source, VGGTFrameResult)
from layer1_cpu_sandbox.reconstruction.prototype_point_repr import fuse_point_cloud
from layer1_cpu_sandbox.reconstruction.confidence import compute_confidence


def _random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _make_synthetic_scene(rng, n_frames=6, h=48, w=64):
    """n_frames cameras along a rough line at ~50m altitude, all looking
    straight down (+Z world = up; camera looks down its own +Z, see
    camera_model.py) at a flat ground plane z=0. Returns:
      true_poses, true_positions, K, per-frame true depth maps (H,W)."""
    K = Intrinsics.from_fov(w, h, hfov_deg=84.0)
    true_positions = np.stack([
        np.array([10.0 * i, 0.5 * i + rng.normal(scale=0.3), 50.0 + rng.normal(scale=0.5)])
        for i in range(n_frames)
    ])
    # looking straight down: camera +Z (optical axis) points along world -Z.
    # CameraPose.forward() = R_wc @ [0,0,1]; want this ~= [0,0,-1].
    R_wc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    true_poses = [CameraPose(position=true_positions[i], R_wc=R_wc.copy()) for i in range(n_frames)]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    rx, ry = (us - K.cx) / K.fx, (vs - K.cy) / K.fy
    true_depths = []
    for pose in true_poses:
        # ray in camera frame: (rx, ry, 1); world dir = R_wc @ ray. Ground at world z=0:
        # pose.position[2] + t * (R_wc @ ray)[2] = 0  =>  t = -pos_z / dir_z  (this IS the depth,
        # since camera-frame ray already has unit "z" component of 1 -- see _project() in
        # depth_estimation.py, which this mirrors).
        dir_z = (R_wc[2, 0] * rx + R_wc[2, 1] * ry + R_wc[2, 2] * 1.0)
        depth = -pose.position[2] / dir_z
        true_depths.append(depth.astype(np.float32))
    return true_poses, true_positions, K, true_depths


def test_umeyama_recovers_known_transform():
    rng = np.random.default_rng(0)
    true_poses, true_positions, K, true_depths = _make_synthetic_scene(rng)

    scale_c, R_c, t_c = 0.37, _random_rotation(rng), np.array([120.0, -40.0, 300.0])
    vggt_positions = (scale_c * (R_c @ true_positions.T).T) + t_c

    scale, R, t = umeyama_alignment(vggt_positions, true_positions)
    recovered = scale * (R @ vggt_positions.T).T + t
    assert np.allclose(recovered, true_positions, atol=1e-6), \
        "Umeyama did not recover the injected similarity transform"
    assert np.isclose(scale, 1.0 / scale_c, rtol=1e-6)


def test_extrinsic_conversion_round_trips():
    rng = np.random.default_rng(1)
    R_wc = _random_rotation(rng)
    position = rng.normal(size=3) * 50
    pose = CameraPose(position=position, R_wc=R_wc)

    # build the VGGT/OpenCV-style world-to-camera 4x4 the way pose_estimation.py does
    R_cw = pose.R_cw()
    t_cw = -R_cw @ pose.position
    extrinsic = np.eye(4)
    extrinsic[:3, :3], extrinsic[:3, 3] = R_cw, t_cw

    recovered = vggt_extrinsic_to_camera_pose(extrinsic)
    assert np.allclose(recovered.position, position, atol=1e-8)
    assert np.allclose(recovered.R_wc, R_wc, atol=1e-8)


def test_build_pose_depth_source_end_to_end_against_real_fusion_code():
    rng = np.random.default_rng(2)
    true_poses, true_positions, K, true_depths = _make_synthetic_scene(rng)
    n = len(true_poses)

    # ---- emulate VGGT's arbitrary relative-scale frame (known corruption) ----
    scale_c, R_c, t_c = 0.37, _random_rotation(rng), np.array([120.0, -40.0, 300.0])
    extrinsics = np.zeros((n, 4, 4))
    depth_maps, depth_confs, image_sizes = [], [], []
    for i, pose in enumerate(true_poses):
        vggt_R_wc = R_c @ pose.R_wc
        vggt_position = scale_c * (R_c @ pose.position) + t_c
        R_cw_vggt = vggt_R_wc.T
        t_cw_vggt = -R_cw_vggt @ vggt_position
        extrinsics[i, :3, :3], extrinsics[i, :3, 3], extrinsics[i, 3, 3] = R_cw_vggt, t_cw_vggt, 1.0
        depth_maps.append(true_depths[i] * scale_c)  # depth scales with position, see module docstring
        depth_confs.append(rng.uniform(0.5, 3.0, size=true_depths[i].shape).astype(np.float32))
        image_sizes.append(true_depths[i].shape)
    intrinsics = np.tile(K.K()[None, :, :], (n, 1, 1))
    vggt_result = VGGTFrameResult(extrinsics=extrinsics, intrinsics=intrinsics,
                                   depth_maps=depth_maps, depth_confs=depth_confs, image_sizes=image_sizes)

    # ---- build RealFrame stand-ins (telemetry position = ground truth; rgb = flat mid-grey) ----
    frames = []
    for i in range(n):
        rgb = np.full((*true_depths[i].shape, 3), 128, dtype=np.uint8)
        frames.append(RealFrame(idx=i, t=float(i), rgb=rgb, pose=true_poses[i],
                                 gps_lat=0.0, gps_lon=0.0, position_residual_m=0.0))

    poses, depth_estimates, Ks, frame_quality, pose_conf = build_pose_depth_source(frames, vggt_result,
                                                                                   verbose=False)

    # ---- poses should match true telemetry poses closely ----
    recovered_positions = np.stack([p.position for p in poses])
    assert np.allclose(recovered_positions, true_positions, atol=1e-4), \
        f"max position error {np.abs(recovered_positions - true_positions).max():.6f}m"
    for p_est, p_true in zip(poses, true_poses):
        assert np.allclose(p_est.R_wc, p_true.R_wc, atol=1e-6)

    assert all(0.0 <= c <= 1.0 for c in pose_conf)
    assert all(0.0 <= q <= 1.0 for q in frame_quality)

    # ---- feed straight into THIS REPO'S REAL fusion + confidence code ----
    fused = fuse_point_cloud(depth_estimates, dynamic_masks=None, poses=poses,
                              rgb_frames=[f.rgb for f in frames], frame_quality_scores=frame_quality,
                              pose_confidences=pose_conf, K=Ks, voxel_size=0.4)
    assert fused.positions.shape[0] > 0, "fusion produced zero points -- something upstream is broken"
    # ground truth is z=0 everywhere; fused points should land close to that
    assert np.abs(fused.positions[:, 2]).max() < 0.5, \
        f"fused points strayed from the true z=0 ground plane by up to {np.abs(fused.positions[:,2]).max():.3f}m"
    # and within the flight's real XY footprint (cameras run x=0..50m at ~50m altitude, 84 deg
    # HFOV, so each camera's ground footprint extends roughly +-45m beyond its own position --
    # NOT off in VGGT's corrupted frame, whose t_c alone was [120,-40,300])
    assert fused.positions[:, 0].min() > -60 and fused.positions[:, 0].max() < 110

    conf = compute_confidence(fused)
    assert conf.confidence.shape[0] == fused.positions.shape[0]
    assert np.all((conf.confidence >= 0) & (conf.confidence <= 1))
    print(f"\nEnd-to-end OK: {fused.positions.shape[0]} fused points, "
          f"mean confidence {conf.confidence.mean():.3f}, "
          f"band counts {dict(zip(*np.unique(conf.band, return_counts=True)))}")


if __name__ == "__main__":
    test_umeyama_recovers_known_transform()
    test_extrinsic_conversion_round_trips()
    test_build_pose_depth_source_end_to_end_against_real_fusion_code()
    print("\nALL PASS")
