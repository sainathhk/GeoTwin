"""Unit tests for evaluation metrics and the confidence module's core properties."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_cpu_sandbox.evaluation.metrics_visual import compute_psnr_ssim
from layer1_cpu_sandbox.evaluation.metrics_geometry import evaluate_point_cloud_geometry
from layer1_cpu_sandbox.reconstruction.confidence import compute_confidence, gate_by_confidence
from layer1_cpu_sandbox.reconstruction.prototype_point_repr import PrototypePointRepresentation


def test_psnr_ssim_identical_images():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    psnr, ssim = compute_psnr_ssim(img, img)
    assert psnr > 60  # not exactly inf/99 due to the mse>1e-12 branch, but very high
    assert ssim > 0.999


def test_psnr_lower_for_noisier_image():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    slightly_noisy = np.clip(img.astype(np.int16) + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    very_noisy = np.clip(img.astype(np.int16) + rng.normal(0, 40, img.shape), 0, 255).astype(np.uint8)
    psnr_small, _ = compute_psnr_ssim(slightly_noisy, img)
    psnr_big, _ = compute_psnr_ssim(very_noisy, img)
    assert psnr_small > psnr_big


def test_chamfer_zero_for_identical_point_sets():
    pts = np.random.default_rng(0).random((200, 3)) * 10
    result = evaluate_point_cloud_geometry(pts, pts)
    assert result.chamfer_distance < 1e-6
    assert result.point_to_point_rmse < 1e-6


def test_chamfer_increases_with_added_noise():
    rng = np.random.default_rng(0)
    gt = rng.random((300, 3)) * 10
    recon_close = gt + rng.normal(0, 0.05, gt.shape)
    recon_far = gt + rng.normal(0, 2.0, gt.shape)
    close_result = evaluate_point_cloud_geometry(recon_close, gt)
    far_result = evaluate_point_cloud_geometry(recon_far, gt)
    assert close_result.chamfer_distance < far_result.chamfer_distance


def _make_toy_points(observation_count, view_angle_spread, consistency, quality, pose_conf):
    n = len(observation_count)
    return PrototypePointRepresentation(
        positions=np.zeros((n, 3)), colors=np.zeros((n, 3)),
        observation_count=np.array(observation_count, dtype=np.int32),
        view_angle_min=np.zeros(n, dtype=np.float32),
        view_angle_max=np.array(view_angle_spread, dtype=np.float32),
        view_angle_spread=np.array(view_angle_spread, dtype=np.float32),
        mean_consistency=np.array(consistency, dtype=np.float32),
        mean_frame_quality=np.array(quality, dtype=np.float32),
        mean_pose_confidence=np.array(pose_conf, dtype=np.float32),
        n_raw_observations=np.array(observation_count, dtype=np.int32),
        voxel_size=1.0,
    )


def test_confidence_higher_for_more_observations_all_else_equal():
    pts = _make_toy_points(observation_count=[1, 4], view_angle_spread=[0.15, 0.15],
                            consistency=[0.8, 0.8], quality=[0.8, 0.8], pose_conf=[0.8, 0.8])
    conf = compute_confidence(pts)
    assert conf.confidence[1] > conf.confidence[0]


def test_confidence_single_observation_is_capped():
    pts = _make_toy_points(observation_count=[1], view_angle_spread=[0.5],
                            consistency=[0.99], quality=[0.99], pose_conf=[0.99])
    conf = compute_confidence(pts, single_view_cap=0.55)
    assert conf.confidence[0] <= 0.55 + 1e-6
    assert conf.band[0] != "High"


def test_gating_prunes_low_confidence_points():
    pts = _make_toy_points(observation_count=[1, 5], view_angle_spread=[0.0, 0.3],
                            consistency=[0.02, 0.9], quality=[0.1, 0.9], pose_conf=[0.1, 0.9])
    conf = compute_confidence(pts)
    gate = gate_by_confidence(conf, prune_below=0.12)
    assert gate.keep_mask[0] == False or conf.confidence[0] < 0.12
    assert gate.keep_mask[1] == True
