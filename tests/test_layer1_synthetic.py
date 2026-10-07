"""Unit tests for the Layer-1 synthetic data generator. Run with:
    cd sih2026-3d-recon && python3 -m pytest tests/ -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_cpu_sandbox.synthetic.scene_generator import generate_scene
from layer1_cpu_sandbox.synthetic.trajectory_generator import generate_single_pass_trajectory
from layer1_cpu_sandbox.synthetic.camera_model import Intrinsics
from layer1_cpu_sandbox.synthetic.renderer import render_frame
from layer1_cpu_sandbox.synthetic.dataset_builder import build_dataset


def test_scene_generation_is_deterministic_and_nonempty():
    s1 = generate_scene(seed=0, extent=30.0)
    s2 = generate_scene(seed=0, extent=30.0)
    assert len(s1.triangles) == len(s2.triangles)
    assert len(s1.triangles) > 0
    assert len(s1.buildings) >= 4
    for b in s1.buildings:
        assert b["h"] > 0 and b["w"] > 0 and b["d"] > 0


def test_scene_generation_different_seeds_differ():
    s1 = generate_scene(seed=0, extent=30.0)
    s2 = generate_scene(seed=1, extent=30.0)
    h1 = sorted(round(b["h"], 3) for b in s1.buildings)
    h2 = sorted(round(b["h"], 3) for b in s2.buildings)
    assert h1 != h2


def test_trajectory_is_single_continuous_pass():
    traj = generate_single_pass_trajectory((-30, 30, -30, 30), seed=0)
    assert len(traj) > 5
    xs = [s.position[0] for s in traj.samples]
    assert xs[-1] > xs[0], "trajectory should traverse the scene, not loop back (single pass)"


def test_renderer_produces_visible_geometry():
    scene = generate_scene(seed=0, extent=30.0)
    traj = generate_single_pass_trajectory(scene.scene_bounds, seed=0)
    K = Intrinsics.from_fov(160, 120, hfov_deg=70)
    mid = len(traj) // 2
    res = render_frame(scene, K, traj.camera_pose(mid))
    assert (res.depth > 0).sum() > 0, "renderer produced an entirely empty frame"
    assert res.rgb.max() > 0


def test_dataset_builder_end_to_end_shapes():
    ds = build_dataset(seed=0, extent=25.0, width=96, height=72, fps=3.0)
    assert len(ds.frames) > 3
    for f in ds.frames:
        assert f.rgb.shape == (72, 96, 3)
        assert f.depth_gt.shape == (72, 96)
    assert ds.gt_points.points.shape[0] > 0
    assert ds.sensor_trace.gps_xyz.shape == (len(ds.frames), 3)


def test_degraded_frames_differ_from_clean_frames():
    ds = build_dataset(seed=0, extent=25.0, width=96, height=72, fps=3.0, degrade=True)
    n_diff = 0
    for f in ds.frames:
        if not np.array_equal(f.rgb, f.rgb_clean):
            n_diff += 1
    assert n_diff > 0, "degradation pipeline had no effect on any frame"
