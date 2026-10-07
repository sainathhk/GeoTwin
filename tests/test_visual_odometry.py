import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_cpu_sandbox.real_data.visual_odometry import (
    VisualOdometryError,
    _compose_pose,
    visual_odometry_poses,
)
from layer1_cpu_sandbox.synthetic.camera_model import CameraPose, Intrinsics


def test_compose_pose_uses_opencv_relative_pose_convention():
    previous = CameraPose(position=np.zeros(3), R_wc=np.eye(3))
    # OpenCV convention: X_current = R X_previous + t.
    angle = np.deg2rad(90.0)
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    current = _compose_pose(previous, R, np.array([1.0, 0.0, 0.0]), 2.0)
    assert np.allclose(current.R_wc, R.T)
    assert np.allclose(current.position, -R.T @ np.array([2.0, 0.0, 0.0]))


def test_visual_odometry_fails_closed_on_featureless_frames():
    blank = np.zeros((48, 64, 3), dtype=np.uint8)
    pose0 = CameraPose(position=np.zeros(3), R_wc=np.eye(3))
    pose1 = CameraPose(position=np.array([1.0, 0.0, 0.0]), R_wc=np.eye(3))
    frames = [SimpleNamespace(idx=0, rgb=blank, pose=pose0),
              SimpleNamespace(idx=1, rgb=blank, pose=pose1)]
    K = Intrinsics.from_fov(64, 48, 84.0)
    try:
        visual_odometry_poses(frames, K, min_matches=8, min_inliers=5)
        assert False, "featureless frames must not produce a guessed pose"
    except VisualOdometryError as e:
        assert "ORB matches" in str(e)
