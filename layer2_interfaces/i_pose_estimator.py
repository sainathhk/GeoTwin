"""
i_pose_estimator.py -- LAYER 2 INTERFACE CONTRACT
STATUS: implemented (interface only), CPU-tested (against sensor_fusion.py)

CPU PROTOTYPE:       layer1_cpu_sandbox/perception/sensor_fusion.py
                      (linear KF on GPS+accel, complementary-filtered
                      pitch/roll around commanded gimbal attitude, heading
                      from filtered velocity)
GPU REPLACEMENT:      (a) a proper multiplicative EKF/UKF on SE(3) fusing
                      GPS+IMU+barometer with a full IMU bias model, OR
                      (b) incremental visual-SLAM / bundle adjustment
                      (COLMAP-style, or a learned pose regressor such as
                      DROID-SLAM / a transformer pose network) refining the
                      GPS/IMU prior with actual image feature matches --
                      this is the "visual odometry" upgrade explicitly
                      deferred from Layer 1 (see sensor_fusion.py docstring)
PRETRAINED CANDIDATE: DROID-SLAM (public weights) as a drop-in visual
                      pose-refinement stage on top of the GPS/IMU prior
CUSTOM CANDIDATE:     a lightweight pose-correction network trained to
                      predict the residual between GPS/IMU pose and true
                      pose, supervised directly by this project's own
                      synthetic trajectory generator (true trajectory is
                      always known in simulation)
RESEARCH NOVELTY:     MODERATE -- fusing a *learned* visual residual with a
                      classical GPS/IMU EKF, gated by the same confidence
                      framework used for reconstruction, is a reasonable
                      paper-worthy extension (confidence-gated sensor
                      fusion, not just confidence-gated geometry).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class IPoseEstimator(ABC):
    @abstractmethod
    def fuse(self, sensor_trace, nominal_pitch_deg: float, nominal_roll_deg: float = 0.0):
        """Consume a NoisySensorTrace (GPS/IMU/baro), return a fused trajectory
        object exposing `.camera_pose(i)` and `.pose_confidence(i)`, matching
        layer1_cpu_sandbox.perception.sensor_fusion.FusedTrajectory's contract."""
        raise NotImplementedError
