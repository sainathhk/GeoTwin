"""
sensor_fusion.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested
LAYER 2 INTERFACE: layer2_interfaces/i_pose_estimator.py

Implements module 5 of the required pipeline ("GPS/IMU fusion") and half of
module 4 ("camera motion estimation"): a constant-acceleration-input Kalman
filter fuses noisy GPS position fixes with IMU accelerometer readings into
a smoothed position/velocity trajectory, with an explicit, propagated
position-uncertainty (this is what feeds the confidence framework's
"pose confidence" term later).

Position/velocity estimation here is a *linear* Kalman filter (the process
and measurement models are linear); heading is recovered non-linearly via
atan2 of the filtered velocity, and pitch/roll are tracked with a
complementary filter around the commanded gimbal attitude (flight
metadata). Calling the whole thing "EKF-style GPS/IMU fusion" reflects that
combination, not a claim that every sub-step is a textbook EKF.

Explicitly NOT implemented at this layer (by design -- see
layer2_interfaces/i_pose_estimator.py): feature-based visual-odometry pose
refinement / bundle adjustment. Doc 2's own priority list only asks for
"camera trajectory estimation", which the GPS/IMU filter below satisfies;
recreating a full VO/BA stack as a slow, fragile CPU approximation was
explicitly out of scope ("do not spend excessive time recreating
sophisticated GPU research algorithms... create clean interfaces instead").
A GPU-stage upgrade path (COLMAP-style incremental SfM, or a learned pose
regressor) is documented in the interface file instead.
"""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np

from ..synthetic.camera_model import CameraPose, rotmat_from_euler
from ..synthetic.degradation import NoisySensorTrace, GPSIMUNoiseConfig


@dataclasses.dataclass
class FusedPoseEstimate:
    t: float
    position: np.ndarray        # (3,) filtered position estimate
    velocity: np.ndarray        # (3,) filtered velocity estimate
    yaw: float
    pitch: float
    roll: float
    position_cov: np.ndarray    # (3,3) filter covariance for position block
    position_std_m: float       # sqrt(trace(cov)/3), single-number summary
    gps_used: bool               # False if this step's GPS update was gated out as an outlier


@dataclasses.dataclass
class FusedTrajectory:
    estimates: List[FusedPoseEstimate]

    def camera_pose(self, i: int) -> CameraPose:
        e = self.estimates[i]
        return CameraPose.from_flight_attitude(e.position, e.roll, e.pitch, e.yaw)

    def pose_confidence(self, i: int, std_scale_m: float = 4.0) -> float:
        """Maps position uncertainty to a [0,1] confidence, saturating past `std_scale_m`."""
        e = self.estimates[i]
        return float(np.clip(1.0 - e.position_std_m / std_scale_m, 0.0, 1.0))

    def __len__(self):
        return len(self.estimates)


def fuse_gps_imu(sensor_trace: NoisySensorTrace, nominal_pitch_deg: float, nominal_roll_deg: float = 0.0,
                  cfg: GPSIMUNoiseConfig = None, mahalanobis_gate: float = 9.0) -> FusedTrajectory:
    cfg = cfg or GPSIMUNoiseConfig()
    n = sensor_trace.t.shape[0]
    dt_arr = np.diff(sensor_trace.t, prepend=sensor_trace.t[0])
    dt_arr[0] = dt_arr[1] if n > 1 else 0.1

    x = np.zeros(6)
    x[:3] = sensor_trace.gps_xyz[0]
    P = np.eye(6) * 25.0

    q_accel = max(cfg.imu_accel_noise_std, 1e-3) ** 2
    R_gps = np.diag([cfg.gps_sigma_xy_m ** 2, cfg.gps_sigma_xy_m ** 2, cfg.gps_sigma_z_m ** 2])
    H = np.zeros((3, 6)); H[:3, :3] = np.eye(3)

    estimates = []
    pitch_est = np.deg2rad(nominal_pitch_deg)
    roll_est = np.deg2rad(nominal_roll_deg)
    yaw_est = 0.0
    nominal_pitch_rad = np.deg2rad(nominal_pitch_deg)
    nominal_roll_rad = np.deg2rad(nominal_roll_deg)

    for k in range(n):
        dt = float(max(dt_arr[k], 1e-3))
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        B = np.zeros((6, 3))
        B[:3, :] = 0.5 * dt * dt * np.eye(3)
        B[3:, :] = dt * np.eye(3)
        u = sensor_trace.accel_meas[k]

        # --- predict ---
        x = F @ x + B @ u
        Qblock = q_accel * np.block([[0.25 * dt ** 4 * np.eye(3), 0.5 * dt ** 3 * np.eye(3)],
                                      [0.5 * dt ** 3 * np.eye(3), dt ** 2 * np.eye(3)]])
        P = F @ P @ F.T + Qblock

        # --- update (GPS), with Mahalanobis-gated outlier rejection ---
        z = sensor_trace.gps_xyz[k]
        y = z - H @ x
        S = H @ P @ H.T + R_gps
        d2 = float(y.T @ np.linalg.solve(S, y))
        gps_used = d2 <= mahalanobis_gate
        if gps_used:
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(6) - K @ H) @ P

        # --- heading from filtered velocity ---
        vxy_norm = np.hypot(x[3], x[4])
        if vxy_norm > 0.5:
            yaw_est = float(np.arctan2(x[4], x[3]))

        # --- pitch/roll: complementary filter toward commanded gimbal attitude ---
        gyro = sensor_trace.gyro_meas_dps[k]
        pitch_gyro = pitch_est + np.deg2rad(gyro[1]) * dt
        roll_gyro = roll_est + np.deg2rad(gyro[0]) * dt
        alpha = 0.85
        pitch_est = alpha * pitch_gyro + (1 - alpha) * nominal_pitch_rad
        roll_est = alpha * roll_gyro + (1 - alpha) * nominal_roll_rad

        pos_cov = P[:3, :3].copy()
        pos_std = float(np.sqrt(max(np.trace(pos_cov), 0.0) / 3.0))

        estimates.append(FusedPoseEstimate(t=float(sensor_trace.t[k]), position=x[:3].copy(),
                                            velocity=x[3:].copy(), yaw=yaw_est, pitch=pitch_est,
                                            roll=roll_est, position_cov=pos_cov,
                                            position_std_m=pos_std, gps_used=gps_used))

    return FusedTrajectory(estimates=estimates)


def naive_pose_from_raw_gps(sensor_trace: NoisySensorTrace, nominal_pitch_deg: float,
                             nominal_roll_deg: float = 0.0) -> FusedTrajectory:
    """
    ABLATION BASELINE ONLY: "no sensor fusion" -- camera trajectory taken directly
    from the raw, noisy GPS fix each step (no Kalman filtering, no outlier
    rejection), with heading from simple finite-differenced consecutive raw GPS
    positions and pitch/roll held at the nominal commanded gimbal angle. This is
    what a naive pipeline (ignoring PS challenge #5, "GPS inaccuracies and sensor
    noise") would do, and it exists so `evaluate_metric_accuracy` /
    `evaluate_point_cloud_geometry` can report a real, measured before/after
    number for the EKF fusion module rather than an asserted one.
    """
    n = sensor_trace.t.shape[0]
    pos = sensor_trace.gps_xyz.copy()
    dpos = np.gradient(pos, sensor_trace.t, axis=0)
    yaw = np.arctan2(dpos[:, 1], dpos[:, 0])
    pitch = np.full(n, np.deg2rad(nominal_pitch_deg))
    roll = np.full(n, np.deg2rad(nominal_roll_deg))

    estimates = []
    for k in range(n):
        estimates.append(FusedPoseEstimate(t=float(sensor_trace.t[k]), position=pos[k], velocity=dpos[k],
                                            yaw=float(yaw[k]), pitch=float(pitch[k]), roll=float(roll[k]),
                                            position_cov=np.eye(3) * 25.0, position_std_m=5.0, gps_used=True))
    return FusedTrajectory(estimates=estimates)


if __name__ == "__main__":
    from ..synthetic.trajectory_generator import generate_single_pass_trajectory
    from ..synthetic.degradation import simulate_gps_imu

    traj = generate_single_pass_trajectory((-40, 40, -40, 40), seed=0)
    trace = simulate_gps_imu(traj, seed=0)
    fused = fuse_gps_imu(trace, nominal_pitch_deg=-35.0)

    true_pos = np.stack([s.position for s in traj.samples])
    fused_pos = np.stack([e.position for e in fused.estimates])
    raw_gps_err = np.linalg.norm(trace.gps_xyz - true_pos, axis=1)
    fused_err = np.linalg.norm(fused_pos - true_pos, axis=1)
    print(f"raw GPS RMSE:   {np.sqrt((raw_gps_err**2).mean()):.2f} m")
    print(f"fused pose RMSE:{np.sqrt((fused_err**2).mean()):.2f} m")
    print(f"outliers gated: {sum(1 for e in fused.estimates if not e.gps_used)}/{len(fused)}")
