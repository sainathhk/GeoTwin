"""
degradation.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Turns the CLEAN, ground-truth trajectory/frames from
trajectory_generator.py / renderer.py into what a real single-pass drone
mission would actually hand the pipeline: noisy GPS, biased/noisy IMU,
motion-blurred and compressed frames, illumination changes, moving
vehicles, and dropped/sparse frames.

Keeping "true" (trajectory_generator/renderer) and "measured" (this file)
strictly separate is what makes every later accuracy number
(georeferencing error, scale error, pose error...) a real, computed
difference instead of a number pulled out of the air.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import cv2

from .scene_generator import SyntheticScene, make_box_triangles
from .trajectory_generator import SinglePassTrajectory


# ---------------------------------------------------------------- sensors --

@dataclasses.dataclass
class GPSIMUNoiseConfig:
    gps_sigma_xy_m: float = 2.5          # civilian-GPS-like horizontal 1-sigma
    gps_sigma_z_m: float = 5.0           # vertical is typically worse than horizontal
    gps_outlier_prob: float = 0.03       # occasional multipath/dropout spike
    gps_outlier_scale_m: float = 12.0
    imu_accel_noise_std: float = 0.25    # m/s^2 white noise
    imu_accel_bias: float = 0.08         # m/s^2 slowly-varying bias
    imu_gyro_noise_std_deg: float = 0.6  # deg/s white noise on attitude rate
    baro_noise_std_m: float = 0.6
    baro_drift_std_m: float = 1.5        # slow drift over the flight


@dataclasses.dataclass
class NoisySensorTrace:
    t: np.ndarray
    gps_xyz: np.ndarray          # (N,3) noisy GPS position estimate
    gps_is_outlier: np.ndarray   # (N,) bool
    accel_meas: np.ndarray       # (N,3) noisy+biased accelerometer (world frame, for simplicity)
    gyro_meas_dps: np.ndarray    # (N,3) noisy gyro (roll/pitch/yaw rates, deg/s)
    baro_alt: np.ndarray         # (N,) noisy barometric altitude


def simulate_gps_imu(traj: SinglePassTrajectory, cfg: GPSIMUNoiseConfig = GPSIMUNoiseConfig(),
                      seed: int = 0) -> NoisySensorTrace:
    rng = np.random.default_rng(seed)
    n = len(traj)
    true_pos = np.stack([s.position for s in traj.samples])
    true_acc = np.stack([s.accel for s in traj.samples])
    ts = np.array([s.t for s in traj.samples])

    xy_noise = rng.normal(0, cfg.gps_sigma_xy_m, size=(n, 2))
    z_noise = rng.normal(0, cfg.gps_sigma_z_m, size=(n,))
    is_outlier = rng.random(n) < cfg.gps_outlier_prob
    outlier_vec = rng.normal(0, cfg.gps_outlier_scale_m, size=(n, 3)) * is_outlier[:, None]

    gps = true_pos.copy()
    gps[:, :2] += xy_noise
    gps[:, 2] += z_noise
    gps += outlier_vec

    accel_bias = rng.normal(0, cfg.imu_accel_bias, size=3)
    accel_meas = true_acc + accel_bias[None, :] + rng.normal(0, cfg.imu_accel_noise_std, size=(n, 3))

    # True attitude rates via finite differencing roll/pitch/yaw.
    rpy = np.stack([[s.roll, s.pitch, s.yaw] for s in traj.samples])
    rpy_rate = np.gradient(np.rad2deg(np.unwrap(rpy, axis=0)), ts, axis=0)
    gyro_meas = rpy_rate + rng.normal(0, cfg.imu_gyro_noise_std_deg, size=(n, 3))

    drift = np.cumsum(rng.normal(0, cfg.baro_drift_std_m / np.sqrt(max(n, 1)), size=n))
    baro = true_pos[:, 2] + drift + rng.normal(0, cfg.baro_noise_std_m, size=n)

    return NoisySensorTrace(t=ts, gps_xyz=gps, gps_is_outlier=is_outlier,
                             accel_meas=accel_meas, gyro_meas_dps=gyro_meas, baro_alt=baro)


# ------------------------------------------------------------- image-space --

def apply_motion_blur(rgb: np.ndarray, speed_pixels: float, angle_rad: float) -> np.ndarray:
    """Linear motion-blur kernel sized by apparent motion (px) during exposure."""
    k = max(1, int(round(speed_pixels)))
    if k <= 1:
        return rgb
    k = k + (1 - k % 2)  # force odd kernel size
    kernel = np.zeros((k, k), dtype=np.float32)
    cx = cy = k // 2
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    for i in range(k):
        t = i - cx
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    if kernel.sum() == 0:
        kernel[cy, cx] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def jpeg_compress(rgb: np.ndarray, quality: int = 35) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def change_illumination(rgb: np.ndarray, gamma: float = 1.0, brightness_delta: int = 0,
                         shadow_band: bool = False, rng: np.random.Generator = None) -> np.ndarray:
    img = rgb.astype(np.float32) / 255.0
    img = np.clip(img, 1e-4, 1.0) ** gamma
    img = img * 255.0 + brightness_delta
    if shadow_band:
        rng = rng or np.random.default_rng()
        h, w = rgb.shape[:2]
        band_center = rng.uniform(0, w)
        band_width = rng.uniform(w * 0.15, w * 0.4)
        xs = np.arange(w)
        atten = 1.0 - 0.45 * np.exp(-0.5 * ((xs - band_center) / (band_width / 2)) ** 2)
        img = img * atten[None, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def add_sensor_noise_and_vignette(rgb: np.ndarray, noise_std: float = 4.0,
                                   rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, noise_std, size=rgb.shape)
    return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ dynamic objs --

@dataclasses.dataclass
class VehicleTrack:
    object_id: int
    start_xy: np.ndarray
    velocity_xy: np.ndarray
    size_wdh: tuple
    color: tuple

    def triangles_at(self, t: float):
        pos = self.start_xy + self.velocity_xy * t
        w, d, h = self.size_wdh
        return make_box_triangles((pos[0], pos[1]), 0.0, w, d, h, self.color,
                                   object_id=self.object_id, dynamic=True)


def make_vehicle_tracks(scene: SyntheticScene, n_vehicles: int = 3, seed: int = 0,
                         road_half_width: float = 2.6) -> list:
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = scene.scene_bounds
    tracks = []
    for i in range(n_vehicles):
        y = rng.uniform(-road_half_width * 0.7, road_half_width * 0.7)
        direction = 1 if i % 2 == 0 else -1
        speed = rng.uniform(6.0, 12.0) * direction
        x0 = xmin if direction > 0 else xmax
        color = tuple(np.clip(rng.random(3) * 0.6 + 0.3, 0, 1))
        tracks.append(VehicleTrack(object_id=90000 + i, start_xy=np.array([x0, y]),
                                    velocity_xy=np.array([speed, 0.0]),
                                    size_wdh=(4.2, 1.8, 1.5), color=color))
    return tracks


# ------------------------------------------------------------- frame drop --

def select_sparse_frame_indices(n_frames: int, keep_fraction: float, seed: int = 0) -> np.ndarray:
    """Uniform-ish random thinning of the frame set (simulates dropped/discarded frames)."""
    rng = np.random.default_rng(seed)
    keep_fraction = float(np.clip(keep_fraction, 0.05, 1.0))
    n_keep = max(2, int(round(n_frames * keep_fraction)))
    idx = np.sort(rng.choice(n_frames, size=n_keep, replace=False))
    return idx
