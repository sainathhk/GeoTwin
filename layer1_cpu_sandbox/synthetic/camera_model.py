"""
camera_model.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Standard pinhole camera model. This is intentionally the *same* intrinsics
representation the GPU stage will use (fx, fy, cx, cy + world_T_cam SE(3)),
so poses estimated/refined here are drop-in compatible with the Layer 3
Gaussian representation and with `diff-gaussian-rasterization`-style camera
structs.
"""
from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float):
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        fy = fx  # square pixels
        return cls(width, height, fx, fy, width / 2.0, height / 2.0)

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]], dtype=np.float64)


def rotmat_from_euler(roll, pitch, yaw) -> np.ndarray:
    """
    Body-321 (yaw-pitch-roll) rotation, radians. Returns R such that
    v_world = R @ v_body. Sign convention: NEGATIVE pitch tilts the body's
    forward axis DOWN (nose-down / looking toward the ground), positive
    pitch tilts it up -- this matches how `pitch_deg` is used throughout
    the trajectory generator (negative = oblique-down survey look angle).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


# Camera-body axis convention: camera looks down its own +Z (optical axis),
# +X right, +Y down (standard CV convention). We rotate camera-frame into a
# body frame where the nominal orientation looks along body +X, +Z up, then
# apply the flight roll/pitch/yaw on top of that.
_CAM_TO_BODY_NOMINAL = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, -1, 0],
], dtype=np.float64)


@dataclasses.dataclass
class CameraPose:
    """world_T_cam: camera position in world + rotation mapping cam-frame dirs to world."""
    position: np.ndarray   # (3,) world position (meters, ENU)
    R_wc: np.ndarray       # (3,3) world_from_camera rotation

    def R_cw(self) -> np.ndarray:
        return self.R_wc.T

    def world_to_cam(self, pts_world: np.ndarray) -> np.ndarray:
        """pts_world: (...,3) -> camera-frame coords (...,3)."""
        rel = pts_world - self.position
        return rel @ self.R_wc  # since R_cw = R_wc.T, (R_cw @ rel.T).T = rel @ R_wc

    def cam_to_world(self, pts_cam: np.ndarray) -> np.ndarray:
        return pts_cam @ self.R_cw() + self.position

    @staticmethod
    def from_flight_attitude(position, roll, pitch, yaw):
        R_body_from_world_nominal = rotmat_from_euler(roll, pitch, yaw)
        R_wc = R_body_from_world_nominal @ _CAM_TO_BODY_NOMINAL
        return CameraPose(position=np.asarray(position, dtype=np.float64), R_wc=R_wc)

    def forward(self) -> np.ndarray:
        return self.R_wc @ np.array([0, 0, 1.0])

    def viewing_ray_angle_to(self, point_world: np.ndarray) -> float:
        """Angle (radians) between the optical axis and the ray to a world point."""
        v = point_world - self.position
        n = np.linalg.norm(v)
        if n < 1e-9:
            return 0.0
        v = v / n
        f = self.forward()
        cosang = np.clip(np.dot(v, f), -1.0, 1.0)
        return float(np.arccos(cosang))


def project_points(K: np.ndarray, pts_cam: np.ndarray):
    """
    pts_cam: (N,3) points already in camera frame (z = depth, +z forward).
    Returns pixel coords (N,2) and depth (N,), with invalid (behind camera)
    marked depth<=0.
    """
    z = pts_cam[..., 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    x = pts_cam[..., 0] / safe_z
    y = pts_cam[..., 1] / safe_z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=-1), z
