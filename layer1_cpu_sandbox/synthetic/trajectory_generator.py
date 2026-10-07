"""
trajectory_generator.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Generates the GROUND-TRUTH single-pass drone trajectory: one continuous
flight path over the scene (NOT a lawnmower grid with multiple overlapping
passes -- that would defeat the entire point of PS 26158). The camera is
mounted at a forward-oblique pitch so a single traverse still observes both
rooftops and facades, matching how a real single-pass reconneissance flight
would be flown.

This module outputs only ground truth. Sensor noise (GPS/IMU) is injected
downstream in `degradation.py` so that "true trajectory" and "what the
sensors actually reported" stay cleanly separated -- this separation is
what lets the evaluation framework later report a real, non-fabricated
"georeferencing error" number.
"""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np

from .camera_model import CameraPose


@dataclasses.dataclass
class TrajectorySample:
    t: float                 # seconds since flight start
    position: np.ndarray      # (3,) true world position, meters ENU
    roll: float
    pitch: float              # negative = nose down (oblique/nadir look)
    yaw: float
    velocity: np.ndarray       # (3,) m/s, finite-differenced
    accel: np.ndarray          # (3,) m/s^2, finite-differenced (for IMU synthesis)


@dataclasses.dataclass
class SinglePassTrajectory:
    samples: List[TrajectorySample]
    fps: float
    seed: int

    def camera_pose(self, i: int) -> CameraPose:
        s = self.samples[i]
        return CameraPose.from_flight_attitude(s.position, s.roll, s.pitch, s.yaw)

    def __len__(self):
        return len(self.samples)


def generate_single_pass_trajectory(scene_bounds, altitude=45.0, speed=8.0, fps=4.0,
                                     pitch_deg=-35.0, seed=0, wobble=True) -> SinglePassTrajectory:
    """
    scene_bounds: (xmin, xmax, ymin, ymax) from the SyntheticScene.
    altitude: nominal AGL flight altitude (m).
    speed: nominal ground speed (m/s).
    fps: frame sampling rate used for reconstruction (NOT the raw video fps --
         this is the rate at which we pull candidate frames from the stream).
    pitch_deg: nominal camera pitch, negative = looking forward-and-down
               (oblique), which is what lets a single pass see facades.
    wobble: adds small drone-flight-realistic path curvature / altitude drift,
            distinct from sensor NOISE (this is true vehicle motion, not
            measurement error).
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = scene_bounds

    # A single diagonal traverse across the scene, not a grid: this is what
    # makes the problem "single-pass" -- roughly half the scene's facades
    # will simply never be seen, by design.
    start = np.array([xmin - 5.0, ymin + (ymax - ymin) * 0.35, altitude])
    end = np.array([xmax + 5.0, ymin + (ymax - ymin) * 0.65, altitude])
    path_length = np.linalg.norm(end[:2] - start[:2])
    duration = path_length / speed
    n_samples = max(8, int(duration * fps))

    ts = np.linspace(0, duration, n_samples)
    alpha = ts / max(duration, 1e-6)

    positions = start[None, :] + alpha[:, None] * (end - start)[None, :]

    if wobble:
        # Gentle true-path curvature + altitude drift (thermals, wind drift),
        # smooth and low-frequency so it is physically plausible flight
        # motion rather than sensor jitter.
        lateral = 3.0 * np.sin(alpha * 2.5 * np.pi + rng.uniform(0, 2 * np.pi))
        alt_drift = 2.0 * np.sin(alpha * 1.7 * np.pi + rng.uniform(0, 2 * np.pi))
        heading = np.arctan2(end[1] - start[1], end[0] - start[0])
        perp = np.array([-np.sin(heading), np.cos(heading)])
        positions[:, 0] += lateral * perp[0]
        positions[:, 1] += lateral * perp[1]
        positions[:, 2] += alt_drift

    # Yaw follows the direction of travel (finite-diff heading).
    dpos = np.gradient(positions, ts, axis=0)
    yaw = np.arctan2(dpos[:, 1], dpos[:, 0])
    pitch = np.full(n_samples, np.deg2rad(pitch_deg))
    if wobble:
        pitch += np.deg2rad(3.0) * np.sin(alpha * 4.0 * np.pi)
    roll = np.deg2rad(4.0) * np.sin(alpha * 3.1 * np.pi) if wobble else np.zeros(n_samples)

    velocity = np.gradient(positions, ts, axis=0)
    accel = np.gradient(velocity, ts, axis=0)

    samples = [
        TrajectorySample(t=float(ts[i]), position=positions[i].astype(np.float64),
                          roll=float(roll[i]), pitch=float(pitch[i]), yaw=float(yaw[i]),
                          velocity=velocity[i], accel=accel[i])
        for i in range(n_samples)
    ]
    return SinglePassTrajectory(samples=samples, fps=fps, seed=seed)


if __name__ == "__main__":
    traj = generate_single_pass_trajectory((-40, 40, -40, 40), seed=0)
    print(f"{len(traj)} samples over {traj.samples[-1].t:.1f}s")
    print("first pose pos:", traj.samples[0].position, "yaw(deg):", np.rad2deg(traj.samples[0].yaw))
    print("last pose pos:", traj.samples[-1].position)
