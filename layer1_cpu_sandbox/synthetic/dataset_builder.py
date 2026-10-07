"""
dataset_builder.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Wires scene_generator + trajectory_generator + renderer + degradation into
one `SyntheticDroneDataset`. This is the single artifact every other module
(perception, reconstruction, evaluation) is built against.

CONTRACT (important -- this is what keeps the evaluation honest):
  * `DroneFrame.rgb`, `dataset.sensor_trace`, `dataset.K` are the ONLY
    things the perception/reconstruction pipeline is allowed to read.
  * `DroneFrame.rgb_clean`, `depth_gt`, `semantic_gt`, `dynamic_mask_gt`,
    `true_pose`, `dataset.gt_points`, `dataset.control_measurements` are
    EVALUATION-ONLY and must never be fed into a reconstruction module.
A module that reaches into the eval-only fields to do its job is, by
definition, cheating and its results should be discarded.
"""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np

from .scene_generator import SyntheticScene, generate_scene
from .trajectory_generator import SinglePassTrajectory, generate_single_pass_trajectory
from .camera_model import Intrinsics, CameraPose
from .renderer import render_frame, RenderResult
from .ground_truth import GTPointCloud, sample_surface_points, compute_control_measurements
from . import degradation as deg


@dataclasses.dataclass
class DroneFrame:
    idx: int
    t: float
    rgb: np.ndarray                # DEGRADED capture -- pipeline input
    rgb_clean: np.ndarray          # EVAL ONLY
    depth_gt: np.ndarray           # EVAL ONLY
    semantic_gt: np.ndarray        # EVAL ONLY
    dynamic_mask_gt: np.ndarray    # EVAL ONLY
    true_pose: CameraPose          # EVAL ONLY
    blur_px: float                 # EVAL ONLY (degradation ground truth, for robustness study)
    jpeg_quality: int              # EVAL ONLY


@dataclasses.dataclass
class SyntheticDroneDataset:
    scene: SyntheticScene
    trajectory: SinglePassTrajectory
    frames: List[DroneFrame]
    sensor_trace: "deg.NoisySensorTrace"   # pipeline input (noisy GPS/IMU/baro)
    K: Intrinsics                          # pipeline input (assumed known camera intrinsics)
    nominal_pitch_deg: float               # pipeline input ("flight metadata": commanded gimbal pitch)
    nominal_roll_deg: float                # pipeline input ("flight metadata")
    gt_points: GTPointCloud                # EVAL ONLY
    control_measurements: dict             # EVAL ONLY
    per_triangle_view_angle_by_frame: List[dict]   # EVAL ONLY (renderer visibility trace)
    per_triangle_visible_frac_by_frame: List[dict]  # EVAL ONLY


def build_dataset(seed: int = 0, extent: float = 40.0, width: int = 160, height: int = 120,
                   hfov_deg: float = 70.0, fps: float = 4.0, altitude: float = 45.0,
                   speed: float = 8.0, pitch_deg: float = -35.0, n_vehicles: int = 2,
                   degrade: bool = True, blur_strength: float = 1.0, jpeg_quality: int = 45,
                   gps_cfg: "deg.GPSIMUNoiseConfig" = None) -> SyntheticDroneDataset:
    scene = generate_scene(seed=seed, extent=extent)
    traj = generate_single_pass_trajectory(scene.scene_bounds, altitude=altitude, speed=speed,
                                            fps=fps, pitch_deg=pitch_deg, seed=seed)
    K = Intrinsics.from_fov(width, height, hfov_deg=hfov_deg)
    sensor_trace = deg.simulate_gps_imu(traj, cfg=gps_cfg or deg.GPSIMUNoiseConfig(), seed=seed)
    vehicle_tracks = deg.make_vehicle_tracks(scene, n_vehicles=n_vehicles, seed=seed)

    rng = np.random.default_rng(seed + 777)
    frames = []
    view_angle_by_frame = []
    visible_frac_by_frame = []

    for i, s in enumerate(traj.samples):
        pose = traj.camera_pose(i)
        extra = []
        for track in vehicle_tracks:
            extra += track.triangles_at(s.t)
        res: RenderResult = render_frame(scene, K, pose, extra_triangles=extra)

        view_angle_by_frame.append(res.per_triangle_view_angle)
        visible_frac_by_frame.append(res.per_triangle_visible_frac)

        rgb = res.rgb.copy()
        blur_px = 0.0
        jq = 100
        if degrade:
            speed_xy = np.linalg.norm(s.velocity[:2])
            blur_px = float(np.clip(blur_strength * speed_xy / 3.0, 0.0, 6.0))
            angle = np.arctan2(s.velocity[1], s.velocity[0]) + rng.normal(0, 0.15)
            rgb = deg.apply_motion_blur(rgb, blur_px, angle)

            gamma = float(np.clip(1.0 + 0.25 * np.sin(i * 0.35 + seed), 0.7, 1.4))
            shadow = rng.random() < 0.12
            rgb = deg.change_illumination(rgb, gamma=gamma, brightness_delta=int(rng.normal(0, 6)),
                                           shadow_band=shadow, rng=rng)
            rgb = deg.add_sensor_noise_and_vignette(rgb, noise_std=3.0, rng=rng)
            jq = int(np.clip(rng.normal(jpeg_quality, 8), 20, 90))
            rgb = deg.jpeg_compress(rgb, quality=jq)

        frames.append(DroneFrame(idx=i, t=s.t, rgb=rgb, rgb_clean=res.rgb, depth_gt=res.depth,
                                  semantic_gt=res.semantic, dynamic_mask_gt=res.dynamic_mask,
                                  true_pose=pose, blur_px=blur_px, jpeg_quality=jq))

    gt_points = sample_surface_points(scene, points_per_sqm=1.2, seed=seed)
    control = compute_control_measurements(scene)

    return SyntheticDroneDataset(scene=scene, trajectory=traj, frames=frames,
                                  sensor_trace=sensor_trace, K=K, nominal_pitch_deg=pitch_deg,
                                  nominal_roll_deg=0.0, gt_points=gt_points,
                                  control_measurements=control,
                                  per_triangle_view_angle_by_frame=view_angle_by_frame,
                                  per_triangle_visible_frac_by_frame=visible_frac_by_frame)


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    ds = build_dataset(seed=0)
    t1 = time.perf_counter()
    print(f"Built dataset: {len(ds.frames)} frames in {t1 - t0:.2f}s, "
          f"{len(ds.scene.triangles)} scene tris, {ds.gt_points.points.shape[0]} GT pts")
    print("mean blur px:", np.mean([f.blur_px for f in ds.frames]))
    print("mean jpeg q:", np.mean([f.jpeg_quality for f in ds.frames]))
    print("gps err (m), first 5:",
          np.linalg.norm(ds.sensor_trace.gps_xyz[:5] - np.stack([s.position for s in ds.trajectory.samples[:5]]), axis=1))
