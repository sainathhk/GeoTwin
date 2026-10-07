"""Fast, image-only relative pose estimation for DJI SRT-backed runs.

DJI FrameCnt SRT files record position and focal length but no camera attitude.
Using the flight direction as camera yaw and a guessed fixed pitch makes every
plane-sweep homography depend on an assumption rather than image evidence.  This
module is deliberately a small visual-odometry front end, not COLMAP: it only
matches consecutive already-extracted frames, recovers their relative rotation
and translation direction, and uses the telemetry step length to set scale.

It is intended for a short, single clip.  It fails closed when the image pairs
do not contain enough geometrically consistent features; callers must not fall
back silently to the guessed SRT attitudes in that case.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable

import cv2
import numpy as np

from ..synthetic.camera_model import CameraPose, Intrinsics


class VisualOdometryError(RuntimeError):
    """Raised when image evidence is insufficient to create reliable poses."""


@dataclasses.dataclass(frozen=True)
class VisualOdometryStats:
    n_frames_input: int
    n_frames: int
    n_frames_skipped: int
    n_pairs: int
    total_matches: int
    total_inliers: int
    median_inliers: float
    telemetry_path_m: float
    n_cross_zoom_pairs: int = 0  # pairs whose two frames had different focal_len_mm (per-frame-K path)


def _match_orb(gray_a: np.ndarray, gray_b: np.ndarray, n_features: int,
               ratio_test: float) -> tuple[np.ndarray, np.ndarray, int]:
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=7)
    kpa, desa = orb.detectAndCompute(gray_a, None)
    kpb, desb = orb.detectAndCompute(gray_b, None)
    if desa is None or desb is None:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desa, desb, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair]
            if m.distance < ratio_test * n.distance]
    if not good:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    a = np.float32([kpa[m.queryIdx].pt for m in good])
    b = np.float32([kpb[m.trainIdx].pt for m in good])
    return a, b, len(good)


def _compose_pose(previous: CameraPose, R_current_from_previous: np.ndarray,
                  t_current: np.ndarray, step_scale: float) -> CameraPose:
    """Compose OpenCV's ``X_current = R X_previous + t`` with a world pose."""
    R_wc = previous.R_wc @ R_current_from_previous.T
    position = previous.position - R_wc @ (t_current.reshape(3) * step_scale)
    return CameraPose(position=position, R_wc=R_wc)


def _normalize_points(pts: np.ndarray, K: Intrinsics) -> np.ndarray:
    """Pixel -> normalized camera coordinates using K's OWN fx/fy/cx/cy. Needed so a frame
    pair with DIFFERENT intrinsics (a zoom lens -- see real_dataset_builder.py's per-frame K)
    can still form a geometrically valid essential-matrix pair: findEssentialMat/recoverPose
    only accept ONE shared cameraMatrix, which is wrong the moment the two frames were
    captured at different focal lengths. Normalizing each side with its own K first, then
    calling with an identity cameraMatrix, is the standard way to hand OpenCV a
    two-different-cameras pair."""
    out = np.empty_like(pts, dtype=np.float64)
    out[:, 0] = (pts[:, 0] - K.cx) / K.fx
    out[:, 1] = (pts[:, 1] - K.cy) / K.fy
    return out


def visual_odometry_poses(frames: Iterable, K: Intrinsics, *, n_features: int = 5000,
                          ratio_test: float = 0.75, min_matches: int = 80,
                          min_inliers: int = 40, min_step_m: float = 0.05,
                          max_frame_gap: int = 3) -> tuple[list[int], list[CameraPose], VisualOdometryStats]:
    """Estimate consecutive image poses, scaled by consecutive telemetry steps.

    ``frames`` must expose ``rgb`` and a telemetry-derived ``pose.position``.
    Rotation comes entirely from the images.  Telemetry is used only for the
    magnitude of each translation, so a constant GPS/world-frame offset or an
    unknown SRT gimbal attitude cannot rotate the recovered camera chain.
    """
    frames = list(frames)
    if len(frames) < 2:
        raise VisualOdometryError("need at least two frames for visual odometry")

    grays = [cv2.cvtColor(f.rgb, cv2.COLOR_RGB2GRAY) for f in frames]
    if max_frame_gap < 1:
        raise ValueError(f"max_frame_gap must be >= 1, got {max_frame_gap}")
    poses = [frames[0].pose]
    accepted = [0]
    pair_inliers: list[int] = []
    total_matches = 0
    total_inliers = 0
    telemetry_path_m = 0.0
    n_cross_zoom_pairs = 0

    def _frame_K(frame) -> Intrinsics:
        return getattr(frame, "K", None) or K

    previous_i = 0
    while previous_i < len(frames) - 1:
        failures = []
        accepted_pair = None
        for current_i in range(previous_i + 1, min(len(frames), previous_i + max_frame_gap + 1)):
            points_prev, points_current, n_matches = _match_orb(
                grays[previous_i], grays[current_i], n_features=n_features, ratio_test=ratio_test)
            total_matches += n_matches
            pair_name = f"frame pair {frames[previous_i].idx}->{frames[current_i].idx}"
            if n_matches < min_matches:
                failures.append(f"{pair_name}: {n_matches} ORB matches")
                continue

            K_prev, K_curr = _frame_K(frames[previous_i]), _frame_K(frames[current_i])
            same_zoom = abs(K_prev.fx - K_curr.fx) < 1e-6 and abs(K_prev.fy - K_curr.fy) < 1e-6
            if same_zoom:
                # Fast path, numerically identical to before this fix: one shared K.
                E, mask = cv2.findEssentialMat(points_prev, points_current, K_prev.K(),
                                               method=cv2.RANSAC, prob=0.999, threshold=1.0)
                pose_K = K_prev.K()
                match_pts_prev, match_pts_current = points_prev, points_current
            else:
                # Different focal length on each side (a zoom lens mid-clip) -- a single shared
                # cameraMatrix is wrong here. Normalize each side with ITS OWN K first (see
                # _normalize_points), then treat both as already-calibrated with an identity K.
                n_cross_zoom_pairs += 1
                match_pts_prev = _normalize_points(points_prev, K_prev)
                match_pts_current = _normalize_points(points_current, K_curr)
                pose_K = np.eye(3)
                # RANSAC's `threshold` is in the same units as the points; 1.0 px only makes
                # sense in pixel space, so rescale it into normalized-coordinate units using the
                # pair's mean focal length (fx is in px/normalized-unit).
                norm_threshold = 1.0 / ((K_prev.fx + K_curr.fx) / 2.0)
                E, mask = cv2.findEssentialMat(match_pts_prev, match_pts_current, pose_K,
                                               method=cv2.RANSAC, prob=0.999, threshold=norm_threshold)
            if E is None or mask is None:
                failures.append(f"{pair_name}: essential-matrix estimation failed")
                continue
            recovered, R, t, pose_mask = cv2.recoverPose(E, match_pts_prev, match_pts_current,
                                                          pose_K, mask=mask)
            n_inliers = int(recovered if pose_mask is not None else 0)
            if n_inliers < min_inliers:
                failures.append(f"{pair_name}: {n_inliers} pose inliers")
                continue
            step = float(np.linalg.norm(
                frames[current_i].pose.position - frames[previous_i].pose.position))
            if step < min_step_m:
                failures.append(f"{pair_name}: telemetry step {step:.3f} m")
                continue
            accepted_pair = (current_i, R, t, n_inliers, step)
            break

        if accepted_pair is None:
            details = "; ".join(failures) or "no candidate pair was evaluated"
            raise VisualOdometryError(
                f"could not find a valid next keyframe after frame {frames[previous_i].idx} "
                f"within --vo_max_frame_gap={max_frame_gap}: {details}. Lower --target_fps or "
                f"use a more textured/correctly focused clip.")

        current_i, R, t, n_inliers, step = accepted_pair
        poses.append(_compose_pose(poses[-1], R, t, step))
        accepted.append(current_i)
        pair_inliers.append(n_inliers)
        total_inliers += n_inliers
        telemetry_path_m += step
        previous_i = current_i

    return accepted, poses, VisualOdometryStats(
        n_frames_input=len(frames), n_frames=len(accepted),
        n_frames_skipped=len(frames) - len(accepted), n_pairs=len(accepted) - 1, total_matches=total_matches,
        total_inliers=total_inliers, median_inliers=float(np.median(pair_inliers)),
        telemetry_path_m=telemetry_path_m, n_cross_zoom_pairs=n_cross_zoom_pairs)


def replace_with_visual_odometry(dataset, **kwargs):
    """Return a dataset copy whose camera poses are recovered from its images."""
    accepted, poses, stats = visual_odometry_poses(dataset.frames, dataset.K, **kwargs)
    refined = [dataclasses.replace(dataset.frames[i], pose=pose) for i, pose in zip(accepted, poses)]
    return dataclasses.replace(dataset, frames=refined), stats
