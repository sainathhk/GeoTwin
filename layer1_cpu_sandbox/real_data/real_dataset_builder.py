"""
real_dataset_builder.py
REAL-DATA INGESTION -- STATUS: implemented, CPU-tested against synthetic mock
video+log pairs (tests/test_real_data.py). NOT yet run against a real
downloaded DJI file in this environment (no network access here) -- that
validation must happen in Colab, where you have the actual file.

Builds a `RealDroneDataset` -- the real-footage counterpart to
`synthetic.dataset_builder.SyntheticDroneDataset` -- from a video file +
matching DJI flight-record CSV. Deliberately mirrors the synthetic
dataset's pipeline-facing shape (frames, sensor-derived poses, K) so as
much of `pipeline.py`'s logic as possible is reusable unchanged; see
`real_pipeline.py` for the (small) adapter that's actually needed.

KEY DIFFERENCE FROM THE SYNTHETIC PATH: DJI flight-record CSVs report
GIMBAL yaw/pitch/roll directly -- that's the drone's own onboard sensor
fusion (multiple IMUs + gimbal encoders + often a compass), which is a
much higher-quality attitude estimate than anything this CPU sandbox's own
`sensor_fusion.fuse_gps_imu` complementary filter could reconstruct from
raw accelerometer/gyro alone. So for real data we USE that telemetry
directly as attitude ground truth, and only apply light smoothing to the
GPS position (a moving-average, not a full EKF -- there's no separate raw
accelerometer stream in this CSV format to fuse against). This is a
legitimate, different (simpler, and arguably more accurate) design choice
for real data, not a downgrade of the synthetic path's EKF.

NO GROUND TRUTH: unlike the synthetic dataset, there is no `gt_points`,
`control_measurements`, or per-triangle visibility trace here -- none of
that exists for real footage. `real_pipeline.py` and its evaluation
correspondingly skip every GT-dependent metric (geometry/metric-accuracy/
completeness/coverage) and fall back to what IS honestly checkable: does
the reconstruction reproduce what the camera actually saw (PSNR/SSIM/LPIPS
against the real captured frame itself), and does the confidence signal
still look internally sensible.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from .dji_log_parser import (parse_dji_csv_log, parse_dji_srt_log, parse_dji_srt_frametelemetry_log,
                              _is_frametelemetry_srt, find_video_recording_segments, DroneLogSamples)
from .geo_utils import latlon_to_local_enu
from .video_loader import extract_frames, probe_video, VideoInfo
from ..synthetic.camera_model import Intrinsics, CameraPose, rotmat_from_euler


@dataclasses.dataclass
class RealFrame:
    idx: int
    t: float            # seconds since VIDEO start (not log start)
    rgb: np.ndarray
    pose: CameraPose     # built from interpolated, smoothed log telemetry
    gps_lat: float
    gps_lon: float
    position_residual_m: float  # |raw GPS position - smoothed position| at this frame's time --
                                 # a real, computable noise proxy used as pose confidence downstream
    focal_len_mm: Optional[float] = None  # only set for Format D logs -- see
                                           # filter_stable_focal_length_frames
    K: Optional[Intrinsics] = None  # per-frame intrinsics for a zoom lens (see build_real_dataset);
                                     # None means "use dataset.K for this frame" -- every consumer
                                     # (build_cameras_from_real, visual_odometry.py) must fall back
                                     # to dataset.K when this is None, not assume it's always set.


@dataclasses.dataclass
class RealDroneDataset:
    frames: List[RealFrame]
    K: Intrinsics
    K_is_assumed: bool           # True unless a real calibration was supplied -- see calibrate.py
    origin_lat: float
    origin_lon: float
    video_info: VideoInfo
    log_format_detected: str
    matched_log_offset_s: float  # where in the log the video was found to start


def _smooth_moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or x.shape[0] < window:
        return x
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: x.shape[0]]


def _interp_angle_deg(t_query: np.ndarray, t_src: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Interpolates an angle correctly through the wraparound at +-180 deg by
    interpolating sin/cos components instead of the raw degrees."""
    rad = np.deg2rad(angle_deg)
    s = np.interp(t_query, t_src, np.sin(rad))
    c = np.interp(t_query, t_src, np.cos(rad))
    return np.rad2deg(np.arctan2(s, c))


def _match_video_to_log(log: DroneLogSamples, video_duration_s: float) -> float:
    """Returns the log-time offset (seconds since log start) at which the video
    most likely begins, using the isVideo flag's recorded segments and picking
    the one whose duration best matches the actual video file's duration --
    the same duration-matching fallback DroneVideoMeasure uses when precise
    GPS-tag video metadata isn't available (see docs/GETTING_STARTED.md)."""
    segments = find_video_recording_segments(log)
    if not segments:
        return float(log.t_sec[0])  # no isVideo flag data at all -- assume log starts with the video
    best = min(segments, key=lambda seg: abs((seg[1] - seg[0]) - video_duration_s))
    return float(best[0])


def build_real_dataset(video_path: str, log_path: str, camera_hfov_deg: float = 84.0,
                        target_fps: float = 2.0, max_frames: int = 80,
                        resize_to=(320, 240), position_smoothing_window: int = 5,
                        K: Optional[Intrinsics] = None,
                        assumed_gimbal_pitch_deg: Optional[float] = None,
                        assumed_gimbal_roll_deg: float = 0.0) -> RealDroneDataset:
    """
    camera_hfov_deg: ASSUMED horizontal field of view if `K` isn't supplied.
    84 deg is DJI's commonly-cited spec for the Phantom 4 Pro's camera --
    correct for that specific aircraft, an approximation for anything else.
    Get a real value via `real_data/calibrate.py` using the dataset's own
    calibration.MOV before trusting any metric/measurement output.

    log_path ending in .srt dispatches to parse_dji_srt_log instead of the
    CSV parser -- see that function's docstring for what's different about
    this format. assumed_gimbal_pitch_deg/assumed_gimbal_roll_deg are ONLY
    used in that case (the SRT format this was built against carries no
    per-frame gimbal attitude, only aircraft heading) and
    assumed_gimbal_pitch_deg has no default: it must be supplied explicitly
    when log_path is .srt, because guessing it wrong produces a plausible-
    looking but geometrically wrong reconstruction rather than an obvious
    failure -- exactly the class of mistake this codebase has already made
    once (see gaussian_model.py's densify_and_prune docstring on the
    camera-extent bug) and shouldn't repeat on something even more
    load-bearing.
    """
    is_srt = log_path.lower().endswith(".srt")
    if is_srt:
        if assumed_gimbal_pitch_deg is None:
            raise ValueError(
                "log_path is an .srt file, which carries no per-frame gimbal pitch -- "
                "assumed_gimbal_pitch_deg must be supplied explicitly (e.g. -90 for straight "
                "down, roughly -30 to -45 for a typical oblique/forward-looking shot, closer "
                "to 0 for near-horizontal). There is no safe default for this.")
        with open(log_path, "r", encoding="utf-8", errors="replace") as _f:
            is_frametelemetry = _is_frametelemetry_srt(_f.read(2000))
        if is_frametelemetry:
            log = parse_dji_srt_frametelemetry_log(log_path, assumed_gimbal_pitch_deg=assumed_gimbal_pitch_deg,
                                                     assumed_gimbal_roll_deg=assumed_gimbal_roll_deg)
        else:
            log = parse_dji_srt_log(log_path, assumed_gimbal_pitch_deg=assumed_gimbal_pitch_deg,
                                     assumed_gimbal_roll_deg=assumed_gimbal_roll_deg)
    else:
        log = parse_dji_csv_log(log_path)
    video_info = probe_video(video_path)
    raw_frames = extract_frames(video_path, target_fps=target_fps, max_frames=max_frames, resize_to=resize_to)
    if not raw_frames:
        raise ValueError(f"No frames extracted from {video_path} -- check the file opens and isn't empty.")

    # SRT timestamps ARE the video's own playback clock (that's what a subtitle track
    # is), so there's no separate-file sync offset to guess -- skip
    # _match_video_to_log's isVideo-segment-duration-matching entirely for this format
    # (it would also just fail: both SRT parsers' is_video is always True, so
    # find_video_recording_segments returns one segment spanning the whole log, and
    # matching that segment's duration against the video's is a needlessly indirect
    # way to arrive back at approximately zero anyway).
    offset = 0.0 if is_srt else _match_video_to_log(log, video_info.duration_s)
    frame_times_log = np.array([offset + t for (t, _) in raw_frames])

    lat0, lon0 = float(log.lat[0]), float(log.lon[0])
    x_all, y_all = latlon_to_local_enu(log.lat, log.lon, lat0, lon0)
    x_smooth = _smooth_moving_average(x_all, position_smoothing_window)
    y_smooth = _smooth_moving_average(y_all, position_smoothing_window)
    z_smooth = _smooth_moving_average(log.height_m, position_smoothing_window)

    if log.yaw_deg is None:
        # Format D gives no orientation at all, not even heading. Derive yaw from the
        # ground track (bearing between consecutive smoothed positions) -- a standard,
        # defensible technique for a flight that's genuinely translating, and in the
        # SAME compass-bearing convention (atan2 of east-delta, north-delta) as the
        # heading fields the OTHER formats report directly, so it's at least internally
        # consistent with how yaw is already used elsewhere here. It is NOT a substitute
        # for real yaw on a flight that turns between samples without much translating,
        # or barely translates at all -- ground track is noise, not signal, in that case.
        # Endpoint uses the last real bearing rather than 0 (an artificial hard stop).
        dx, dy = np.diff(x_smooth), np.diff(y_smooth)
        bearing = np.degrees(np.arctan2(dx, dy))  # 0=north, 90=east, matching compass heading fields
        yaw_all = np.concatenate([bearing, bearing[-1:]])
    else:
        yaw_all = log.yaw_deg

    x_i = np.interp(frame_times_log, log.t_sec, x_smooth)
    y_i = np.interp(frame_times_log, log.t_sec, y_smooth)
    z_i = np.interp(frame_times_log, log.t_sec, z_smooth)
    x_i_raw = np.interp(frame_times_log, log.t_sec, x_all)
    y_i_raw = np.interp(frame_times_log, log.t_sec, y_all)
    residual_m = np.hypot(x_i_raw - x_i, y_i_raw - y_i)
    yaw_i = _interp_angle_deg(frame_times_log, log.t_sec, yaw_all)
    pitch_i = _interp_angle_deg(frame_times_log, log.t_sec, log.pitch_deg)
    roll_i = _interp_angle_deg(frame_times_log, log.t_sec, log.roll_deg)
    lat_i = np.interp(frame_times_log, log.t_sec, log.lat)
    lon_i = np.interp(frame_times_log, log.t_sec, log.lon)

    if K is None:
        w, h = resize_to if resize_to else (video_info.width, video_info.height)
        K_final = Intrinsics.from_fov(w, h, hfov_deg=camera_hfov_deg)
        k_was_assumed = True
    else:
        K_final = K
        k_was_assumed = False

    focal_i = (np.interp(frame_times_log, log.t_sec, log.focal_len_mm)
               if log.focal_len_mm is not None else None)

    # Per-frame intrinsics for a zoom lens (2026-09 fix -- see filter_stable_focal_length_frames's
    # docstring, which explicitly deferred this: "adding real per-frame-intrinsics support is a
    # bigger, riskier change than there was time to build and test carefully right now"). Only
    # computed when the caller didn't hand us a real calibration (K is None -- an explicit
    # calibration is left exactly as given) and this log format actually carries focal_len_mm.
    #
    # camera_hfov_deg is only measured/assumed to be correct AT ONE focal length -- implicitly
    # frame 0's, since that's what a user pointing a known-FOV camera and reading off "84 deg"
    # is calibrating against. For a FIXED sensor, Intrinsics.from_fov's fx = (width/2)/tan(hfov/2)
    # is directly proportional to physical focal length (tan(hfov_physical/2) is itself inversely
    # proportional to focal length for a fixed sensor width), so scaling fx/fy by the ratio of
    # each frame's own focal_len_mm to the reference frame's is the correct extension of ONE
    # trusted HFOV to every zoom level, without needing to independently know the sensor's real
    # physical width. This does NOT model the DJI Air 3's wide/tele lenses being two physically
    # separate cameras (different optical center on the airframe) rather than one continuous
    # zoom -- see RUN_WITH_SRT_POSE_FIX.md / the chat that added this -- so frames deep into the
    # tele range still carry a small, unmodelled baseline error on top of the FOV correction.
    # Strictly better than the alternative (a single, badly-wrong HFOV, or dropping the frame
    # entirely), not a full multi-rig calibration.
    per_frame_K = None
    if K is None and focal_i is not None:
        focal_ref = float(focal_i[0])
        if focal_ref > 1e-6:
            per_frame_K = []
            for f in focal_i:
                ratio = float(f) / focal_ref
                per_frame_K.append(Intrinsics(width=K_final.width, height=K_final.height,
                                               fx=K_final.fx * ratio, fy=K_final.fy * ratio,
                                               cx=K_final.cx, cy=K_final.cy))

    frames = []
    for idx, (t, rgb) in enumerate(raw_frames):
        pos = np.array([x_i[idx], y_i[idx], z_i[idx]])
        pose = CameraPose.from_flight_attitude(pos, roll=np.deg2rad(roll_i[idx]),
                                                pitch=np.deg2rad(pitch_i[idx]), yaw=np.deg2rad(yaw_i[idx]))
        frames.append(RealFrame(idx=idx, t=t, rgb=rgb, pose=pose, gps_lat=float(lat_i[idx]),
                                 gps_lon=float(lon_i[idx]), position_residual_m=float(residual_m[idx]),
                                 focal_len_mm=float(focal_i[idx]) if focal_i is not None else None,
                                 K=per_frame_K[idx] if per_frame_K is not None else None))

    return RealDroneDataset(frames=frames, K=K_final, K_is_assumed=k_was_assumed, origin_lat=lat0, origin_lon=lon0,
                             video_info=video_info, log_format_detected=log.format_detected,
                             matched_log_offset_s=offset)
