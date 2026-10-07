"""
video_loader.py
REAL-DATA INGESTION -- STATUS: implemented, CPU-tested (writes and re-reads a
real synthetic .mp4 via OpenCV in tests/test_real_data.py -- actual video I/O,
not mocked).

Extracts frames from a real video file at a target sampling rate using
OpenCV's VideoCapture -- no ffmpeg binary dependency, portable to a plain
Colab runtime with zero extra apt-get installs.
"""
from __future__ import annotations
import dataclasses
from typing import List, Tuple

import cv2
import numpy as np


@dataclasses.dataclass
class VideoInfo:
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(fps=fps, frame_count=frame_count, duration_s=duration, width=width, height=height)


def extract_frames(path: str, target_fps: float = 2.0, max_frames: int = 120,
                    resize_to: Tuple[int, int] = None) -> List[Tuple[float, np.ndarray]]:
    """
    Returns [(t_seconds_since_video_start, rgb_frame), ...], sampled at
    `target_fps` (subsampling the source video, never upsampling it).
    `resize_to`: (width, height) -- real 4K/1080p footage should almost
    always be downsized before this repo's CPU-sandbox modules touch it
    (they were validated at ~160x120 for CPU speed; see
    docs/GETTING_STARTED.md for the resolution/accuracy tradeoff).
    """
    info = probe_video(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    src_fps = info.fps
    step = max(1, int(round(src_fps / max(target_fps, 1e-6))))

    frames = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if resize_to is not None:
                frame_bgr = cv2.resize(frame_bgr, resize_to, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t = idx / src_fps
            frames.append((t, rgb))
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames
