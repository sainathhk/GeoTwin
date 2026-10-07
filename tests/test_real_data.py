"""
Tests for layer1_cpu_sandbox/real_data/. Since no real DJI file is available in
this environment (no network access to Zenodo here), these tests build
SYNTHETIC MOCK video+log files that match the real column schemas/formats
exactly (schemas confirmed against the public DroneVideoMeasure project's own
parsing code, not guessed) and run the real ingestion + pipeline code against
them end to end. This proves the CODE works; it does not replace validating
against an actual downloaded file, which docs/GETTING_STARTED.md asks you to
do separately once you have one.
"""
import csv
import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_cpu_sandbox.real_data.dji_log_parser import parse_dji_csv_log, find_video_recording_segments
from layer1_cpu_sandbox.real_data.geo_utils import latlon_to_local_enu, haversine_distance_m
from layer1_cpu_sandbox.real_data.video_loader import extract_frames, probe_video
from layer1_cpu_sandbox.real_data.real_dataset_builder import build_real_dataset
from layer1_cpu_sandbox.real_data.real_pipeline import run_real_pipeline, evaluate_real_result, RealPipelineConfig


def _write_mock_format_a_csv(path, n=200, hz=10.0, video_start_row=20, video_len_rows=100,
                              lat0=55.470, lon0=10.323):
    t0 = datetime(2018, 7, 4, 11, 19, 31)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["CUSTOM.updateTime", "GIMBAL.yaw", "GIMBAL.pitch", "GIMBAL.roll",
                    "OSD.height [m]", "CUSTOM.isVideo", "OSD.latitude", "OSD.longitude"])
        for i in range(n):
            t = t0 + timedelta(seconds=i / hz)
            lat = lat0 + i * 2e-6      # ~0.22 m/sample northward drift
            lon = lon0 + i * 3e-6      # eastward drift
            yaw = 10.0 + 0.01 * i
            pitch = -35.0 + 2.0 * np.sin(i * 0.05)
            roll = 1.5 * np.sin(i * 0.1)
            height = 45.0 + 0.5 * np.sin(i * 0.02)
            is_video = 1 if video_start_row <= i < video_start_row + video_len_rows else 0
            w.writerow([t.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3], f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}",
                        f"{height:.2f}", is_video, f"{lat:.8f}", f"{lon:.8f}"])


def _write_mock_format_b_csv(path, n=200, hz=10.0, video_start_row=20, video_len_rows=100,
                              lat0=55.470, lon0=10.323):
    t0 = datetime(2018, 7, 4, 11, 19, 31)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time(millisecond)", "datetime(utc)", "latitude", "longitude",
                    "height_above_takeoff(meters)", "isVideo", "gimbal_heading(degrees)",
                    "gimbal_pitch(degrees)", "gimbal_roll(degrees)"])
        for i in range(n):
            t = t0 + timedelta(seconds=i / hz)
            lat = lat0 + i * 2e-6
            lon = lon0 + i * 3e-6
            yaw = 10.0 + 0.01 * i
            pitch = -35.0 + 2.0 * np.sin(i * 0.05)
            roll = 1.5 * np.sin(i * 0.1)
            height = 45.0 + 0.5 * np.sin(i * 0.02)
            is_video = 1 if video_start_row <= i < video_start_row + video_len_rows else 0
            w.writerow([int(i * 1000 / hz), t.strftime("%Y-%m-%d %H:%M:%S"), f"{lat:.8f}", f"{lon:.8f}",
                        f"{height:.2f}", is_video, f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}"])


def _write_mock_video(path, n_frames=100, fps=30.0, w=96, h=72):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    for i in range(n_frames):
        frame = np.roll(base, i % w, axis=1)
        writer.write(frame)
    writer.release()


# ------------------------------------------------------------- CSV parsing --

def test_parse_format_a():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log_a.csv")
        _write_mock_format_a_csv(path)
        log = parse_dji_csv_log(path)
        assert log.format_detected.startswith("A")
        assert log.lat.shape[0] == 200
        assert log.is_video.sum() == 100
        assert abs(log.pitch_deg.mean() - (-35.0)) < 3.0


def test_parse_format_b():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log_b.csv")
        _write_mock_format_b_csv(path)
        log = parse_dji_csv_log(path)
        assert log.format_detected.startswith("B")
        assert log.lat.shape[0] == 200
        assert log.is_video.sum() == 100


def test_find_video_segments():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log_a.csv")
        _write_mock_format_a_csv(path, video_start_row=20, video_len_rows=100, hz=10.0)
        log = parse_dji_csv_log(path)
        segs = find_video_recording_segments(log)
        assert len(segs) == 1
        start, end = segs[0]
        assert abs(start - 2.0) < 0.2   # row 20 at 10Hz -> t=2.0s
        assert abs((end - start) - 10.0) < 0.3  # 100 rows at 10Hz -> 10s


def test_unrecognized_format_raises_clear_error():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.csv")
        with open(path, "w") as f:
            f.write("foo,bar\n1,2\n")
        try:
            parse_dji_csv_log(path)
            assert False, "should have raised"
        except ValueError as e:
            assert "Unrecognized" in str(e)


# --------------------------------------------------------------- geo utils --

def test_latlon_to_local_enu_matches_haversine():
    lat0, lon0 = 55.470, 10.323
    lat1, lon1 = 55.471, 10.324
    x, y = latlon_to_local_enu(np.array([lat1]), np.array([lon1]), lat0, lon0)
    planar_dist = float(np.hypot(x[0], y[0]))
    great_circle_dist = haversine_distance_m(lat0, lon0, lat1, lon1)
    assert abs(planar_dist - great_circle_dist) / great_circle_dist < 0.01  # <1% at this scale


def test_latlon_origin_maps_to_zero():
    x, y = latlon_to_local_enu(np.array([55.470]), np.array([10.323]), 55.470, 10.323)
    assert abs(x[0]) < 1e-6 and abs(y[0]) < 1e-6


# -------------------------------------------------------------- video I/O --

def test_video_probe_and_extract():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vid.mp4")
        _write_mock_video(path, n_frames=90, fps=30.0)
        info = probe_video(path)
        assert info.frame_count == 90 and abs(info.fps - 30.0) < 0.5
        frames = extract_frames(path, target_fps=5.0, max_frames=50)
        assert 10 <= len(frames) <= 20
        assert frames[0][1].ndim == 3 and frames[0][1].shape[2] == 3


# --------------------------------------------------- end-to-end (mock data) --

def test_build_real_dataset_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        video_path = os.path.join(d, "flight.mov")
        log_path = os.path.join(d, "flight.csv")
        _write_mock_video(video_path, n_frames=100, fps=10.0)  # 10s clip
        _write_mock_format_a_csv(log_path, n=200, hz=10.0, video_start_row=20, video_len_rows=100)

        ds = build_real_dataset(video_path, log_path, target_fps=2.0, max_frames=20, resize_to=(64, 48))
        assert len(ds.frames) > 5
        assert ds.log_format_detected.startswith("A")
        assert ds.K_is_assumed is True
        # translation should be nonzero (mock log has steady lat/lon drift)
        xs = [f.pose.position[0] for f in ds.frames]
        assert (max(xs) - min(xs)) > 0.5


def test_real_pipeline_runs_end_to_end_on_mock_data():
    with tempfile.TemporaryDirectory() as d:
        video_path = os.path.join(d, "flight.mov")
        log_path = os.path.join(d, "flight.csv")
        _write_mock_video(video_path, n_frames=80, fps=10.0)
        _write_mock_format_a_csv(log_path, n=160, hz=10.0, video_start_row=10, video_len_rows=80)

        ds = build_real_dataset(video_path, log_path, target_fps=2.0, max_frames=12, resize_to=(48, 36))
        result = run_real_pipeline(ds, RealPipelineConfig(n_depths=12, depth_min=2.0, depth_max=60.0))
        assert result.efficiency["counts"]["n_frames_kept"] >= 3

        ev = evaluate_real_result(ds, result, n_self_consistency_views=2)
        assert len(ev.self_consistency_views) >= 1
        assert ev.n_final_points == result.final_positions.shape[0]
