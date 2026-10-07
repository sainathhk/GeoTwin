"""
dji_log_parser.py
REAL-DATA INGESTION -- STATUS: implemented, CPU-tested against synthetic mock CSVs
matching the real column schemas below (see tests/test_real_data.py). NOT yet
run against an actual downloaded DJI flight-record file (no network access to
Zenodo in this environment) -- validate against your real file in Colab.

Parses DJI flight-record CSVs into a plain `DroneLogSamples` array bundle.
Two known real-world export formats are auto-detected, column names taken
directly from the public DroneVideoMeasure project (github.com/egemose/
DroneVideoMeasure, src/dvm/drone/drone_log_data.py) -- the same tool this
dataset's own `DJIFlightRecord_....csv` files are formatted for. Using the
project's own confirmed column names beats guessing.

FORMAT A -- "TXTlogToCSVtool.exe" (what this dataset's DJIFlightRecord*.csv is):
    CUSTOM.updateTime, GIMBAL.yaw, GIMBAL.pitch, GIMBAL.roll,
    OSD.height [m], CUSTOM.isVideo, OSD.latitude, OSD.longitude

FORMAT B -- airdata.com export:
    time(millisecond), datetime(utc), latitude, longitude,
    height_above_takeoff(meters), isVideo, gimbal_heading(degrees),
    gimbal_pitch(degrees), gimbal_roll(degrees)

Both formats are commonly saved with stray null bytes and a non-UTF8
encoding (iso8859_10) -- handled here exactly as DroneVideoMeasure handles
them, since that's a real, previously-solved practical annoyance, not a
detail worth rediscovering by trial and error.
"""
from __future__ import annotations

import csv
import dataclasses
import re
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclasses.dataclass
class DroneLogSamples:
    t_sec: np.ndarray          # (N,) seconds since the FIRST log sample (not since video start)
    lat: np.ndarray
    lon: np.ndarray
    height_m: np.ndarray       # height above takeoff, meters
    yaw_deg: Optional[np.ndarray]   # gimbal/aircraft heading -- None if the format doesn't report one
                                     # at all (build_real_dataset then derives it from ground track)
    pitch_deg: np.ndarray      # gimbal pitch (negative = looking down, matches this repo's convention)
    roll_deg: np.ndarray
    is_video: np.ndarray       # bool -- True while the log says video was actively recording
    format_detected: str
    focal_len_mm: Optional[np.ndarray] = None  # per-sample 35mm-equiv focal length, when the format
                                                # reports it (a zoom lens changing focal length mid-clip
                                                # breaks the single-shared-K assumption everywhere else
                                                # in this repo -- see filter_stable_focal_length_frames)


def _strip_null_bytes(path: str) -> str:
    """DJI's own export tool sometimes embeds \\x00 bytes that break csv.DictReader.
    Returns a cleaned in-memory string rather than mutating the file on disk."""
    with open(path, "rb") as f:
        data = f.read()
    return data.replace(b"\x00", b"").decode("iso8859_10", errors="replace")


_FORMAT_A_COLUMNS = {
    "time": "CUSTOM.updateTime", "yaw": "GIMBAL.yaw", "pitch": "GIMBAL.pitch", "roll": "GIMBAL.roll",
    "height": "OSD.height [m]", "is_video": "CUSTOM.isVideo", "lat": "OSD.latitude", "lon": "OSD.longitude",
}
_FORMAT_B_COLUMNS = {
    "time": "datetime(utc)", "time_ms": "time(millisecond)", "yaw": "gimbal_heading(degrees)",
    "pitch": "gimbal_pitch(degrees)", "roll": "gimbal_roll(degrees)",
    "height": "height_above_takeoff(meters)", "is_video": "isVideo", "lat": "latitude", "lon": "longitude",
}


def _detect_format(fieldnames) -> Optional[dict]:
    fs = set(fieldnames or [])
    if all(c in fs for c in [_FORMAT_A_COLUMNS["time"], _FORMAT_A_COLUMNS["yaw"], _FORMAT_A_COLUMNS["lat"]]):
        return _FORMAT_A_COLUMNS
    if all(c in fs for c in [_FORMAT_B_COLUMNS["time"], _FORMAT_B_COLUMNS["yaw"], _FORMAT_B_COLUMNS["lat"]]):
        return _FORMAT_B_COLUMNS
    return None


def parse_dji_csv_log(path: str) -> DroneLogSamples:
    text = _strip_null_bytes(path)
    reader = csv.DictReader(text.splitlines())
    cols = _detect_format(reader.fieldnames)
    if cols is None:
        raise ValueError(
            f"Unrecognized DJI flight-log CSV format. Found columns: {reader.fieldnames[:12]}... "
            f"Expected either format A (CUSTOM.updateTime, GIMBAL.yaw, OSD.latitude, ...) or "
            f"format B (datetime(utc), gimbal_heading(degrees), latitude, ...). If this is a real "
            f"DJI export with different column names, pass a custom `column_map` -- see "
            f"docs/GETTING_STARTED.md 'Real data' section."
        )
    fmt_name = "A (TXTlogToCSVtool.exe)" if cols is _FORMAT_A_COLUMNS else "B (airdata.com)"

    rows = list(reader)
    t_raw, lat, lon, height, yaw, pitch, roll, is_video = [], [], [], [], [], [], [], []
    t0 = None
    for row in rows:
        try:
            if cols is _FORMAT_A_COLUMNS:
                try:
                    ts = datetime.strptime(row[cols["time"]], "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    ts = datetime.strptime(row[cols["time"]], "%Y/%m/%d %H:%M:%S")
                iv = row[cols["is_video"]] not in ("", "0", "False", "false")
            else:
                ts = datetime.strptime(row[cols["time"]], "%Y-%m-%d %H:%M:%S")
                iv = row[cols["is_video"]] == "1"

            lat_v, lon_v = float(row[cols["lat"]]), float(row[cols["lon"]])
            h_v = float(row[cols["height"]])
            yaw_v, pitch_v, roll_v = float(row[cols["yaw"]]), float(row[cols["pitch"]]), float(row[cols["roll"]])
        except (ValueError, KeyError):
            continue  # skip malformed/partial rows, same tolerance DroneVideoMeasure uses

        if t0 is None:
            t0 = ts
        t_raw.append((ts - t0).total_seconds())
        lat.append(lat_v); lon.append(lon_v); height.append(h_v)
        yaw.append(yaw_v); pitch.append(pitch_v); roll.append(roll_v); is_video.append(iv)

    if len(t_raw) < 2:
        raise ValueError(f"Parsed fewer than 2 valid rows from {path} using format {fmt_name} -- "
                          f"file may be corrupt, empty, or use yet another column naming variant.")

    return DroneLogSamples(t_sec=np.array(t_raw), lat=np.array(lat), lon=np.array(lon),
                            height_m=np.array(height), yaw_deg=np.array(yaw), pitch_deg=np.array(pitch),
                            roll_deg=np.array(roll), is_video=np.array(is_video, dtype=bool),
                            format_detected=fmt_name)


_SRT_TIMESTAMP_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->")
# GPS/altitude/heading line, e.g. "17.59942, 78.09021, 447.9m, 212\xb0" -- deliberately
# loose (just "two decimals, a third decimal followed by m, a fourth followed by the
# degree sign") rather than anchored to a fixed field count, since real DJI SRT exports
# vary a lot by drone model/firmware (many pack in iso/shutter/fnum/focal_len/etc. on
# the same or adjacent lines) -- this only looks for the four numbers this repo actually
# uses, wherever they sit in the block.
_SRT_GPS_RE = re.compile(r"(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.?\d*)\s*m,\s*(-?\d+\.?\d*)\s*\xb0")


def parse_dji_srt_log(path: str, assumed_gimbal_pitch_deg: float, assumed_gimbal_roll_deg: float = 0.0,
                       altitude_is_relative: bool = False) -> DroneLogSamples:
    """Parses a DJI-style .srt subtitle telemetry track (per-frame GPS/altitude/heading
    burned into the video's own subtitle stream) into the same DroneLogSamples shape
    `parse_dji_csv_log` produces, format_detected="C (SRT subtitle)".

    Two structural differences from the CSV formats that matter downstream, both
    handled here rather than left for the caller to trip over:

    1. NO GIMBAL PITCH/ROLL. This SRT variant (confirmed against an actual sample: 23
       blocks, fields are exactly lat, lon, "<alt>m", "<heading>\xb0" -- no pitch/roll at
       all) only gives aircraft/gimbal YAW. pitch_deg is filled with
       `assumed_gimbal_pitch_deg` for every sample -- there is NO safe default for this
       (it's the single most load-bearing value for camera pose, i.e. for everything
       the reconstruction does), so it's a REQUIRED argument, not a keyword default; a
       wrong guess here silently produces a plausible-looking but geometrically wrong
       reconstruction rather than an obvious failure. roll_deg defaults to 0 (a
       stabilized gimbal genuinely does hold roll near 0 by design, unlike pitch, which
       varies by shot -- this one IS safe to default).

    2. ALTITUDE IS AMBIGUOUS. The bare "<N>m" field could be height-above-takeoff (AGL,
       what `height_m` is documented as and what the rest of this pipeline assumes) or
       absolute MSL altitude -- this format doesn't label which, and DJI SRT exports do
       both depending on drone/firmware. altitude_is_relative=False (default) treats it
       as ambiguous/absolute and BASELINE-SUBTRACTS the first sample so height_m starts
       at 0, matching what an AGL trace would look like and what the rest of the
       pipeline (world Z used directly, no baseline subtraction elsewhere) already
       assumes. This preserves the flight's relative climb/descent profile correctly
       either way (a constant offset in Z is harmless for reconstruction, same as the
       lat/lon origin below) -- it only loses true absolute elevation, which nothing in
       this pipeline currently uses. Pass altitude_is_relative=True if you've confirmed
       the field is already AGL and want the raw values kept as-is.

    t_sec is read directly from the SRT timestamps, which are the video's OWN playback
    clock by construction -- no `_match_video_to_log` duration-matching guess needed;
    frame_times_log in build_real_dataset should be compared against this with zero
    offset. is_video is all-True for the same reason (every SRT block covers a
    real video interval).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    t_raw, lat, lon, alt, heading = [], [], [], [], []
    for block in re.split(r"\n\s*\n", text):
        ts_m = _SRT_TIMESTAMP_RE.search(block)
        gps_m = _SRT_GPS_RE.search(block)
        if not ts_m or not gps_m:
            continue  # blank trailing block, malformed entry, etc. -- same tolerance parse_dji_csv_log uses
        h, m, s, ms = (int(x) for x in ts_m.groups())
        t_raw.append(h * 3600 + m * 60 + s + ms / 1000.0)
        lat_v, lon_v, alt_v, heading_v = (float(x) for x in gps_m.groups())
        lat.append(lat_v); lon.append(lon_v); alt.append(alt_v); heading.append(heading_v)

    if len(t_raw) < 2:
        raise ValueError(f"Parsed fewer than 2 valid SRT telemetry blocks from {path} -- "
                          f"check it actually contains 'lat, lon, altm, heading\xb0' lines "
                          f"(some DJI SRT exports use different field layouts/units; if so this "
                          f"regex needs adjusting for your specific export, not just a retry).")

    alt = np.array(alt)
    if not altitude_is_relative:
        alt = alt - alt[0]

    n = len(t_raw)
    return DroneLogSamples(t_sec=np.array(t_raw), lat=np.array(lat), lon=np.array(lon), height_m=alt,
                            yaw_deg=np.array(heading), pitch_deg=np.full(n, assumed_gimbal_pitch_deg),
                            roll_deg=np.full(n, assumed_gimbal_roll_deg), is_video=np.ones(n, dtype=bool),
                            format_detected="C (SRT subtitle)")


def _is_frametelemetry_srt(text_head: str) -> bool:
    """Content sniff, not extension -- both this format and the Format C 'C (SRT
    subtitle)' one use .srt, so the file extension alone can't tell them apart."""
    return "FrameCnt:" in text_head


_D_FIELD_RES = {
    "lat": re.compile(r"latitude:\s*(-?\d+\.?\d*)"),
    "lon": re.compile(r"longitude:\s*(-?\d+\.?\d*)"),
    "rel_alt": re.compile(r"rel_alt:\s*(-?\d+\.?\d*)"),
    "focal_len": re.compile(r"focal_len:\s*(-?\d+\.?\d*)"),
}


def parse_dji_srt_frametelemetry_log(path: str, assumed_gimbal_pitch_deg: float,
                                      assumed_gimbal_roll_deg: float = 0.0) -> DroneLogSamples:
    """Parses the OTHER common DJI .srt telemetry style -- per-VIDEO-FRAME (not
    per-second) blocks shaped like:

        427
        00:00:14,213 --> 00:00:14,247
        <font size="28">FrameCnt: 427, DiffTime: 34ms
        2025-05-16 08:56:21.174
        [iso: 100] [shutter: 1/1000.0] [fnum: 1.7] [ev: -0.3] [color_md: default]
        [focal_len: 70.00] [latitude: 17.547721] [longitude: 78.209236]
        [rel_alt: 68.900 abs_alt: 646.400] [ct: 5608] </font>

    Confirmed against a real sample: dense (hundreds of blocks, ~30fps-spaced, not
    ~1/sec like Format C), and rel_alt is given directly (true AGL, no MSL-vs-AGL
    ambiguity to guess at the way Format C's bare "<N>m" needs).

    NO ORIENTATION AT ALL -- not even yaw/heading (Format C at least gave heading;
    this format gives none of yaw/pitch/roll). yaw_deg comes back None here;
    build_real_dataset derives it from the ground track (consecutive GPS deltas)
    when that happens -- a standard, defensible technique for a genuinely
    translating flight, NOT a substitute for real yaw on a flight that turns
    sharply between samples or barely translates at all (ground track is noise,
    not signal, when there's little translation to compute a track FROM). pitch_deg
    still has no source at all and is filled from the same required
    assumed_gimbal_pitch_deg build_real_dataset already demands for Format C.

    focal_len_mm is populated (Format C leaves it None) -- see
    filter_stable_focal_length_frames for why a changing value here needs
    handling before training, not just parsing.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if not _is_frametelemetry_srt(text[:2000]):
        raise ValueError(f"{path} doesn't look like FrameCnt-style DJI telemetry (no 'FrameCnt:' found "
                          f"in the first 2000 chars) -- if it's Format C ('lat, lon, altm, heading' "
                          f"per block), use parse_dji_srt_log instead.")

    t_raw, lat, lon, alt, focal = [], [], [], [], []
    for block in re.split(r"\n\s*\n", text):
        ts_m = _SRT_TIMESTAMP_RE.search(block)
        if not ts_m:
            continue
        field_vals = {}
        ok = True
        for name, rx in _D_FIELD_RES.items():
            m = rx.search(block)
            if not m:
                ok = False
                break
            field_vals[name] = float(m.group(1))
        if not ok:
            continue
        h, m_, s, ms = (int(x) for x in ts_m.groups())
        t_raw.append(h * 3600 + m_ * 60 + s + ms / 1000.0)
        lat.append(field_vals["lat"]); lon.append(field_vals["lon"])
        alt.append(field_vals["rel_alt"]); focal.append(field_vals["focal_len"])

    if len(t_raw) < 2:
        raise ValueError(f"Parsed fewer than 2 valid frame-telemetry blocks from {path} -- "
                          f"file may use yet another field layout than the one this was built against.")

    n = len(t_raw)
    return DroneLogSamples(t_sec=np.array(t_raw), lat=np.array(lat), lon=np.array(lon),
                            height_m=np.array(alt), yaw_deg=None,
                            pitch_deg=np.full(n, assumed_gimbal_pitch_deg),
                            roll_deg=np.full(n, assumed_gimbal_roll_deg),
                            is_video=np.ones(n, dtype=bool),
                            format_detected="D (SRT FrameCnt telemetry)", focal_len_mm=np.array(focal))


def filter_stable_focal_length_frames(dataset, max_drift_frac: float = 0.1):
    """A zoom lens changing focal length mid-clip breaks the single-shared-K
    (Intrinsics.from_fov computed ONCE for the whole dataset) assumption every
    other part of this repo makes -- adding real per-frame-intrinsics support is
    a bigger, riskier change than there was time to build and test carefully
    right now, so this takes the cheaper, safer route: find the longest
    contiguous run of frames whose focal length stays within max_drift_frac of
    that run's own first frame, and drop everything outside it. Confirmed useful
    on a real sample where focal_len drifts continuously from 24mm to 70mm over
    ~14s -- with max_drift_frac=0.1 this keeps only the near-24mm prefix, which
    is a real loss of usable frames/coverage, not a free fix; raising
    max_drift_frac trades that back for more residual intrinsics error in the
    frames it keeps. Only meaningful for frames carrying focal_len_mm (Format D);
    a no-op (returns dataset.frames unchanged, empty rejected list) otherwise.

    Returns (kept_frames, rejected_frames) -- same shape/calling convention as
    filter_downward_mapping_frames, for the same reason: train_gpu.py can log
    "Excluded N frames because ..." consistently regardless of which filter ran.
    """
    frames = dataset.frames
    focal = [getattr(f, "focal_len_mm", None) for f in frames]
    if not frames or any(v is None for v in focal):
        return frames, []

    focal = np.array(focal, dtype=float)
    best_start, best_len = 0, 1
    start = 0
    for i in range(1, len(focal) + 1):
        if i == len(focal) or abs(focal[i] - focal[start]) > max_drift_frac * focal[start]:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = i
    kept_idx = set(range(best_start, best_start + best_len))
    kept = [f for i, f in enumerate(frames) if i in kept_idx]
    rejected = [f for i, f in enumerate(frames) if i not in kept_idx]
    return kept, rejected


def find_video_recording_segments(log: DroneLogSamples) -> List[tuple]:
    """Contiguous (start_t_sec, end_t_sec) ranges where the log reports isVideo=True --
    used to line up which stretch of the log corresponds to a given .MOV file."""
    segments = []
    start = None
    for i, v in enumerate(log.is_video):
        if v and start is None:
            start = log.t_sec[i]
        if not v and start is not None:
            segments.append((start, log.t_sec[i]))
            start = None
    if start is not None:
        segments.append((start, log.t_sec[-1]))
    return segments
