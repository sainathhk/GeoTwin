"""
geo_utils.py
REAL-DATA INGESTION -- STATUS: implemented, CPU-tested (see tests/test_real_data.py)

Converts GPS lat/lon into local East-North-Up meters, matching the world
frame convention used everywhere else in this repo (synthetic/camera_model.py:
X=east, Y=north, Z=up). Uses a local equirectangular tangent-plane
approximation around a chosen origin (the flight's first GPS fix by
default) -- accurate to well under 1% distortion for flights spanning up to
a few kilometers, which covers every drone-mapping scenario this project
targets. NOT a substitute for a proper geodetic/UTM library (e.g. `utm`,
`pyproj`) if you need true survey-grade absolute georeferencing across
larger areas -- swap this out for one of those if/when that matters.
"""
from __future__ import annotations
import numpy as np

_EARTH_RADIUS_M = 6371000.0
_METERS_PER_DEG_LAT = 110540.0  # ~constant; latitude arc-length varies <0.5% pole-to-equator


def latlon_to_local_enu(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    """Returns (x_east_m, y_north_m) arrays relative to (lat0, lon0)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    meters_per_deg_lon = _METERS_PER_DEG_LAT * np.cos(np.deg2rad(lat0))
    x_east = (lon - lon0) * meters_per_deg_lon
    y_north = (lat - lat0) * _METERS_PER_DEG_LAT
    return x_east, y_north


def haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    """Independent cross-check for latlon_to_local_enu -- great-circle distance,
    used only in tests to confirm the local-tangent-plane approximation agrees
    with a completely different (non-approximated-in-the-same-way) formula."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return float(2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a)))
