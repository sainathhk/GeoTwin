"""
frustum_check.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested (tests/test_frustum_check.py),
CPU-executed against the real 2026-09 checkpoint that motivated it (it is the tool
that produced that diagnosis -- see docs/MESH_QUALITY_AUDIT.md).

THE ONE CHECK THAT WOULD HAVE CAUGHT THE 2026-09 FAILURE ON DAY ONE.

A reconstruction can be photometrically excellent and geometrically worthless at
the same time, and nothing in a PSNR/SSIM/LPIPS report distinguishes the two. On
the checkpoint that prompted this module, 3DGS reached PSNR 30.0 / SSIM 0.908 on
a training view -- renders visually near-indistinguishable from the input frame --
while the underlying 3-D point set was not a scene at all. This happens because
photometric loss constrains only where a Gaussian PROJECTS, not where it SITS: a
Gaussian anywhere along a pixel's viewing ray renders to the same pixel. With
limited or forward-dominant camera motion, depth is then almost free to be wrong
(the shape-radiance ambiguity), and a depth-seeded pipeline whose depth estimates
were arbitrary will happily train to a beautiful render over nonsense geometry.

WHAT THIS DETECTS: whether a point cloud is shaped like a CAMERA FRUSTUM (a cone
with its apex at a viewpoint, opening at the camera's field of view) rather than
like a SCENE. Filling the frustum is the characteristic signature of depth
estimates that carry no real information -- every pixel back-projected to some
arbitrary depth within the tested range. The test is quantitative and cheap: fit
cross-sectional width against distance along the dominant axis and check (a) the
fit is linear with R^2 near 1, (b) the intercept is ~0 (a real apex), and (c) the
implied opening angle matches the camera's known FOV. On the real checkpoint all
three fired hard: R^2 = 0.995, intercept 1.86 units on a 130-unit axis, and
measured opening angles of 84.6 x 56.8 degrees against the camera's actual
84.0 x 53.7 -- matching the frustum to within about a degree in both axes.

RUN THIS BEFORE TRAINING (on the fused seed cloud) AND AFTER (on Gaussian
centres). Before is where it is cheap to act on; after tells you whether training
fixed or preserved the problem. It is not a substitute for looking at the cloud --
it is what makes "looks a bit cone-ish" into a number you can gate a pipeline on.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass
class FrustumCheckResult:
    is_frustum_shaped: bool
    r_squared: float                  # linearity of width-vs-distance (a cone => ~1.0)
    apex_intercept: float             # width at the apex (a true cone => ~0)
    apex_intercept_relative: float    # intercept as a fraction of the axis length
    implied_fov_deg_wide: float       # opening angle in the wider cross-axis
    implied_fov_deg_narrow: float     # opening angle in the narrower cross-axis
    axis: int                         # which world axis (0/1/2) the cone opens along
    apex_at_max: bool                 # True if the apex is at the axis MAXIMUM
    fov_match_deg: Optional[float]    # closest |measured - expected| FOV mismatch, if expected given
    n_points: int
    reasons: Tuple[str, ...] = ()

    def format_report(self) -> str:
        verdict = ("FRUSTUM-SHAPED -- this point cloud is probably the camera viewing volume, "
                    "NOT the scene" if self.is_frustum_shaped else
                    "no frustum signature detected")
        lines = [
            f"  verdict:                 {verdict}",
            f"  points analysed:         {self.n_points}",
            f"  cone axis:               world axis {self.axis} "
            f"(apex at {'max' if self.apex_at_max else 'min'})",
            f"  width-vs-distance R^2:   {self.r_squared:.4f}   (a cone gives ~1.0)",
            f"  apex intercept:          {self.apex_intercept:.3f} "
            f"({self.apex_intercept_relative*100:.1f}% of axis length; a true apex gives ~0)",
            f"  implied opening angles:  {self.implied_fov_deg_wide:.1f} deg (wide) x "
            f"{self.implied_fov_deg_narrow:.1f} deg (narrow)",
        ]
        if self.fov_match_deg is not None:
            lines.append(f"  vs expected camera FOV:  off by {self.fov_match_deg:.1f} deg "
                          f"(a close match is damning -- it means the cloud IS the frustum)")
        for r in self.reasons:
            lines.append(f"  * {r}")
        return "\n".join(lines)


def check_frustum_shape(positions: np.ndarray, expected_hfov_deg: Optional[float] = None,
                         expected_vfov_deg: Optional[float] = None, n_bands: int = 12,
                         band_half_width_frac: float = 0.015, percentile_clip: float = 2.0,
                         r2_threshold: float = 0.95, intercept_rel_threshold: float = 0.08,
                         fov_match_threshold_deg: float = 8.0) -> FrustumCheckResult:
    """
    positions: (N,3) world-space points -- a fused seed cloud, or trained Gaussian centres.
    expected_hfov_deg / expected_vfov_deg: the camera's real field of view, if known
        (for a pinhole camera, vfov = 2*atan(tan(hfov/2) * height/width)). Supplying these
        makes the check far stronger: a cloud that merely happens to taper is ambiguous, but
        one that tapers at EXACTLY the camera's opening angle is the frustum.
    r2_threshold / intercept_rel_threshold / fov_match_threshold_deg: the three criteria.
        `is_frustum_shaped` is set when the shape is convincingly conical (R^2 and intercept)
        AND -- when expected FOV was supplied -- the angles match. Without expected FOV, the
        conical criteria alone set the flag, which is a weaker claim; the report says so.

    Returns a FrustumCheckResult. Never raises on degenerate input; a cloud too small or
    too flat to analyse comes back with is_frustum_shaped=False and a reason explaining why.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {positions.shape}")
    n = positions.shape[0]
    if n < 200:
        return FrustumCheckResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, True, None, n,
                                   ("too few points (<200) to fit a cone",))

    extent = positions.max(0) - positions.min(0)
    if np.any(extent <= 0):
        return FrustumCheckResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, True, None, n,
                                   ("degenerate bounding box (zero extent on some axis)",))

    best = None
    for axis in range(3):
        cross = [a for a in range(3) if a != axis]
        coord = positions[:, axis]
        lo, hi = coord.min(), coord.max()
        span = hi - lo
        for apex_at_max in (True, False):
            dists, widths_a, widths_b = [], [], []
            for frac in np.linspace(0.10, 0.90, n_bands):
                centre = (hi - frac * span) if apex_at_max else (lo + frac * span)
                band = positions[np.abs(coord - centre) < span * band_half_width_frac]
                if len(band) < 40:
                    continue
                dist = (hi - centre) if apex_at_max else (centre - lo)
                wa = (np.percentile(band[:, cross[0]], 100 - percentile_clip)
                      - np.percentile(band[:, cross[0]], percentile_clip)) / 2
                wb = (np.percentile(band[:, cross[1]], 100 - percentile_clip)
                      - np.percentile(band[:, cross[1]], percentile_clip)) / 2
                dists.append(dist); widths_a.append(wa); widths_b.append(wb)
            if len(dists) < 5:
                continue
            d = np.array(dists)
            for widths in (np.array(widths_a), np.array(widths_b)):
                A = np.vstack([d, np.ones_like(d)]).T
                (slope, intercept), *_ = np.linalg.lstsq(A, widths, rcond=None)
                pred = A @ [slope, intercept]
                denom = ((widths - widths.mean()) ** 2).sum()
                r2 = 1 - ((widths - pred) ** 2).sum() / denom if denom > 0 else 0.0
                if slope <= 0:
                    continue  # narrowing away from the apex: not a cone opening this way
                score = r2 - abs(intercept) / max(span, 1e-9)
                if best is None or score > best["score"]:
                    slope_a = np.polyfit(d, np.array(widths_a), 1)[0]
                    slope_b = np.polyfit(d, np.array(widths_b), 1)[0]
                    ang_a = np.degrees(np.arctan(abs(slope_a))) * 2
                    ang_b = np.degrees(np.arctan(abs(slope_b))) * 2
                    best = dict(score=score, r2=float(r2), intercept=float(intercept), axis=axis,
                                 apex_at_max=apex_at_max, span=float(span),
                                 wide=float(max(ang_a, ang_b)), narrow=float(min(ang_a, ang_b)))

    if best is None:
        return FrustumCheckResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, True, None, n,
                                   ("could not fit a cone on any axis (no consistent widening)",))

    intercept_rel = abs(best["intercept"]) / max(best["span"], 1e-9)
    reasons = []
    conical = best["r2"] >= r2_threshold and intercept_rel <= intercept_rel_threshold
    if best["r2"] >= r2_threshold:
        reasons.append(f"cross-section widens LINEARLY with distance (R^2={best['r2']:.4f}) -- "
                        f"a scene does not do this; a viewing volume does")
    if intercept_rel <= intercept_rel_threshold:
        reasons.append(f"the taper extrapolates to a POINT ({intercept_rel*100:.1f}% of axis length) -- "
                        f"consistent with a real camera apex")

    fov_match = None
    if expected_hfov_deg is not None or expected_vfov_deg is not None:
        diffs = []
        for measured in (best["wide"], best["narrow"]):
            for expected in (expected_hfov_deg, expected_vfov_deg):
                if expected is not None:
                    diffs.append(abs(measured - expected))
        fov_match = float(min(diffs)) if diffs else None
        if fov_match is not None and fov_match <= fov_match_threshold_deg:
            reasons.append(f"opening angle matches the camera's real FOV to within {fov_match:.1f} deg -- "
                            f"this is the decisive signature: the cloud IS the frustum")

    if fov_match is not None:
        is_frustum = conical and fov_match <= fov_match_threshold_deg
        if conical and not (fov_match <= fov_match_threshold_deg):
            reasons.append("shape is conical but the angle does NOT match the camera FOV -- could be a "
                            "genuinely cone-like scene (a valley, a quarry); inspect before acting")
    else:
        is_frustum = conical
        if conical:
            reasons.append("NOTE: expected_hfov_deg/expected_vfov_deg were not supplied, so this is the "
                            "weaker conical-shape test only -- pass the camera FOV to confirm")

    return FrustumCheckResult(is_frustum_shaped=bool(is_frustum), r_squared=best["r2"],
                               apex_intercept=best["intercept"], apex_intercept_relative=float(intercept_rel),
                               implied_fov_deg_wide=best["wide"], implied_fov_deg_narrow=best["narrow"],
                               axis=best["axis"], apex_at_max=best["apex_at_max"], fov_match_deg=fov_match,
                               n_points=n, reasons=tuple(reasons))


def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    """Pinhole vertical FOV implied by a horizontal FOV and frame aspect -- convenience so
    callers can pass both angles to check_frustum_shape without re-deriving the trigonometry."""
    return float(np.degrees(2 * np.arctan(np.tan(np.radians(hfov_deg) / 2) * height / width)))
