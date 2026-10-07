"""
gaussian_geometry.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested (tests/test_gaussian_geometry.py),
CPU-executed against a real checkpoint during the 2026-09 mesh-quality audit (see
docs/MESH_QUALITY_AUDIT.md) -- this is not a synthetic-only prototype, it has been run
against a real full_state_iter8000.pt with 150,000 Gaussians.

WHY THIS MODULE EXISTS: mesh_export.py's original `points_to_mesh` called
`pcd.estimate_normals()` -- generic local-PCA normal estimation over the raw
point positions, exactly like you'd run on an arbitrary, unstructured point
cloud from a laser scanner. That throws away information a converged 3D
Gaussian actually HAS and a bare point does not: an anisotropic scale (N,3)
and a rotation quaternion (N,4) that, together, describe an oriented
ellipsoid. A Gaussian that has converged onto a real surface during training
becomes flat (highly anisotropic) with its shortest axis aligned to the true
local surface normal -- that is a direct, physically-grounded estimate of
"which way does this surface face here", derived from the SAME optimization
that fit the Gaussian to multi-view photometric evidence, not re-estimated
from scratch by looking at where neighboring points happen to sit.

AUDIT FINDING (see docs/MESH_QUALITY_AUDIT.md for the full numbers): on the
2026-09 real checkpoint, 85.9% of Gaussians surviving the confidence/opacity
gate have anisotropy (longest/shortest scale ratio) > 5, i.e. are already
disk-like and carry a well-defined normal direction this way. Swapping the
normal SOURCE alone (covariance vs. local-PCA) did NOT, by itself, fix that
checkpoint's mesh (see the audit doc's ablation) -- the dominant problem
there was upstream, in the depth-estimation stage that seeds Gaussian
positions (see depth_estimation.py's module docstring). This module is still
worth using: it is strictly more information than throwing the covariance
away, it is free (no extra training or GPU pass), it makes
`orient_normals_consistent_tangent_plane`'s job easier (starting from
mostly-correct directions instead of re-deriving them), and on checkpoints
that do NOT have this project's specific depth-quantization problem it is
expected to matter more than it did here. Use it, but don't expect it alone
to fix a badly-seeded point cloud -- fix the seeding too.

QUATERNION CONVENTION: (w, x, y, z), matching gaussian_model.py's
`_quat_rotate` exactly. Getting this wrong silently produces a plausible-
looking but wrong rotation matrix (still a valid rotation, just not the
Gaussian's actual one) -- there is no shape-mismatch error to catch it, which
is exactly why `tests/test_gaussian_geometry.py` checks it against a
hand-rotated axis-aligned Gaussian rather than only checking output shapes.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class GaussianGeometry:
    normals: np.ndarray          # (N,3) unit vectors, direction of the shortest scale axis (sign arbitrary)
    anisotropy: np.ndarray       # (N,) longest/shortest axis ratio, >=1
    scale_mean: np.ndarray       # (N,) mean of the 3 per-axis scales ("how big is this Gaussian")
    scale_min: np.ndarray        # (N,) shortest axis -- ~0 for a well-converged surface-hugging Gaussian
    rotation_matrix: np.ndarray  # (N,3,3) full rotation, for callers that want more than just the normal


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """(N,4) quaternions in (w,x,y,z) order -> (N,3,3) rotation matrices. Vectorized.
    Normalizes internally (matches GaussianModel.get_rotation(); the RAW saved
    `rotation` parameter is NOT unit-norm -- see inspect_checkpoint findings in
    the audit doc, |q| ranged 0.36-2.25 on the real checkpoint)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(n, 1e-12, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    N = q.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2); R[:, 0, 1] = 2 * (x*y - w*z);       R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z);       R[:, 1, 1] = 1 - 2 * (x**2 + z**2); R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y);       R[:, 2, 1] = 2 * (y*z + w*x);       R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


def compute_gaussian_geometry(scale: np.ndarray, rotation_wxyz: np.ndarray) -> GaussianGeometry:
    """
    scale: (N,3) POSITIVE per-axis scale, i.e. already exp(log_scaling) -- not the raw
        log-parameter. rotation_wxyz: (N,4) raw or normalized quaternion, (w,x,y,z).

    Returns a GaussianGeometry with one normal per input Gaussian: the world-space
    direction of that Gaussian's SHORTEST axis (the flat direction, for a converged
    surface-hugging Gaussian). Sign is arbitrary (a covariance matrix has no notion of
    "outward") -- fix it with orient_normals_consistent_tangent_plane or a
    visibility-based method before feeding a Poisson-style reconstructor that needs
    globally consistent signs.
    """
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape[-1] != 3 or rotation_wxyz.shape[-1] != 4:
        raise ValueError(f"expected scale (N,3) and rotation (N,4), got {scale.shape} and {rotation_wxyz.shape}")
    R = quat_to_rotmat(rotation_wxyz)
    n = scale.shape[0]

    min_axis = np.argmin(scale, axis=1)
    normals = R[np.arange(n), :, min_axis]
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norm_len, 1e-12, None)

    scale_sorted = np.sort(scale, axis=1)
    scale_min = scale_sorted[:, 0]
    scale_max = scale_sorted[:, 2]
    anisotropy = scale_max / np.clip(scale_min, 1e-9, None)
    scale_mean = scale.mean(axis=1)

    return GaussianGeometry(normals=normals, anisotropy=anisotropy, scale_mean=scale_mean,
                             scale_min=scale_min, rotation_matrix=R)


def suggest_threshold_from_valley(values: np.ndarray, n_bins: int = 40, search_range=(0.02, 0.7)) -> float:
    """
    Finds a data-driven separation threshold for a [0,1]-ish score (confidence, opacity, ...)
    by histogramming `values` and returning the center of the emptiest bin within
    `search_range` -- i.e. the natural "valley" between a low-score cluster and a
    high-score cluster, if the distribution actually has one.

    WHY NOT JUST A FIXED PERCENTILE: a percentile (e.g. "drop the bottom 20%") assumes
    the fraction of bad primitives is roughly constant across checkpoints/scenes, which
    has no reason to be true. A histogram valley adapts to whatever THIS checkpoint's
    distribution actually looks like -- see docs/MESH_QUALITY_AUDIT.md, where this exact
    approach found a threshold (~0.55) at the real valley in the confidence histogram,
    vs. the old fixed default (0.15) which happened to be a complete no-op (100% of
    Gaussians cleared it) on that checkpoint.

    Peaks are searched for over the FULL [0,1] histogram, not just `search_range` --
    a real cluster (e.g. this project's real checkpoint has its largest confidence
    cluster around 0.85-0.95, see the audit doc) can legitimately sit anywhere, and
    restricting peak detection to `search_range` was an earlier bug in this function
    (it silently amputated exactly the cluster boundary needed to find the valley:
    caught by tests/test_gaussian_geometry.py, which is why that test exists rather
    than only checking output shape/type). `search_range` instead only bounds the
    final reported threshold, as a sanity backstop against a degenerate answer at
    the very edge of the distribution.

    Falls back to the 20th percentile if the (full-range) histogram has fewer than
    two real interior peaks -- i.e. the distribution is unimodal/monotonic, as this
    project's real checkpoint's OPACITY histogram turned out to be (see the audit
    doc) -- unlike its confidence histogram, which was clearly trimodal (this
    function uses the two TALLEST peaks, so a trimodal or noisier-than-strictly-
    bimodal histogram like that one still resolves to a sensible split). Always
    returns a threshold; never raises on an unclear distribution.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return float(np.percentile(values, 20)) if values.size else 0.0

    hist, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    # light 3-tap smoothing so single-bin sampling noise doesn't masquerade as a peak/valley
    smooth = np.convolve(hist.astype(np.float64), [0.25, 0.5, 0.25], mode="same")

    # a "peak" = a real, separated cluster: an interior local max with non-trivial mass
    # (>=5% of the tallest bin anywhere), not just boundary noise or a single stray bin.
    peak_floor = 0.05 * smooth.max()
    peaks = [i for i in range(1, n_bins - 1) if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]
             and smooth[i] >= peak_floor]
    if len(peaks) < 2:
        # unimodal / monotonic (e.g. this project's real OPACITY histogram) -- no
        # well-separated clusters to split apart, so a valley-threshold isn't meaningful.
        return float(np.percentile(values, 20))

    peaks_sorted_by_height = sorted(peaks, key=lambda i: smooth[i], reverse=True)
    p1, p2 = sorted(peaks_sorted_by_height[:2])  # the two tallest peaks, left-to-right
    between = np.arange(p1, p2 + 1)
    valley = between[np.argmin(smooth[between])]
    threshold = float(centers[valley])
    lo, hi = search_range
    return float(np.clip(threshold, lo, hi))


def floater_mask(geometry: GaussianGeometry, anisotropy_below: float = 3.0,
                  scale_above_percentile: float = 90.0) -> np.ndarray:
    """
    Flags Gaussians that look like haze/sky/background blobs rather than surface
    samples: roughly ISOTROPIC (no clear flat surface direction -- a converged
    surface point is almost never round) AND unusually LARGE (bigger than most
    of the scene's Gaussians). Either signal alone is common and fine (small
    round Gaussians are normal for fine detail; a few big flat ones are normal
    for e.g. a large flat rooftop); it's the COMBINATION this project's real
    checkpoint showed only ~1% of the time (see audit doc) but which, when
    present, is a reasonable, cheap thing to exclude before reconstruction.
    """
    scale_thresh = np.percentile(geometry.scale_mean, scale_above_percentile)
    return (geometry.anisotropy < anisotropy_below) & (geometry.scale_mean > scale_thresh)
