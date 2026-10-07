"""
depth_estimation.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested
LAYER 2 INTERFACE: layer2_interfaces/i_depth_estimator.py

Classical plane-sweep multi-view stereo: for a reference frame and a small
set of neighboring frames (selected using the ESTIMATED, fused poses -- not
ground truth), sweep fronto-parallel depth hypotheses, warp each neighbor
into the reference view via the depth-induced homography, and score
photo-consistency. This produces both a depth map AND a per-pixel
"depth consistency" signal (cost-curve peakiness) that later feeds directly
into the confidence framework -- a flat, ambiguous cost curve means "this
pixel's depth is not really known", which is exactly the situation a
single-pass flight creates over poorly-covered regions.

GPU replacement (see interface file): a learned MVS cost-volume network
(e.g. an MVSNet/CasMVSNet-style architecture) regularized with 3-D CNNs,
or a monocular depth network fused across views -- both operate on the
same reference/neighbor-frame contract as this module.

2026-09 MESH QUALITY AUDIT FINDING (docs/MESH_QUALITY_AUDIT.md) -- READ THIS
IF YOU ARE TOUCHING depth_min/depth_max/n_depths:
`RealPipelineConfig`'s old defaults (n_depths=32, LINEAR spacing over
depth_min=3.0..depth_max=100.0 meters -- world units ARE meters, see
real_data/geo_utils.py) were traced as the root cause of a badly-shaped
mesh on a real checkpoint, well upstream of anything mesh_export.py does.
Two independent, confirmed problems:

  1. QUANTIZATION: `np.linspace(3, 100, 32)` puts every tested depth
     hypothesis ~3.1m apart, EVERYWHERE, including close range (a facade at
     10-20m gets the same ~3m depth resolution as something at 90m). Every
     pixel's depth snaps to one of only 32 values. Visualizing the resulting
     fused point cloud (before any Gaussian training, before any meshing)
     showed clean, regular horizontal banding -- the depth quantization,
     directly visible in 3D. Fix: more hypotheses (`n_depths`, now defaults
     to 128) and LOG spacing (`depth_spacing="log"`, now the default) so
     resolution is finer at the depths that actually need it, not wasted
     uniformly across the whole range.
  2. RANGE: the SAME checkpoint's trained Gaussian positions span >150m from
     their centroid -- comfortably past this scene's old depth_max=100m.
     Any true surface point farther than depth_max from a given camera
     CANNOT be represented; its cost volume has no correct hypothesis to
     land on, so photo-consistency (weakly, and sometimes wrongly) picks
     whichever tested plane happens to look least-bad, and errors pile up
     near the depth_max boundary. Aggregated over a moving camera along a
     single flight pass, this is consistent with the pyramid/frustum-shaped
     point cloud actually observed on that checkpoint. `check_depth_range_coverage`
     below is a new, automatic diagnostic for exactly this -- run it on your
     own footage; do not assume depth_max=100 (or any other specific number)
     is either right or wrong for a DIFFERENT flight without checking.

Both fixes are here; NEITHER was validated against real video by the person
making them (no camera poses or footage were available in the environment
this fix was written in -- see the audit doc). Re-run real_pipeline.py on
real footage and inspect the printed coverage diagnostic before trusting a
production run on it.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Union

import numpy as np
import cv2

from ..synthetic.camera_model import Intrinsics


@dataclasses.dataclass
class DepthEstimate:
    frame_idx: int
    depth: np.ndarray            # (H,W) float32, 0 = unresolved
    consistency: np.ndarray      # (H,W) float32 in [0,1], higher = sharper/more trustworthy cost curve
    min_cost: np.ndarray         # (H,W) float32, raw best photo-consistency cost (lower is better)
    n_neighbors_used: int
    # 2026-09 (docs/MESH_QUALITY_AUDIT.md): fraction of pixels whose cost-volume argmin
    # landed on either END of the depth sweep, i.e. whose true depth is probably OUTSIDE
    # [depth_min, depth_max] and whose reported depth is therefore arbitrary. A high value
    # here means the depth RANGE is wrong for this scene -- see plane_sweep_depth's
    # `boundary_reject`. Defaults to 0.0 so hand-built DepthEstimates in tests stay valid.
    boundary_fraction: float = 0.0


def _project(K: Intrinsics, R_wc: np.ndarray, pos: np.ndarray, world_pts: np.ndarray):
    rel = world_pts - pos
    cam = rel @ R_wc  # (...,3)
    z = cam[..., 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K.fx * cam[..., 0] / safe_z + K.cx
    v = K.fy * cam[..., 1] / safe_z + K.cy
    return u, v, z


def _zncc_cost(ref: np.ndarray, warped: np.ndarray, valid: np.ndarray, win: int = 5) -> np.ndarray:
    """Local zero-mean normalized cross-correlation cost (0=perfect match, 1=uncorrelated/anti-correlated).
    Computed via box filters (mean/variance/covariance) -- the standard efficient patch-NCC trick,
    far more discriminative than raw pixel differences for textured surfaces."""
    r = ref.astype(np.float32)
    w = np.where(valid, warped, 0.0).astype(np.float32)
    k = (win, win)
    mr = cv2.boxFilter(r, -1, k)
    mw = cv2.boxFilter(w, -1, k)
    mrr = cv2.boxFilter(r * r, -1, k)
    mww = cv2.boxFilter(w * w, -1, k)
    mrw = cv2.boxFilter(r * w, -1, k)
    cov = mrw - mr * mw
    var_r = np.clip(mrr - mr * mr, 1e-3, None)
    var_w = np.clip(mww - mw * mw, 1e-3, None)
    ncc = cov / np.sqrt(var_r * var_w)
    ncc = np.clip(ncc, -1.0, 1.0)
    cost = (1.0 - ncc) / 2.0
    return np.where(valid, cost, 4.0).astype(np.float32)


def plane_sweep_depth(ref_gray: np.ndarray, ref_pose, neighbor_grays: List[np.ndarray],
                       neighbor_poses: List, K: Intrinsics, depth_min: float = 8.0,
                       depth_max: float = 140.0, n_depths: int = 40,
                       match_window: int = 5, depth_spacing: str = "log",
                       boundary_reject: bool = False,
                       neighbor_Ks: Optional[List[Intrinsics]] = None) -> DepthEstimate:
    """
    K: the REFERENCE frame's own intrinsics (kept as the name every existing positional
        caller already uses) -- this is what un-projects ref_gray's pixels into rays.
    neighbor_Ks: 2026-09-21 fix. Each neighbor's OWN intrinsics, same order as
        neighbor_grays/neighbor_poses. None (default) reuses K for every neighbor, i.e.
        the original single-shared-K behavior -- correct whenever the whole sweep window
        came from one focal length, which used to be guaranteed (a zoom lens's frames
        never survived --max_focal_drift_frac's old 0.1 default). Now that
        --max_focal_drift_frac defaults to 1.0 and per-frame K is real (see
        real_dataset_builder.py), a reference frame and its temporal neighbors CAN have
        different focal lengths, and warping a neighbor into the reference view with the
        WRONG intrinsics corrupts that neighbor's entire contribution to the cost volume
        -- silently: the resulting depth still looks like an ordinary number, just a
        wrong one. This is not hypothetical -- it is exactly what produced the
        2026-09-21 regression (mean 79%/max 96.6% of pixels flagged "dynamic" by a filter
        that trusted this depth, and the post-training frustum check on the exported
        Gaussians failing outright: opening angle matched the camera's real FOV to
        within 4.6 deg, "the cloud IS the frustum"). Pass each neighbor's real K, as
        estimate_depth_for_sequence now does by default.

        DELIBERATELY DOES NOT SUBSET THIS FUNCTION'S OWN CONTRACT: ref_gray always gets
        unprojected with K (single value), because plane-sweep only ever has ONE
        reference frame per call -- only the neighbor side needed to become a list.
    depth_spacing: "log" (default, RECOMMENDED -- see module docstring's 2026-09
        audit note) or "linear" (the old behavior, kept only for direct
        before/after comparison on your own footage; do not use for a new run).
        Log spacing concentrates depth hypotheses at close range, where a fixed
        angular pixel error corresponds to a much smaller depth error than it
        does far away -- the standard reason stereo/MVS systems sample depth
        (or inverse depth) logarithmically rather than linearly.
    boundary_reject: if True, a pixel whose best-matching depth is the FIRST or
        LAST hypothesis in the sweep is reported as unresolved (depth 0) instead
        of being given that boundary depth. See the long comment at the rejection
        site below for the reasoning -- an endpoint argmin is not evidence that a
        surface was found there, and accepting those wholesale is what turns a
        depth map into a frustum-shaped point cloud.

        DEFAULTS TO FALSE, deliberately, and this is a measured decision rather
        than caution for its own sake. Boundary rejection cannot distinguish "my
        true depth is outside the range" from "my true depth is legitimately very
        near the edge of the range". On this project's own synthetic scene, whose
        ground-truth depth runs to 99.4 against a depth_max of 100.0, the far
        surfaces genuinely sit at the boundary: enabling rejection there discards
        ~25% of otherwise-correct pixels and drops resolved coverage from 98.7% to
        74.6%, breaking downstream evaluation. So it is opt-in.

        `DepthEstimate.boundary_fraction` is reported ALWAYS, whether or not
        rejection is enabled -- that is the part you actually want on by default,
        because it makes a wrong depth range announce itself as a number instead
        of silently producing plausible-looking garbage. Read that number first;
        turn this flag on (and/or widen the range) when it is high AND you have
        confirmed the scene really does extend past depth_max, which is the
        situation the 2026-09 audit found on real footage (horizon in frame,
        true depths in the hundreds-to-thousands of metres against depth_max=100).
    """
    H, W = ref_gray.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ray_x = (us - K.cx) / K.fx
    ray_y = (vs - K.cy) / K.fy
    ones = np.ones_like(ray_x)

    ref_f = ref_gray.astype(np.float32)
    neighbor_f = [g.astype(np.float32) for g in neighbor_grays]
    neighbor_Ks = list(neighbor_Ks) if neighbor_Ks is not None else [K] * len(neighbor_grays)
    if len(neighbor_Ks) != len(neighbor_grays):
        raise ValueError(f"neighbor_Ks has length {len(neighbor_Ks)}, expected "
                          f"{len(neighbor_grays)} (one per neighbor_grays/neighbor_poses)")

    if depth_spacing == "log":
        if depth_min <= 0:
            raise ValueError(f"depth_spacing='log' needs depth_min > 0, got {depth_min}")
        depths = np.geomspace(depth_min, depth_max, n_depths)
    elif depth_spacing == "linear":
        depths = np.linspace(depth_min, depth_max, n_depths)
    else:
        raise ValueError(f"depth_spacing must be 'log' or 'linear', got {depth_spacing!r}")
    cost_volume = np.full((n_depths, H, W), 4.0, dtype=np.float32)  # sentinel cost for "no valid neighbor"

    for di, d in enumerate(depths):
        cam_pts = np.stack([ray_x * d, ray_y * d, ones * d], axis=-1)   # (H,W,3) in ref camera frame
        world_pts = cam_pts @ ref_pose.R_cw() + ref_pose.position

        costs = []
        for gj, pose_j, K_j in zip(neighbor_f, neighbor_poses, neighbor_Ks):
            u_j, v_j, z_j = _project(K_j, pose_j.R_wc, pose_j.position, world_pts)
            valid = (z_j > 0.3) & (u_j >= 0) & (u_j <= W - 1) & (v_j >= 0) & (v_j <= H - 1)
            warped = cv2.remap(gj, u_j.astype(np.float32), v_j.astype(np.float32), cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            cost = _zncc_cost(ref_f, warped, valid, win=match_window)
            costs.append(cost)
        if costs:
            stacked = np.stack(costs, axis=0)
            valid_mask = stacked < 3.9
            n_valid = valid_mask.sum(axis=0)
            summed = np.where(valid_mask, stacked, 0.0).sum(axis=0)
            mean_cost = np.where(n_valid > 0, summed / np.maximum(n_valid, 1), 4.0)
            cost_volume[di] = mean_cost

    best_idx = np.argmin(cost_volume, axis=0)
    min_cost = np.take_along_axis(cost_volume, best_idx[None, :, :], axis=0)[0]
    depth_map = depths[best_idx].astype(np.float32)
    depth_map = np.where(min_cost < 3.9, depth_map, 0.0)

    # ---------------------------------------------------------------------------
    # BOUNDARY / OUT-OF-RANGE REJECTION (2026-09, docs/MESH_QUALITY_AUDIT.md)
    # ---------------------------------------------------------------------------
    # A pixel whose TRUE depth lies outside [depth_min, depth_max] has no correct
    # hypothesis in the cost volume at all. Its cost curve is then MONOTONIC across
    # the whole sweep, and `argmin` lands on whichever END of the range happens to
    # look least-bad -- an arbitrary answer that is nonetheless indistinguishable
    # from a real one downstream, because it comes with a perfectly ordinary-looking
    # depth value. Back-projecting a whole frame of those produces points spread
    # through the CAMERA FRUSTUM rather than on any surface.
    #
    # This is not hypothetical: it is precisely what happened on the real
    # 2026-09 checkpoint. The fused point cloud fitted a cone (R^2=0.995, apex
    # intercept ~0) whose opening angle matched the camera frustum to within a
    # degree in both axes (84.6 x 56.8 measured vs 84.0 x 53.7 actual) -- the
    # geometry was the viewing volume, not the scene. The footage was an oblique
    # aerial shot with the HORIZON IN FRAME, so most pixels' true depth was
    # hundreds to thousands of metres, against a sweep capped at depth_max=100.
    #
    # `boundary_reject` below refuses to report a depth whose argmin sits at
    # either end of the sweep. That is the cheap, local, always-correct version of
    # the check: an interior minimum means some tested depth genuinely beat its
    # neighbours on BOTH sides, which is the actual evidence that a surface was
    # localized. An endpoint minimum means only "nothing tested was better in the
    # one direction I could look", which is not evidence of anything.
    at_lower_boundary = best_idx == 0
    at_upper_boundary = best_idx == (n_depths - 1)
    boundary_hit = at_lower_boundary | at_upper_boundary
    if boundary_reject:
        depth_map = np.where(boundary_hit, 0.0, depth_map)

    # Second-best cost (excluding a small neighbourhood around the best index) as an
    # ambiguity signal: a sharp global minimum => confident; a flat cost curve => not.
    second_best = np.full((H, W), 4.0, dtype=np.float32)
    for di in range(n_depths):
        far_enough = np.abs(di - best_idx) > max(1, n_depths // 10)
        cand = np.where(far_enough, cost_volume[di], 4.0)
        second_best = np.minimum(second_best, cand)
    margin = np.clip(second_best - min_cost, 0, 1.0)
    consistency = np.clip(margin / 0.15, 0.0, 1.0)
    consistency = np.where(depth_map > 0, consistency, 0.0)

    boundary_fraction = float(boundary_hit.mean())
    return DepthEstimate(frame_idx=-1, depth=depth_map, consistency=consistency.astype(np.float32),
                          min_cost=min_cost.astype(np.float32), n_neighbors_used=len(neighbor_grays),
                          boundary_fraction=boundary_fraction)


def estimate_depth_for_sequence(gray_frames: List[np.ndarray], poses: List,
                                 K: Union[Intrinsics, List[Intrinsics]],
                                 window: int = 2, **kwargs) -> List[DepthEstimate]:
    """Runs plane-sweep MVS for every frame in `gray_frames`, using up to `window`
    neighbors on each side (already the estimated-pose-based frame order).

    K: a single Intrinsics shared by every frame (old behavior -- still correct if
        every frame really does share one focal length), OR a list of per-frame
        Intrinsics the same length as gray_frames. 2026-09-21: real_pipeline.py now
        always passes the per-frame list (each frame's own K when it has one, else
        dataset.K) -- see plane_sweep_depth's neighbor_Ks docstring for why a shared
        K silently corrupts the cost volume the moment a sweep window spans two
        different focal lengths, which --max_focal_drift_frac's new 1.0 default
        makes routine rather than impossible.
    """
    n = len(gray_frames)
    Ks = list(K) if isinstance(K, (list, tuple)) else [K] * n
    if len(Ks) != n:
        raise ValueError(f"K list has length {len(Ks)}, expected {n} (one per frame in gray_frames)")
    results = []
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        nb_idx = [j for j in range(lo, hi) if j != i]
        est = plane_sweep_depth(gray_frames[i], poses[i], [gray_frames[j] for j in nb_idx],
                                 [poses[j] for j in nb_idx], Ks[i],
                                 neighbor_Ks=[Ks[j] for j in nb_idx], **kwargs)
        est.frame_idx = i
        results.append(est)
    return results


@dataclasses.dataclass
class DepthRangeCoverageReport:
    frac_resolved: float          # fraction of all pixels with a valid (>0) depth
    frac_near_max_boundary: float # of RESOLVED pixels, fraction within the top 5% of [depth_min,depth_max]
    frac_near_min_boundary: float # of RESOLVED pixels, fraction within the bottom 5%
    likely_range_too_small: bool  # frac_near_max_boundary exceeds the warn threshold
    warning: str = ""


def check_depth_range_coverage(depth_estimates: List[DepthEstimate], depth_min: float, depth_max: float,
                                boundary_band_frac: float = 0.05, warn_above: float = 0.15) -> DepthRangeCoverageReport:
    """
    2026-09 addition (see module docstring). A real, automatic check for the
    failure mode that prompted it: if a meaningful fraction of resolved pixels
    land within `boundary_band_frac` of `depth_max` (default: top 5% of the
    tested range), that's consistent with real geometry sitting BEYOND
    depth_max being forced to alias onto the farthest testable plane, not with
    those pixels genuinely having a best-match depth near the boundary by
    coincidence. Call this right after `estimate_depth_for_sequence` (see
    real_pipeline.py) and read the printed warning before trusting the run --
    this only flags the SYMPTOM; whether depth_max needs to go up, camera
    height/framing needs to change, or the scene is just genuinely that big is
    a judgment call this function cannot make for you.
    """
    all_depths = np.concatenate([e.depth[e.depth > 0].ravel() for e in depth_estimates]) \
        if depth_estimates else np.zeros((0,))
    total_pixels = sum(e.depth.size for e in depth_estimates) if depth_estimates else 1
    frac_resolved = float(all_depths.size / max(total_pixels, 1))

    span = depth_max - depth_min
    hi_band = depth_max - boundary_band_frac * span
    lo_band = depth_min + boundary_band_frac * span
    frac_near_max = float((all_depths >= hi_band).mean()) if all_depths.size else 0.0
    frac_near_min = float((all_depths <= lo_band).mean()) if all_depths.size else 0.0

    likely_too_small = frac_near_max > warn_above
    warning = ""
    if likely_too_small:
        warning = (
            f"{frac_near_max*100:.1f}% of resolved depth pixels landed within the top "
            f"{boundary_band_frac*100:.0f}% of the tested range (>= {hi_band:.1f} of "
            f"depth_max={depth_max}). This is the same symptom found on the checkpoint that "
            f"prompted docs/MESH_QUALITY_AUDIT.md: real geometry farther than depth_max gets "
            f"forced onto the farthest testable plane instead of its true depth. Consider "
            f"raising depth_max (e.g. from a rough scene-scale estimate: flight altitude AGL "
            f"+ expected ground-plane extent) and re-running."
        )
    return DepthRangeCoverageReport(frac_resolved=frac_resolved, frac_near_max_boundary=frac_near_max,
                                     frac_near_min_boundary=frac_near_min,
                                     likely_range_too_small=likely_too_small, warning=warning)
