"""
vggt_pose_depth_source.py
LAYER 1 (CPU SANDBOX GLUE) -- bridges recon3d's VGGT output into this
repo's EXISTING confidence.py / prototype_point_repr.py / train_gpu.py
pipeline. Nothing downstream of `build_pose_depth_source()` changes.

WHY THIS BOUNDARY (replace {visual_odometry.py + depth_estimation.py},
keep {confidence.py, Gaussian training, mesh export} exactly as-is):

fuse_point_cloud() and compute_confidence() only need, per kept frame:
a depth map, a per-pixel consistency/confidence map, a CameraPose, an
Intrinsics, a frame-quality score, and a pose-confidence score. They do
not care whether those came from classical plane-sweep MVS + pairwise
VO or from a learned joint model -- so a genuinely novel confidence
FUSION framework (this project's actual thesis) does not have to be
rebuilt to fix an upstream pose/depth problem.

Mixing sources is the trap to avoid: VGGT jointly predicts pose, depth,
AND intrinsics as one mutually-consistent package. Pairing VGGT poses
with this repo's SRT-derived K, or with the classical plane-sweep depth,
reintroduces exactly the kind of pose/depth mismatch already fixed twice
in this repo's history (see docs/MESH_QUALITY_AUDIT.md,
docs/RUN_AFTER_ZOOM_AND_MESH_FIX.md). So: VGGT's pose + VGGT's depth +
VGGT's intrinsics travel together, as a package, into this repo's fusion.

METRIC SCALE: recon3d's own metric_align.py recovers ONE global scale
factor from MoGe-2 monocular metric depth on ~5 reference frames
(recon3d's README: "typically within 10-20% of ground truth"). This
project already has a better metric reference sitting right there:
real DJI GPS telemetry (dji_log_parser.py -> geo_utils.py ENU positions),
already trusted for visual_odometry.py's translation scale. This module
instead fits a similarity transform (Umeyama 1991: scale + rotation +
translation, closed-form via SVD) from EVERY VGGT-recovered camera
centre onto its telemetry position -- 15-20 correspondences on a typical
clip here, vs 5 for the MoGe-2 route, and a directly physically-grounded
reference instead of a second monocular network's own guess.

WHAT IS AND ISN'T VERIFIED HERE:
This sandbox has no GPU and no HuggingFace network access, so the VGGT
forward pass itself has NOT been run here -- that is the one genuinely
unverified piece, and the only part of this design that real footage
could still surprise you on. Everything else here (extrinsic-convention
conversion, the Umeyama solver, and the fact that the objects this
module builds are actually accepted by YOUR real fuse_point_cloud() and
compute_confidence()) is checked in test_vggt_integration.py against
synthetic VGGT-shaped arrays, runnable right now with no GPU. Run the
real thing on the Colab T4 already used earlier for this project --
see run_on_colab.py.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from ..synthetic.camera_model import CameraPose, Intrinsics
from ..reconstruction.depth_estimation import DepthEstimate


def umeyama_alignment(source_pts: np.ndarray, target_pts: np.ndarray):
    """Closed-form similarity transform (Umeyama, 1991): finds scale s,
    rotation R (3,3), translation t (3,) minimizing
        sum_i || target_i - (s * R @ source_i + t) ||^2
    Returns (scale, R, t) such that target ~= s * R @ source + t.
    Standard algorithm (SVD of the cross-covariance, reflection-corrected);
    no external dependency beyond numpy.
    """
    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)
    if source_pts.shape != target_pts.shape or source_pts.shape[1] != 3:
        raise ValueError(f"expected matching (N,3) arrays, got {source_pts.shape} vs {target_pts.shape}")
    n = source_pts.shape[0]
    if n < 3:
        raise ValueError(f"Umeyama alignment needs >= 3 point correspondences, got {n}")

    mu_s, mu_t = source_pts.mean(axis=0), target_pts.mean(axis=0)
    src_c, tgt_c = source_pts - mu_s, target_pts - mu_t

    cov = (tgt_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    var_src = (src_c ** 2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    t = mu_t - scale * (R @ mu_s)
    return scale, R, t


def vggt_extrinsic_to_camera_pose(extrinsic_4x4: np.ndarray) -> CameraPose:
    """VGGT/recon3d convention (pose_estimation.py): extrinsics are (4,4)
    WORLD-TO-CAMERA matrices, i.e. x_cam = R @ x_world + t. This repo's
    CameraPose (synthetic/camera_model.py) stores world-FROM-camera
    rotation + world position, so: R_wc = R.T, position = -R.T @ t.
    """
    R = extrinsic_4x4[:3, :3]
    t = extrinsic_4x4[:3, 3]
    R_wc = R.T
    position = -R_wc @ t
    return CameraPose(position=position, R_wc=R_wc)


def normalize_confidence(conf: np.ndarray) -> np.ndarray:
    """VGGT's depth_conf is an unbounded positive confidence (recon3d
    thresholds it directly against a raw cutoff like 1.5 -- see
    pose_estimation.py), not the [0,1] range DepthEstimate.consistency
    documents. Per-frame 1st/99th-percentile min-max scaling is a simple,
    defensible default. PRINT A HISTOGRAM ON YOUR REAL DATA FIRST -- if
    VGGT's confidence turns out heavily skewed on this footage (e.g.
    almost everything near the top of its range), a percentile-rank
    normalization would spread it out more usefully than this does.

    IMPORTANT: this is VGGT's learned per-pixel confidence, not an independently
    measured cross-view depth residual and not a calibrated probability of metric
    accuracy. It is placed in DepthEstimate.consistency only to fit the existing
    fusion interface. The independent multi-view evidence in this pipeline comes
    from fused observation count and view-angle spread. Validate/calibrate the
    learned confidence on held-out real geometry before making accuracy claims.
    """
    lo, hi = np.percentile(conf, [1, 99])
    if hi - lo < 1e-6:
        return np.full_like(conf, 0.5, dtype=np.float32)
    return np.clip((conf - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


@dataclasses.dataclass
class VGGTFrameResult:
    """Minimal stand-in for what this module needs from a VGGT run --
    matches recon3d.pose_estimation.PoseEstimationResult's fields for
    extrinsics/intrinsics/depth_maps/image_sizes, PLUS depth_confs, which
    that dataclass computes internally but does not return. See
    estimate_poses_vggt_with_conf() below, which is recon3d's own
    estimate_poses_vggt (pose_estimation.py) with exactly one change:
    keeping the per-pixel depth_conf map it already computes instead of
    discarding it after using it once to prune `point_cloud`.
    """
    extrinsics: np.ndarray        # (N,4,4) world-to-camera
    intrinsics: np.ndarray        # (N,3,3)
    depth_maps: List[np.ndarray]  # N x (H,W), VGGT's own relative-scale units
    depth_confs: List[np.ndarray]  # N x (H,W), unbounded positive
    image_sizes: List[tuple]      # N x (H,W)


def estimate_poses_vggt_with_conf(image_paths: List[str], device: str = "cuda",
                                   max_batch_frames: int = 40) -> VGGTFrameResult:
    """Adapted from recon3d/recon3d/pose_estimation.py's estimate_poses_vggt.
    The only change: keeps per-frame depth_conf (already computed there,
    just never returned) so build_pose_depth_source() below can turn it
    into DepthEstimate.consistency instead of a flat placeholder. If
    recon3d's upstream file changes, re-diff against this.

    NOT RUNNABLE IN THIS SANDBOX (needs a CUDA GPU + a HuggingFace
    download of facebook/VGGT-1B, neither available here) -- run on
    Colab, see run_on_colab.py.
    """
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    n_images = len(image_paths)
    if max_batch_frames < 1:
        raise ValueError(f"max_batch_frames must be positive, got {max_batch_frames}")
    if n_images > max_batch_frames:
        raise ValueError(
            f"This VGGT adapter needs one joint pass, but {n_images} images exceed "
            f"max_batch_frames={max_batch_frames}. Separate VGGT batches have independent "
            "world coordinate frames; concatenating them would silently corrupt fusion. "
            "Reduce/sparsely sample frames to fit one pass or add an explicit cross-window alignment stage."
        )
    all_extrinsics, all_intrinsics, all_depths, all_confs, all_sizes = [], [], [], [], []

    for batch_start in range(0, n_images, max_batch_frames):
        batch_paths = image_paths[batch_start:batch_start + max_batch_frames]
        images = load_and_preprocess_images(batch_paths).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            predictions = model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]

        ext_np = extrinsic.squeeze(0).cpu().numpy()
        extrinsics_4x4 = np.zeros((ext_np.shape[0], 4, 4))
        extrinsics_4x4[:, :3, :] = ext_np
        extrinsics_4x4[:, 3, 3] = 1.0
        all_extrinsics.append(extrinsics_4x4)
        all_intrinsics.append(intrinsic.squeeze(0).cpu().numpy())

        for i in range(len(batch_paths)):
            d = depth[0, i, :, :, 0].cpu().numpy()
            c = depth_conf.squeeze(0)[i].cpu().numpy()
            all_depths.append(d)
            all_confs.append(c)
            all_sizes.append(d.shape[:2])

        del predictions, images, depth, depth_conf
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return VGGTFrameResult(
        extrinsics=np.concatenate(all_extrinsics, axis=0),
        intrinsics=np.concatenate(all_intrinsics, axis=0),
        depth_maps=all_depths, depth_confs=all_confs, image_sizes=all_sizes)


def build_pose_depth_source(frames, vggt_result, verbose: bool = True):
    """
    frames: the SAME `kept` list of RealFrame objects already passed to
        visual_odometry_poses (real_dataset_builder.build_real_dataset
        output). Used only for .rgb (frame quality) and .pose.position
        (telemetry, for alignment) -- NOT for its VO-estimated pose.
    vggt_result: a VGGTFrameResult (or a recon3d PoseEstimationResult with
        a `.depth_confs` attribute bolted on) covering these SAME frames,
        IN THE SAME ORDER, from one un-chunked VGGT batch (this project's
        clips run well under recon3d's own single-batch capacity at this
        frame count -- no chunking/GTSAM merge-across-windows needed here).

    Returns (poses, depth_estimates, Ks, frame_quality_scores,
    pose_confidences) -- exactly fuse_point_cloud()'s and
    compute_confidence()'s existing input contract. Everything downstream
    (confidence gating, Gaussian seeding/training, mesh export, the
    frustum sanity check) runs completely unmodified.
    """
    from ..perception.frame_quality import compute_frame_quality

    n = len(frames)
    if vggt_result.extrinsics.shape[0] != n:
        raise ValueError(
            f"{vggt_result.extrinsics.shape[0]} VGGT frames vs {n} input frames -- "
            f"they must match 1:1, in order. Did VGGT drop or reorder any?")

    vggt_positions = np.stack(
        [vggt_extrinsic_to_camera_pose(vggt_result.extrinsics[i]).position for i in range(n)])
    telemetry_positions = np.stack([f.pose.position for f in frames])

    scale, R_align, t_align = umeyama_alignment(vggt_positions, telemetry_positions)
    aligned_vggt_positions = (scale * (R_align @ vggt_positions.T).T) + t_align
    residuals = np.linalg.norm(aligned_vggt_positions - telemetry_positions, axis=1)

    if verbose:
        worst = int(np.argmax(residuals))
        print(f"[vggt align] scale={scale:.4f}  telemetry residual after alignment: "
              f"mean={residuals.mean():.2f}m median={np.median(residuals):.2f}m "
              f"max={residuals.max():.2f}m (frame idx {frames[worst].idx}). "
              f"A max much bigger than the median means that one frame's VGGT pose "
              f"disagrees with telemetry a lot more than the rest -- worth eyeballing "
              f"that frame (motion blur, glare, near-featureless content) before fully "
              f"trusting it; its pose_confidence below is already downweighted for this.")

    conf_scale = max(float(np.median(residuals)), 0.5)  # metres; guards div-by-~0 on a near-perfect fit
    poses, Ks, depth_estimates, frame_quality_scores, pose_confidences = [], [], [], [], []
    for i, f in enumerate(frames):
        cam = vggt_extrinsic_to_camera_pose(vggt_result.extrinsics[i])
        poses.append(CameraPose(
            position=scale * (R_align @ cam.position) + t_align,
            R_wc=R_align @ cam.R_wc))  # rotation composes directly under a similarity transform

        fx, fy = float(vggt_result.intrinsics[i, 0, 0]), float(vggt_result.intrinsics[i, 1, 1])
        cx, cy = float(vggt_result.intrinsics[i, 0, 2]), float(vggt_result.intrinsics[i, 1, 2])
        h, w = vggt_result.image_sizes[i]
        Ks.append(Intrinsics(width=int(w), height=int(h), fx=fx, fy=fy, cx=cx, cy=cy))

        depth = vggt_result.depth_maps[i].astype(np.float32) * scale
        conf = normalize_confidence(vggt_result.depth_confs[i])
        depth_estimates.append(DepthEstimate(
            frame_idx=f.idx, depth=depth, consistency=conf,
            min_cost=np.zeros_like(depth), n_neighbors_used=n - 1, boundary_fraction=0.0))

        frame_quality_scores.append(compute_frame_quality(f.rgb).overall)
        pose_confidences.append(float(np.clip(1.0 - residuals[i] / (3.0 * conf_scale), 0.05, 1.0)))

    return poses, depth_estimates, Ks, frame_quality_scores, pose_confidences


def run_dynamic_filter_with_vggt(frames, poses, Ks, forward_flag_deg: float = 25.0):
    """Optional bonus, not required for the core fix: this repo's OWN
    dynamic_object_filter.detect_dynamic_pixels() takes exactly
    (gray_ref, gray_next, depth_ref, pose_ref, pose_next, K_ref, K_next).
    Its false-positive rate on the real run (54-78% of pixels flagged, vs
    a plausible few percent of real traffic) is consistent with the same
    depth/pose noise this whole module exists to fix -- swapping in VGGT's
    poses+depth here, with NO change to that module's own code, is the
    direct test of that theory. Wired here for convenience; not exercised
    by test_vggt_integration.py (needs consecutive real frames, not
    synthetic ones).
    """
    from ..perception.dynamic_object_filter import detect_dynamic_pixels
    import cv2
    results = []
    for i in range(len(frames) - 1):
        gray_ref = cv2.cvtColor(frames[i].rgb, cv2.COLOR_RGB2GRAY)
        gray_next = cv2.cvtColor(frames[i + 1].rgb, cv2.COLOR_RGB2GRAY)
        results.append(detect_dynamic_pixels(
            gray_ref, gray_next, depth_ref=None,  # fill in from build_pose_depth_source's depth_estimates
            pose_ref=poses[i], pose_next=poses[i + 1], K_ref=Ks[i], K_next=Ks[i + 1]))
    return results
