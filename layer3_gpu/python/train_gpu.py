"""
train_gpu.py
LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, NOT GPU-executed
anywhere yet. Needs a CUDA device AND the external `diff-gaussian-rasterization`
package (see requirements-gpu.txt / colab_bootstrap.sh) -- neither available
in Environment A. This is real, complete training-loop code, not a stub; it
has been reviewed and syntax-checked, and every non-CUDA-specific piece it
depends on (GaussianModel, confidence_gaussian_model, confidence_propagation,
losses) has been CPU-executed and tested. Run this in Colab; treat its first
run as a real experiment, not a formality -- read what it prints.

WHAT THIS DOES:
  1. Loads a dataset -- either the synthetic CPU-sandbox generator (for a
     controlled first test) or `real_data`'s real-footage loader.
  2. Runs the SAME Layer-1 pipeline (frame quality -> depth -> point fusion
     -> observation confidence) already validated on CPU, to get an
     initial confidence-fused point cloud.
  3. Initializes one Gaussian per fused point (GaussianModel.
     initialize_from_fused_points) -- confidence carries over directly.
  4. Trains via the external, UNMODIFIED diff-gaussian-rasterization
     renderer (see docs/ARCHITECTURE.md for why this project does not fork
     it), with a frequency-weighted-L1 + MS-SSIM loss.
  5. Every `densify_interval` iterations: accumulates observation-confidence
     evidence from the views rendered since the last interval, propagates it
     (STCP-inspired temporal+spatial smoothing), computes an OSAD-inspired
     optimizer-struggle signal from Adam's own state, and calls
     GaussianModel's confidence-gated clone_and_split / prune_points --
     the three-signal design validated on CPU in
     tests/test_layer3_gaussian_model.py and
     tests/test_layer3_confidence_gaussian_cpu.py.
  6. Periodically saves a checkpoint point cloud and logs loss/Gaussian
     count/confidence breakdown, so a human can watch what's actually
     happening rather than trusting a final number.
  7. Once training finishes, exports the final Gaussian centers as a
     surface mesh (.obj) via layer1_cpu_sandbox/reconstruction/mesh_export.py
     -- an easy-to-share deliverable format, not a change to what's actually
     trained/rendered (see that module's docstring). Gated on opacity AND
     confidence (--mesh_min_opacity / --mesh_min_confidence) so a stray
     low-evidence Gaussian doesn't get to vote on the exported surface.
     Failure here (e.g. too few confident points survive) is caught and
     reported, never allowed to crash a run after everything else already
     finished -- use export_obj.py to retry with different settings, or
     against a different saved checkpoint, without retraining.

WHAT THIS DELIBERATELY DOES NOT DO (see docs/STOP_CONDITIONS.md): increase
SH degree progressively, opacity-reset intervals, or other 3DGS/FastGS
hyperparameter refinements not load-bearing for the confidence-gating
ablation this project's core claim depends on (docs/EXPERIMENT_PLAN.md,
Experiment 1) -- add those AFTER that experiment has a result, not before.

AGENT HOOKS (layer4_agents/, added after the above was already GPU-executed):
--use_agents wires in two optional agent decisions, OFF by default (nothing
above/below changes unless this flag is passed):
  a) A Frame Agent (same one layer4_agents/orchestrator.py already uses on
     Layer 1) decides keep_fraction/min_overall for the frame-selection step
     `main()` already runs before Gaussian training starts, instead of the
     dense-seeding config's fixed defaults.
  b) A NEW Training Control Agent watches held-out PSNR/SSIM/LPIPS at each
     checkpoint and may freeze further densification or stop training early
     -- see the `training_agent` parameter on `train()` below and
     docs/AGENTIC_ARCHITECTURE.md's "Training Control Agent" section for the
     full rationale, including why this is calibrated against this file's
     own documented iter-6500-peak-then-decline finding.
This addition is independent of, and layers cleanly on top of, the SRT-log /
mesh-export / quality-knob work above: --use_agents only ever chooses VALUES
for the SAME dense_seed_kwargs / TrainConfig fields (keep_fraction,
min_overall, densify_until_iter-via-effective_densify_until_iter) that
--source real/synthetic, the SRT gimbal flags, and the mesh-export flags all
already populate -- it never bypasses frame-orientation filtering, focal-
length-drift filtering, or mesh export, and none of those need to know an
agent is involved at all.
STATUS of this addition specifically: the decision logic itself
(schemas.TrainingControlDecision, mock_agents.MockTrainingControlAgent,
context_builders.build_training_agent_context) is CPU-tested with no GPU
involved at all. Its WIRING into the loop below is reviewed and
syntax-checked, exactly like the rest of this file was before its first
real GPU run -- but unlike the rest of this file, this specific addition
has not yet HAD that first real GPU run. Treat --use_agents the way you
originally treated this whole script: as a real experiment to watch, not
an assumed-correct feature.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from typing import List, Optional

import numpy as np
import torch

from gaussian_model import GaussianModel, GaussianTrainingConfig
from confidence_gaussian_model import GaussianConfidenceStats
from confidence_propagation import PropagationConfig, ObservationConfidencePropagator, optimizer_state_struggle_signal
from losses import combined_loss


# ------------------------------------------------------------- camera bridge --

@dataclasses.dataclass
class TrainingCamera:
    """Renderer-agnostic camera + its ground-truth image -- built once from
    either dataset type (see build_cameras_from_synthetic / _from_real below)
    so the training loop itself never needs to know which source it came from."""
    pose: object
    K: object
    gt_image: torch.Tensor
    frame_quality: float
    pose_confidence: float
    frame_idx: int = -1          # original video-frame index; never used for training


def _camera_to_rasterizer_matrices(cam: TrainingCamera, device: str,
                                   znear: float = 0.05, zfar: float = 500.0):
    """Return matrices in the layout required by diff-gaussian-rasterization.

    ``world_view`` and ``projection`` are ordinary column-vector matrices. The
    reference 3DGS rasterizer consumes their transposes, then composes them as
    ``viewmatrix @ projection_matrix``. Keeping this separate from the CUDA
    settings object makes this convention CPU-testable.
    """
    W, H = cam.K.width, cam.K.height
    fovx = 2 * math.atan(W / (2 * cam.K.fx))
    fovy = 2 * math.atan(H / (2 * cam.K.fy))
    tanfovx, tanfovy = math.tan(fovx / 2), math.tan(fovy / 2)

    R_cw = cam.pose.R_cw()
    t = -R_cw @ cam.pose.position
    world_view = torch.eye(4, device=device)
    world_view[:3, :3] = torch.from_numpy(R_cw).float().to(device)
    world_view[:3, 3] = torch.from_numpy(t).float().to(device)

    projection = torch.zeros(4, 4, device=device)
    projection[0, 0] = 1.0 / tanfovx
    projection[1, 1] = 1.0 / tanfovy
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)
    projection[3, 2] = 1.0

    viewmatrix = world_view.T
    projection_matrix = projection.T
    # (projection @ world_view).T, matching the official 3DGS camera bridge.
    # Do not use ``world_view @ projection``: that reverses the transform order.
    full_proj = viewmatrix @ projection_matrix
    return viewmatrix, full_proj, tanfovx, tanfovy


def _camera_to_rasterizer_settings(cam: TrainingCamera, device: str, bg_color=(0.0, 0.0, 0.0),
                                    sh_degree: int = 0, znear: float = 0.05, zfar: float = 500.0):
    """
    Converts this project's CameraPose+Intrinsics (synthetic/camera_model.py --
    used identically by both the synthetic and real-data paths) into the
    view/projection matrices `GaussianRasterizationSettings` expects. This
    matrix layout is dictated by the external rasterizer's public API (the
    same for every 3DGS-family project that calls it), not by any one
    implementation's design.
    """
    from diff_gaussian_rasterization import GaussianRasterizationSettings

    W, H = cam.K.width, cam.K.height
    viewmatrix, full_proj, tanfovx, tanfovy = _camera_to_rasterizer_matrices(
        cam, device, znear=znear, zfar=zfar)

    return GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tanfovx, tanfovy=tanfovy,
        bg=torch.tensor(bg_color, device=device), scale_modifier=1.0,
        viewmatrix=viewmatrix, projmatrix=full_proj,
        sh_degree=sh_degree, campos=torch.from_numpy(cam.pose.position).float().to(device),
        prefiltered=False, debug=False,
    )


def render_gaussian_view(model: GaussianModel, cam: TrainingCamera, device: str, bg_color=(0.0, 0.0, 0.0),
                          active_sh_degree: int = None):
    """
    Renders `model` from `cam` via the external rasterizer. Returns (rendered_rgb,
    screenspace_points (needs .grad after backward, for accumulate_view_gradient),
    visibility_filter (N,) bool, radii (N,)).

    `active_sh_degree`: if given (and <= model.cfg.sh_degree), tells the rasterizer to
    only evaluate this many SH bands, even though `model.get_features()` returns the
    full tensor allocated for `model.cfg.sh_degree`. This is what makes PROGRESSIVE SH
    growth possible (see train()'s `sh_degree_interval`) without touching the parameter
    shapes: the rasterizer reads the same coefficient tensor either way, it just stops
    reading further bands early in training. Standard technique in the reference 3DGS
    implementation, not something specific to this project.
    """
    from diff_gaussian_rasterization import GaussianRasterizer

    degree = model.cfg.sh_degree if active_sh_degree is None else min(active_sh_degree, model.cfg.sh_degree)
    settings = _camera_to_rasterizer_settings(cam, device, bg_color=bg_color, sh_degree=degree)
    rasterizer = GaussianRasterizer(raster_settings=settings)

    screenspace_points = torch.zeros_like(model.xyz, requires_grad=True, device=device)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    rendered, radii = rasterizer(
        means3D=model.xyz, means2D=screenspace_points, shs=model.get_features(),
        colors_precomp=None, opacities=model.get_opacity(), scales=model.get_scaling(),
        rotations=model.get_rotation(), cov3D_precomp=None,
    )
    visibility_filter = radii > 0
    return rendered, screenspace_points, visibility_filter, radii


def _per_gaussian_view_angle(model: GaussianModel, cam: TrainingCamera) -> torch.Tensor:
    """Angle between the camera's optical axis and the ray to each Gaussian's position --
    the same analytic quantity Layer 1's confidence.py derives from pixel location; here
    it's derived directly from 3-D position since every Gaussian's world position is known."""
    forward = torch.from_numpy(cam.pose.forward()).float().to(model.xyz.device)
    to_gaussian = model.xyz.detach() - torch.from_numpy(cam.pose.position).float().to(model.xyz.device)
    to_gaussian = torch.nn.functional.normalize(to_gaussian, dim=-1)
    cos_angle = (to_gaussian * forward).sum(dim=-1).clamp(-1, 1)
    return torch.acos(cos_angle)


@dataclasses.dataclass
class TrainConfig:
    max_iterations: int = 3000
    densify_interval: int = 200
    densify_from_iter: int = 300
    densify_until_iter: int = 2500
    opacity_thresh: float = 0.02
    confidence_prune_below: float = 0.12
    confidence_densify_above: float = 0.60
    min_densify_observations: int = 2
    max_gaussians: int = 150000
    grad_thresh: float = 0.0002
    struggle_thresh: float = 3.0
    split_scale_thresh: float = 0.01
    # Opt-in FastGS_QADS-style periodic opacity reset (see arguments/__init__.py's
    # opacity_reset_interval=3000 in that project). 0 = disabled, matching the
    # original design (this file's own docstring on WHAT THIS DELIBERATELY DOES
    # NOT DO). Added because confidence-gated pruning alone screens a Gaussian once,
    # at birth, and never re-checks it -- opacity reset forces every surviving
    # Gaussian to re-earn its opacity under the CURRENT loss gradient every N
    # iterations, which is what actually catches "coasting" Gaussians that got
    # summoned early and never justified themselves since. On the reef data this
    # did NOT beat plain training at matched iteration counts -- worth re-testing
    # on this new scene rather than assuming that result carries over, but don't
    # expect it to be a free win.
    opacity_reset_interval: int = 0
    opacity_reset_target: float = 0.05
    # Guard added after a run where opacity_reset_interval evenly divided
    # max_iterations -- a reset fired on the LAST iteration, so the final
    # saved/eval'd checkpoint was every Gaussian frozen mid-collapse with zero
    # recovery iterations, not a converged model. Never fire a reset this close
    # to the end of the run.
    opacity_reset_min_recovery_iters: int = 500
    # FastGS_QADS's / stock 3DGS's big_points_ws floater prune -- see
    # GaussianModel.densify_and_prune's docstring. 0.1 matches the reference.
    # Didn't visibly fix the reef data's floater streaks (wrong defect: those were
    # normal-sized but MISPLACED Gaussians, not oversized ones) -- kept on by
    # default anyway since it's cheap insurance against genuinely oversized
    # Gaussians, which this new scene (buildings/trees, very different depth
    # range per pixel than a flat reef) may produce more of.
    max_world_scale_frac: float = 0.1
    lambda_ssim: float = 0.2
    propagation_gamma: float = 0.85
    propagation_voxel_size: float = 1.0
    propagation_spatial_beta: float = 0.3
    # Was 0.15. Set to 0 (no held-out split -- train on every available camera)
    # per request, now that camera count is the scarce resource again. Means
    # checkpoint renders go back to being training-view fit, not a genuine
    # generalization check -- fine for a single held-out-free production run,
    # but you lose the "is this actually working" signal held-out views gave you
    # on the reef data. Pass --holdout_fraction 0.15 to bring it back if you
    # end up with enough cameras that a 3-view holdout doesn't starve training.
    holdout_fraction: float = 0.0
    sh_degree_interval: int = 1000
    log_every: int = 100
    checkpoint_every: int = 500
    n_eval_views: int = 3
    use_agents: bool = False  # see module docstring "AGENT HOOKS" -- off changes nothing below
    out_dir: str = "outputs/gpu_train_run1"
    # --- final .obj mesh export (see _export_mesh / mesh_export.py) ---
    export_mesh: bool = True
    # 2026-09 mesh quality audit (docs/MESH_QUALITY_AUDIT.md): a FIXED --mesh_min_confidence
    # turned out to be a complete no-op on the real checkpoint that prompted the audit (100% of
    # Gaussians cleared 0.15, because it had never been checked against a real confidence
    # distribution). "auto" (the new default for both) instead derives each threshold from THIS
    # checkpoint's own histogram valley (gaussian_geometry.suggest_threshold_from_valley) --
    # pass an explicit float to override. This is the exact export_obj.py behavior, now also
    # applied to the mesh _export_mesh writes automatically at the end of training, which
    # previously used the fixed numbers below and none of the audit's other fixes (covariance
    # normals, floater filtering, debris/component cleanup) -- see _export_mesh's docstring.
    mesh_min_opacity: "float | str" = "auto"
    mesh_min_confidence: "float | str" = "auto"
    mesh_method: str = "poisson"      # or "ball_pivoting" -- see mesh_export.points_to_mesh
    mesh_poisson_depth: int = 9
    mesh_density_trim_quantile: float = 0.03
    mesh_use_covariance_normals: bool = True
    mesh_floater_filter: bool = True
    mesh_floater_anisotropy_below: float = 3.0
    mesh_floater_scale_percentile: float = 90.0
    # 0 (the pre-audit default) means "no debris cleanup" -- this is what let the shipped
    # mesh_iter8000.obj ship with 1042 disconnected components, 97.7% of vertices in the
    # largest one (near-identical to the audit's own "before" numbers: 1049 / 97.8%), even
    # though the underlying fix already existed in mesh_export.py. 20 is a light default that
    # only removes obviously-tiny fragments; raise it (or set --mesh_keep_top_k_components) if
    # debris persists, see docs/MESH_QUALITY_AUDIT.md's "Tuning guide".
    mesh_min_component_vertices: int = 20
    mesh_keep_top_k_components: Optional[int] = None
    mesh_voxel_size: Optional[float] = None
    # Camera FOV/frame size for the POST-training frustum-shape sanity check (see
    # frustum_check.py) -- this now also runs on the trained, filtered Gaussians before the
    # mesh is written, not only on the pre-training seed cloud. Defaults match --camera_hfov_deg
    # / --width / --height; main() overwrites these from the actual run's args.
    mesh_camera_hfov_deg: float = 84.0
    mesh_frame_width: int = 640
    mesh_frame_height: int = 480


def _limit_densify_candidates(densify_mask: torch.Tensor, grad_accum: torch.Tensor,
                              observation_count: torch.Tensor, min_observations: int,
                              max_new: int) -> torch.Tensor:
    """Return only evidence-backed, highest-gradient densification candidates.

    A clone inherits its parent's confidence but has no independent image
    evidence.  Letting it immediately split again creates exponential growth
    from a few early observations.  Requiring fresh observations and a hard
    per-scene budget makes confidence a capacity-allocation signal rather than
    a self-reinforcing Gaussian-count multiplier.
    """
    eligible = densify_mask & (observation_count >= min_observations)
    if max_new <= 0 or not eligible.any():
        return torch.zeros_like(eligible)
    candidate_idx = torch.nonzero(eligible, as_tuple=True)[0]
    if candidate_idx.numel() <= max_new:
        return eligible
    best_local = torch.topk(grad_accum[candidate_idx], k=max_new, largest=True, sorted=False).indices
    kept = torch.zeros_like(eligible)
    kept[candidate_idx[best_local]] = True
    return kept


def split_train_holdout_cameras(cameras: List[TrainingCamera], holdout_fraction: float = 0.15):
    """
    Splits `cameras` into (train_cameras, held_out_cameras), holding out every
    Nth camera by index (spread evenly through the flight, not taken from one
    end) so held-out views aren't all from the same part of the trajectory.
    Falls back to returning the SAME list for both (with a printed warning)
    if there are too few cameras to hold any out meaningfully -- callers can
    detect the fallback via `held_out is train` (identity, not equality).

    Extracted as a standalone function (rather than left inline in `train()`)
    specifically so it's unit-testable without needing CUDA or the external
    rasterizer -- see tests/test_layer3_gaussian_model.py.
    """
    n_cam = len(cameras)
    if holdout_fraction <= 0:
        # Explicit opt-out, not the "too few cameras to bother" fallback below --
        # distinguished in the log so it's clear this was asked for, not silently
        # forced by a tiny camera count. Still returns via the same `held_out is
        # train` identity contract callers already rely on.
        print(f"Training on all {n_cam} cameras, no held-out split (holdout_fraction<=0) -- "
              f"checkpoint renders will be TRAINING views (memorization risk), not a genuine "
              f"generalization check, by request.")
        return cameras, cameras
    holdout_every = max(2, round(1.0 / max(holdout_fraction, 1e-6)))
    held_out_idx = set(range(0, n_cam, holdout_every))
    if len(held_out_idx) == 0 or len(held_out_idx) >= n_cam:
        held_out_idx = set()
    train_cameras = [c for i, c in enumerate(cameras) if i not in held_out_idx]
    held_out_cameras = [c for i, c in enumerate(cameras) if i in held_out_idx]
    if not held_out_cameras:
        print(f"WARNING: only {n_cam} cameras total -- holding any out would starve training, so "
              f"checkpoint renders will be TRAINING views (memorization risk), not a genuine "
              f"generalization check. Use more input frames if you want this to mean something.")
        held_out_cameras = train_cameras
    else:
        print(f"Training on {len(train_cameras)} cameras, holding out {len(held_out_cameras)} "
              f"for genuine novel-view checkpoint renders (never used in the training loop below).")
    return train_cameras, held_out_cameras


def apply_training_agent_at_checkpoint(training_agent, held_out_history: List[dict], best_held_out: dict,
                                         current_iter: int, max_iterations: int, checkpoint_every: int,
                                         effective_densify_until_iter: int, n_gaussians: int, max_gaussians: int,
                                         min_checkpoints_before_acting: int = 2):
    """One Training Control Agent decision at a single checkpoint, extracted as a
    standalone function specifically so it's unit-testable without CUDA -- same reason
    `split_train_holdout_cameras` and `_limit_densify_candidates` above are standalone
    functions rather than inlined in `train()`'s loop (see tests/test_layer3_gaussian_model.py
    and this project's own established pattern for why that matters here). `training_agent`
    is duck-typed (anything with `.decide(context) -> TrainingControlDecision`, see
    layer4_agents/interfaces.py's ITrainingControlAgent) rather than imported/type-checked
    here, so this module still does not require layer4_agents to be installed unless a
    caller actually passes a real agent object in.

    Returns (new_effective_densify_until_iter, should_early_stop, decision_record_or_None).
    `decision_record` is None -- no agent call made at all -- when there aren't yet
    `min_checkpoints_before_acting` checkpoints of history: a trend can't be judged from
    one data point, so this skips even calling the agent rather than spending a call
    (a real API call, for the LLM path) that would just have to say "too early" anyway.
    """
    if len(held_out_history) < min_checkpoints_before_acting:
        return effective_densify_until_iter, False, None

    from layer4_agents.context_builders import build_training_agent_context
    densifying_active = current_iter <= effective_densify_until_iter
    ctx = build_training_agent_context(
        held_out_history, best_held_out, current_iter=current_iter,
        max_iterations=max_iterations, checkpoint_every=checkpoint_every,
        densifying_active=densifying_active, n_gaussians=n_gaussians, max_gaussians=max_gaussians)
    decision = training_agent.decide(ctx)
    record = {"iter": current_iter, "action": decision.action, "reasoning": decision.reasoning,
               "agent_confidence": decision.agent_confidence, "warnings": decision.warnings}

    new_effective = effective_densify_until_iter
    should_stop = False
    if decision.action == "stop_densifying":
        new_effective = min(effective_densify_until_iter, current_iter)
    elif decision.action == "early_stop":
        should_stop = True

    return new_effective, should_stop, record


def train(model: GaussianModel, cameras: List[TrainingCamera], cfg: TrainConfig, device: str = "cuda",
          training_agent: Optional[object] = None):
    """`training_agent`: anything with a `.decide(context) -> TrainingControlDecision` method
    (layer4_agents.mock_agents.MockTrainingControlAgent or
    layer4_agents.llm_agents.LLMTrainingControlAgent -- see layer4_agents/interfaces.py's
    ITrainingControlAgent). Only consulted when `cfg.use_agents` is True AND this is not
    None; leaving it None (the default) runs exactly the code path this file had before
    the "AGENT HOOKS" addition described in the module docstring -- not typed against
    ITrainingControlAgent directly so importing this module never requires layer4_agents
    (or, transitively, the optional `anthropic` package) to be installed at all unless
    a caller actually passes one in.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    model.setup_optimizer()
    propagator = model.propagator
    propagator.cfg = PropagationConfig(gamma=cfg.propagation_gamma, voxel_size=cfg.propagation_voxel_size,
                                        spatial_beta=cfg.propagation_spatial_beta)

    # TRAIN / HELD-OUT SPLIT (added after the first two real runs, both of which only ever
    # rendered TRAINING views for the checkpoint PNGs -- a good-looking render there proves
    # the model fits pixels it directly optimized against, not that it reconstructed the
    # scene. See split_train_holdout_cameras() above; same discipline
    # evaluate_result.py's `_interpolated_novel_pose` already uses on the CPU-sandbox side.
    train_cameras, held_out_cameras = split_train_holdout_cameras(cameras, cfg.holdout_fraction)

    positions = torch.stack([torch.from_numpy(c.pose.position).float() for c in cameras])
    scene_extent = float((positions.max(dim=0).values - positions.min(dim=0).values).norm())

    history = []
    best_held_out = {
        "psnr": {"value": -float("inf"), "iter": None},
        "ssim": {"value": -float("inf"), "iter": None},
        "lpips": {"value": float("inf"), "iter": None},
    }
    # AGENT HOOKS state (see module docstring; all unused/inert when training_agent is None):
    # held_out_history is a proper per-checkpoint list (best_held_out above only ever kept
    # the single best value per metric, which can't show a TREND -- e.g. "3 checkpoints
    # with no improvement" needs to see more than one point).
    held_out_history: List[dict] = []
    agent_decisions: List[dict] = []
    effective_densify_until_iter = cfg.densify_until_iter  # may only be LOWERED by an agent, never raised
    should_early_stop = False
    MIN_CHECKPOINTS_BEFORE_AGENT_ACTS = 2  # don't let even a real LLM agent judge a trend off 1 point
    t0 = time.perf_counter()
    for it in range(1, cfg.max_iterations + 1):
        cam = train_cameras[np.random.randint(len(train_cameras))]
        active_sh_degree = min(model.cfg.sh_degree, it // max(cfg.sh_degree_interval, 1))
        rendered, screenspace, vis_mask, radii = render_gaussian_view(model, cam, device,
                                                                       active_sh_degree=active_sh_degree)

        loss_dict = combined_loss(rendered, cam.gt_image.to(device), lambda_ssim=cfg.lambda_ssim)
        model.optimizer.zero_grad()
        loss_dict["total"].backward()

        with torch.no_grad():
            model.accumulate_view_gradient(screenspace, vis_mask)
            if vis_mask.any():
                view_angle = _per_gaussian_view_angle(model, cam)[vis_mask]
                # NOTE (see module docstring): consistency here is a per-VIEW scalar
                # (1 - mean L1 residual for this render), not the finer per-pixel/
                # per-Gaussian signal Layer 1's plane-sweep MVS derives -- a true
                # per-Gaussian consistency would need pixel-to-Gaussian attribution
                # inside the rasterizer's forward pass (the metric_map-style mechanism
                # discussed for FastGS-QADS), which this project deliberately does not
                # fork the rasterizer to obtain (see docs/ARCHITECTURE.md).
                consistency_this_view = float((1.0 - loss_dict["l1"]).clamp(0, 1))
                consistency = torch.full_like(view_angle, consistency_this_view)
                touched_idx = torch.nonzero(vis_mask, as_tuple=True)[0]
                model.confidence_stats.update(touched_idx, view_angle, consistency,
                                               cam.frame_quality, cam.pose_confidence)

        model.optimizer.step()

        if cfg.densify_from_iter <= it <= effective_densify_until_iter and it % cfg.densify_interval == 0:
            with torch.no_grad():
                raw_conf = model.confidence()
                propagated = propagator.update(model.xyz.detach(), raw_conf)
                struggle = optimizer_state_struggle_signal(model.optimizer, model.xyz)
                grad_accum = model.mean_view_gradient()

            # NOT scene_extent (that's camera-trajectory spread -- fine for
            # clone_and_split's split-vs-clone threshold below, but for a
            # narrow-baseline flight the trajectory can be much smaller than the
            # ground/scene footprint it's actually photographing, which on the
            # reef data fed the floater check a threshold smaller than the
            # Gaussians' own seeded scale and pruned the entire model to 0 on
            # the first pass). Use the actual reconstructed point cloud's own
            # extent instead.
            with torch.no_grad():
                cloud_extent = float((model.xyz.detach().max(dim=0).values
                                       - model.xyz.detach().min(dim=0).values).norm())
            prune_mask, densify_mask, _ = model.densify_and_prune(
                grad_accum, opacity_thresh=cfg.opacity_thresh, confidence_prune_below=cfg.confidence_prune_below,
                grad_thresh=cfg.grad_thresh, confidence_densify_above=cfg.confidence_densify_above,
                propagated_confidence=propagated, optimizer_struggle_signal=struggle,
                struggle_thresh=cfg.struggle_thresh,
                extent=cloud_extent, max_world_scale_frac=cfg.max_world_scale_frac)

            # A split replaces one parent with two children and a clone appends
            # one child, so every selected candidate adds exactly one Gaussian
            # after pruning.  Limit growth *before* structural changes so 16
            # training views cannot balloon into a million weakly constrained
            # primitives.
            n_after_prune = model.n_gaussians - int(prune_mask.sum().item())
            growth_budget = max(0, cfg.max_gaussians - n_after_prune)
            densify_mask = _limit_densify_candidates(
                densify_mask, grad_accum, model.confidence_stats.obs_count,
                cfg.min_densify_observations, growth_budget)

            # ORDER MATTERS: prune_mask and densify_mask are computed over the SAME
            # pre-change index space and are disjoint by construction (densify_and_prune
            # never marks a Gaussian both). Pruning FIRST, then re-indexing densify_mask
            # through the identical keep_mask, keeps both masks correctly aligned to the
            # array as it actually changes size -- doing it in the other order (as an
            # earlier draft of this loop did) requires re-deriving which post-clone_and_split
            # rows correspond to which pre-change rows, which clone_and_split's internal
            # split-parent removal makes non-trivial to get right from outside the call.
            keep_mask = ~prune_mask
            densify_mask_after_prune = densify_mask[keep_mask]
            if prune_mask.any():
                model.prune_points(prune_mask)
            if model.n_gaussians == 0:
                raise RuntimeError(
                    "All Gaussians were pruned -- densify/prune thresholds are miscalibrated for this "
                    "scene (see the [densify] line printed just above this for which check did it). "
                    "Stopping cleanly here instead of letting the rasterizer crash on an empty model "
                    "with an opaque CUDA gradient-shape error a few iterations later.")
            model.clone_and_split(densify_mask_after_prune, split_scale_thresh=cfg.split_scale_thresh,
                                   extent=scene_extent)
            if it % cfg.log_every == 0:
                print(f"      [densify] post-prune={n_after_prune} growth_budget={growth_budget} "
                      f"selected={int(densify_mask.sum())} cap={cfg.max_gaussians}")
            model._grad_accum.zero_()
            model._grad_denom.zero_()

        if (cfg.opacity_reset_interval > 0 and it % cfg.opacity_reset_interval == 0
                and it <= cfg.max_iterations - cfg.opacity_reset_min_recovery_iters):
            model.reset_opacity(target=cfg.opacity_reset_target)
            if it % cfg.log_every == 0:
                print(f"      [opacity_reset] clamped all {model.n_gaussians} Gaussians to "
                      f"<={cfg.opacity_reset_target}; next few hundred iters will show which "
                      f"ones climb back vs fade -- the fading ones are the coasting ones.")

        if it % cfg.log_every == 0 or it == 1:
            elapsed = time.perf_counter() - t0
            entry = {"iter": it, "n_gaussians": model.n_gaussians, "loss_total": loss_dict["total"].item(),
                     "loss_l1": loss_dict["l1"].item(), "loss_ssim": loss_dict["ssim_loss"].item(),
                     "elapsed_s": round(elapsed, 1)}
            history.append(entry)
            running = [h["loss_total"] for h in history[-10:]]
            running_mean = sum(running) / len(running)
            print(f"iter {it:5d}  n_gaussians={model.n_gaussians:6d}  "
                  f"loss={entry['loss_total']:.4f} (running-10={running_mean:.4f}, "
                  f"l1={entry['loss_l1']:.4f} ssim={entry['loss_ssim']:.4f})  {elapsed:.1f}s")

        if it % cfg.checkpoint_every == 0 or it == cfg.max_iterations:
            _save_checkpoint(model, cfg.out_dir, it)
            _save_full_state(model, cfg.out_dir, it)
            is_ho = (held_out_cameras is not train_cameras)
            ckpt_psnr, ckpt_ssim, ckpt_lpips = _render_and_save_views(
                model, held_out_cameras, cfg.out_dir, it, device, n_views=cfg.n_eval_views,
                is_held_out=is_ho, active_sh_degree=active_sh_degree)
            # Smooth renders can win SSIM while losing perceptual detail. Keep
            # independent winners so a single metric never silently decides
            # what “best reconstruction” means for a single-pass scene.
            if is_ho:
                if ckpt_psnr > best_held_out["psnr"]["value"]:
                    best_held_out["psnr"] = {"value": ckpt_psnr, "iter": it}
                if ckpt_ssim > best_held_out["ssim"]["value"]:
                    best_held_out["ssim"] = {"value": ckpt_ssim, "iter": it}
                if not math.isnan(ckpt_lpips) and ckpt_lpips < best_held_out["lpips"]["value"]:
                    best_held_out["lpips"] = {"value": ckpt_lpips, "iter": it}

                # AGENT HOOKS (see module docstring; inert unless cfg.use_agents and
                # training_agent is set). held_out_history is appended regardless of
                # cfg.use_agents -- it's cheap bookkeeping and means turning the flag on
                # mid-run-analysis later never requires having planned for it up front.
                # The decision itself is delegated to a standalone function (see
                # `apply_training_agent_at_checkpoint` above `train()`) specifically so it
                # is unit-testable without CUDA -- same reason `split_train_holdout_cameras`
                # and `_limit_densify_candidates` are standalone functions rather than
                # inlined here.
                held_out_history.append({"iter": it, "psnr": ckpt_psnr, "ssim": ckpt_ssim,
                                           "lpips": ckpt_lpips, "n_gaussians": model.n_gaussians})
                if cfg.use_agents and training_agent is not None:
                    effective_densify_until_iter, should_early_stop, record = \
                        apply_training_agent_at_checkpoint(
                            training_agent, held_out_history, best_held_out, current_iter=it,
                            max_iterations=cfg.max_iterations, checkpoint_every=cfg.checkpoint_every,
                            effective_densify_until_iter=effective_densify_until_iter,
                            n_gaussians=model.n_gaussians, max_gaussians=cfg.max_gaussians,
                            min_checkpoints_before_acting=MIN_CHECKPOINTS_BEFORE_AGENT_ACTS)
                    if record is not None:
                        agent_decisions.append(record)
                        print(f"      [agent] iter {it}: {record['action']} -- {record['reasoning']}")

        if should_early_stop:
            print(f"      [agent] stopping training early at iter {it} (see agent_decisions.json "
                  f"for the full reasoning trail)")
            break

    with open(os.path.join(cfg.out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(cfg.out_dir, "best_checkpoint.json"), "w") as f:
        json.dump(best_held_out, f, indent=2)
    if agent_decisions:
        with open(os.path.join(cfg.out_dir, "agent_decisions.json"), "w") as f:
            json.dump({"held_out_history": held_out_history, "decisions": agent_decisions,
                        "final_effective_densify_until_iter": effective_densify_until_iter,
                        "stopped_early": should_early_stop}, f, indent=2)
    if is_ho:
        lpips_note = (f"iter {best_held_out['lpips']['iter']} "
                      f"(LPIPS={best_held_out['lpips']['value']:.3f})"
                      if best_held_out["lpips"]["iter"] is not None else "unavailable")
        print(f"\nBEST held-out checkpoints: PSNR iter {best_held_out['psnr']['iter']} "
              f"({best_held_out['psnr']['value']:.2f}); SSIM iter {best_held_out['ssim']['iter']} "
              f"({best_held_out['ssim']['value']:.3f}); LPIPS {lpips_note}. "
              f"Choose based on the product goal, not simply the final iteration. If that best "
              f"iteration isn't cfg.max_iterations, export_obj.py can regenerate the .obj below "
              f"from its full_state_iter*.pt instead of the final one.")

    if cfg.export_mesh:
        _export_mesh(model, cfg.out_dir, cfg.max_iterations, cfg.mesh_min_opacity, cfg.mesh_min_confidence,
                     method=cfg.mesh_method, poisson_depth=cfg.mesh_poisson_depth,
                     density_trim_quantile=cfg.mesh_density_trim_quantile,
                     use_covariance_normals=cfg.mesh_use_covariance_normals,
                     floater_filter=cfg.mesh_floater_filter,
                     floater_anisotropy_below=cfg.mesh_floater_anisotropy_below,
                     floater_scale_percentile=cfg.mesh_floater_scale_percentile,
                     min_component_vertices=cfg.mesh_min_component_vertices,
                     keep_top_k_components=cfg.mesh_keep_top_k_components,
                     voxel_size=cfg.mesh_voxel_size,
                     camera_hfov_deg=cfg.mesh_camera_hfov_deg,
                     frame_width=cfg.mesh_frame_width, frame_height=cfg.mesh_frame_height)
    return history, agent_decisions


def _gaussian_positions_and_colors(model: GaussianModel):
    """Shared position/RGB extraction -- pulled out so the checkpoint .ply
    (_save_checkpoint) and the mesh export (_export_mesh) never risk silently
    computing color differently. SH-DC-only (sh_c0 * dc + 0.5): the flat/ambient
    color term, not full view-dependent SH -- correct for "what color is this
    Gaussian, roughly", which is all either a point-cloud viewer or a mesh vertex
    can use anyway."""
    positions = model.xyz.detach().cpu().numpy()
    sh_c0 = 0.28209479177387814
    colors = (model.features_dc.detach().cpu().numpy()[:, 0, :] * sh_c0 + 0.5).clip(0, 1)
    return positions, colors


def _save_checkpoint(model: GaussianModel, out_dir: str, it: int):
    """Saves position+color only, as a .ply for quick viewing in MeshLab/open3d/CloudCompare.
    This alone CANNOT be reloaded as a working Gaussian model (no scale/opacity/rotation/SH) --
    see _save_full_state below for that. Kept because a .ply is the fastest way to just look at
    where the reconstruction landed."""
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    from layer1_cpu_sandbox.utils import save_ply

    positions, colors = _gaussian_positions_and_colors(model)
    save_ply(os.path.join(out_dir, f"checkpoint_iter{it}.ply"), positions, colors)
    conf = model.confidence().detach().cpu().numpy()
    print(f"      [checkpoint @ iter {it}] saved {positions.shape[0]} Gaussians, "
          f"mean confidence={conf.mean():.3f}, high-conf fraction={float((conf >= 0.6).mean()):.3f}")


def _export_mesh(model: GaussianModel, out_dir: str, it: int, min_opacity, min_confidence,
                  method: str = "poisson", poisson_depth: int = 9, density_trim_quantile: float = 0.03,
                  use_covariance_normals: bool = True, floater_filter: bool = True,
                  floater_anisotropy_below: float = 3.0, floater_scale_percentile: float = 90.0,
                  min_component_vertices: int = 20, keep_top_k_components: Optional[int] = None,
                  voxel_size: Optional[float] = None, camera_hfov_deg: float = 84.0,
                  frame_width: int = 640, frame_height: int = 480) -> bool:
    """Confidence/opacity-gated .obj export of the current Gaussian centers (see
    layer1_cpu_sandbox/reconstruction/mesh_export.py for the actual reconstruction
    and why it lives there). Returns True on success, False on failure -- NEVER
    raises, because this runs after training has already finished and a meshing
    problem (e.g. too few points survive the confidence gate for this particular
    scene) must not discard hours of GPU time and every OTHER artifact already on
    disk (.pt/.ply/history/best_checkpoint). Rerun export_obj.py against the saved
    full_state_iter*.pt checkpoint to retry with different settings -- no
    retraining needed.

    2026-09 fix: this used to build `keep_mask` from fixed opacity/confidence floats and
    call export_point_cloud_to_obj with none of the docs/MESH_QUALITY_AUDIT.md fixes (auto
    thresholds, covariance normals, floater filtering, debris/component cleanup) -- those
    all existed in mesh_export.py / gaussian_geometry.py / export_obj.py already, but only
    the STANDALONE export_obj.py script actually used them. A run's automatic end-of-training
    mesh therefore shipped with the pre-audit topology (measured on a real checkpoint: 1042
    components, 97.7% of vertices in the largest one -- essentially the audit's own "before"
    numbers, not its "after" ones) unless someone remembered to separately re-run
    export_obj.py. This function now does what export_obj.py does, inline, so the automatic
    export gets the same treatment as a manual one -- see that script for the same logic
    with per-step console output, if you want to re-run it standalone against a saved
    checkpoint with different settings.
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    from layer1_cpu_sandbox.reconstruction.mesh_export import export_point_cloud_to_obj, MeshExportError
    from layer1_cpu_sandbox.reconstruction.gaussian_geometry import (
        compute_gaussian_geometry, suggest_threshold_from_valley, floater_mask)
    from layer1_cpu_sandbox.reconstruction.frustum_check import check_frustum_shape, vfov_from_hfov

    positions, colors = _gaussian_positions_and_colors(model)
    opacity = model.get_opacity().detach().cpu().numpy().reshape(-1)
    confidence = model.confidence().detach().cpu().numpy().reshape(-1)
    scale = model.get_scaling().detach().cpu().numpy()
    rotation = model.get_rotation().detach().cpu().numpy()
    geometry = compute_gaussian_geometry(scale, rotation)

    resolved_min_opacity = (suggest_threshold_from_valley(opacity) if min_opacity == "auto"
                             else float(min_opacity))
    resolved_min_confidence = (suggest_threshold_from_valley(confidence) if min_confidence == "auto"
                                else float(min_confidence))
    if min_opacity == "auto" or min_confidence == "auto":
        print(f"      [mesh export @ iter {it}] auto thresholds -> "
              f"min_opacity={resolved_min_opacity:.4g} (requested {min_opacity!r}), "
              f"min_confidence={resolved_min_confidence:.4g} (requested {min_confidence!r})")
    keep_mask = (opacity >= resolved_min_opacity) & (confidence >= resolved_min_confidence)
    if floater_filter:
        flt_mask = floater_mask(geometry, anisotropy_below=floater_anisotropy_below,
                                 scale_above_percentile=floater_scale_percentile)
        print(f"      [mesh export @ iter {it}] floater_filter: {int(flt_mask.sum())}/{model.n_gaussians} "
              f"flagged (roughly-isotropic & unusually large)")
        keep_mask &= ~flt_mask
    print(f"      [mesh export @ iter {it}] {model.n_gaussians} Gaussians -> "
          f"{int(keep_mask.sum())} survive opacity>={resolved_min_opacity:.4g} and "
          f"confidence>={resolved_min_confidence:.4g}"
          f"{' and floater filter' if floater_filter else ''}")

    mesh_path = os.path.join(out_dir, f"mesh_iter{it}.obj")
    normals = geometry.normals if use_covariance_normals else None
    try:
        stats = export_point_cloud_to_obj(mesh_path, positions, colors, keep_mask=keep_mask, normals=normals,
                                           method=method, poisson_depth=poisson_depth,
                                           density_trim_quantile=density_trim_quantile, voxel_size=voxel_size,
                                           min_component_vertices=min_component_vertices,
                                           keep_largest_n_components=keep_top_k_components)
    except (MeshExportError, ImportError, IOError) as e:
        print(f"      WARNING: mesh export failed ({type(e).__name__}: {e}). Every other artifact "
              f"(.pt/.ply/history) is unaffected -- retry via export_obj.py once you've addressed this, "
              f"e.g. lower --min_opacity/--min_confidence or try --method ball_pivoting.")
        return False
    print(f"      [mesh export @ iter {it}] wrote {stats.path}: {stats.n_vertices} vertices, "
          f"{stats.n_triangles} triangles, {stats.n_connected_components} connected components "
          f"({100*stats.largest_component_fraction:.1f}% of vertices in the largest one, "
          f"{stats.n_components_removed_as_debris} removed as debris)")

    # Post-training geometry sanity check, on the TRAINED and FILTERED Gaussians -- the
    # pre-training check in real_pipeline.py only looked at the seed cloud; training can in
    # principle move things, and this is the check that would catch it. See frustum_check.py
    # and docs/MESH_QUALITY_AUDIT.md for what a positive verdict here means.
    try:
        vfov = vfov_from_hfov(camera_hfov_deg, frame_width, frame_height)
        fres = check_frustum_shape(positions[keep_mask], expected_hfov_deg=camera_hfov_deg,
                                    expected_vfov_deg=vfov)
        print(f"      [mesh export @ iter {it}] geometry sanity check on exported Gaussians:")
        for line in fres.format_report().splitlines():
            print(f"      {line}")
        if fres.is_frustum_shaped:
            print(f"      WARNING [mesh export @ iter {it}]: the EXPORTED Gaussians are shaped like the "
                  f"camera's viewing volume, not a scene. Good PSNR/SSIM does not clear this (shape-radiance "
                  f"ambiguity) -- see docs/MESH_QUALITY_AUDIT.md before trusting mesh_iter{it}.obj.")
    except Exception as e:  # never let the sanity check itself break a finished run
        print(f"      NOTE: post-export geometry sanity check failed to run ({type(e).__name__}: {e}); "
              f"the mesh above was still written.")
    return True


def _save_full_state(model: GaussianModel, out_dir: str, it: int):
    """
    Saves EVERY parameter needed to reload a working, re-renderable Gaussian model --
    this did not exist before this fix, which meant a finished training run's model
    only survived as a position+color point cloud (see _save_checkpoint above):
    no way to re-render it, no way to compute PSNR/SSIM/LPIPS against held-out frames,
    no way to continue training, once the process that trained it exited. Real gap,
    caught the first time this loop actually finished a run (10,000 real iterations,
    see chat history) and someone asked "where can I see the reconstruction" and the
    honest answer for anything beyond a bare point cloud was "you can't, anymore".

    Also saves `confidence_stats`' raw accumulators (obs_count/angle_min/angle_max/
    consistency_sum/quality_sum/pose_conf_sum/n_accum), added while wiring
    confidence-gated mesh export through a RELOADED checkpoint (export_obj.py) and
    finding a second, related gap: without these, `load_full_state(...).confidence()`
    silently returns each Gaussian's creation-time `prior_confidence` instead of what
    training actually accumulated -- see `blended_confidence` in
    confidence_gaussian_model.py, `has_obs = stats.n_accum > 0` is all-False right
    after a fresh `GaussianConfidenceStats()`, so it falls through to the prior every
    time. Invisible for rendering/PSNR (those don't touch confidence at all) and for
    the ALREADY-in-process `train()` call to `_export_mesh` (the live model's stats
    were never reset), which is exactly why it went unnoticed until something -- this
    feature -- specifically needed confidence to survive a save/load round trip.
    """
    cs = model.confidence_stats
    state = {
        "sh_degree": model.cfg.sh_degree,
        "xyz": model.xyz.detach().cpu(),
        "features_dc": model.features_dc.detach().cpu(),
        "features_rest": model.features_rest.detach().cpu(),
        "log_scaling": model.log_scaling.detach().cpu(),
        "rotation": model.rotation.detach().cpu(),
        "opacity_logit": model.opacity_logit.detach().cpu(),
        "prior_confidence": model.prior_confidence.detach().cpu(),
        "confidence_obs_count": cs.obs_count.detach().cpu(),
        "confidence_angle_min": cs.angle_min.detach().cpu(),
        "confidence_angle_max": cs.angle_max.detach().cpu(),
        "confidence_consistency_sum": cs.consistency_sum.detach().cpu(),
        "confidence_quality_sum": cs.quality_sum.detach().cpu(),
        "confidence_pose_conf_sum": cs.pose_conf_sum.detach().cpu(),
        "confidence_n_accum": cs.n_accum.detach().cpu(),
        "iteration": it,
    }
    torch.save(state, os.path.join(out_dir, f"full_state_iter{it}.pt"))


def load_full_state(path: str, device: str = "cuda") -> GaussianModel:
    """Reloads a model saved by _save_full_state -- e.g.
    model = load_full_state('outputs/gpu_train_real1/full_state_iter10000.pt')
    Then render_gaussian_view(model, some_camera, device) works exactly as it did mid-training.

    Checkpoints written before the confidence_stats fix above lack the
    "confidence_*" keys; they still load (falling back to a fresh, all-zero
    GaussianConfidenceStats, i.e. the OLD behavior of confidence() == prior_confidence)
    rather than raising, so nothing already on disk from an earlier run breaks."""
    state = torch.load(path, map_location=device)
    model = GaussianModel(GaussianTrainingConfig(sh_degree=state["sh_degree"]))
    model.xyz = torch.nn.Parameter(state["xyz"].to(device))
    model.features_dc = torch.nn.Parameter(state["features_dc"].to(device))
    model.features_rest = torch.nn.Parameter(state["features_rest"].to(device))
    model.log_scaling = torch.nn.Parameter(state["log_scaling"].to(device))
    model.rotation = torch.nn.Parameter(state["rotation"].to(device))
    model.opacity_logit = torch.nn.Parameter(state["opacity_logit"].to(device))
    n = model.xyz.shape[0]
    model.confidence_stats = GaussianConfidenceStats(n, device=device)
    if "confidence_obs_count" in state:
        model.confidence_stats.obs_count = state["confidence_obs_count"].to(device)
        model.confidence_stats.angle_min = state["confidence_angle_min"].to(device)
        model.confidence_stats.angle_max = state["confidence_angle_max"].to(device)
        model.confidence_stats.consistency_sum = state["confidence_consistency_sum"].to(device)
        model.confidence_stats.quality_sum = state["confidence_quality_sum"].to(device)
        model.confidence_stats.pose_conf_sum = state["confidence_pose_conf_sum"].to(device)
        model.confidence_stats.n_accum = state["confidence_n_accum"].to(device)
    model.propagator = ObservationConfidencePropagator(n, PropagationConfig(), device=device)
    model.register_buffer("prior_confidence", state["prior_confidence"].to(device))
    return model


def _render_and_save_views(model: GaussianModel, cameras: List[TrainingCamera], out_dir: str, it: int,
                            device: str, n_views: int = 3, is_held_out: bool = True, active_sh_degree: int = None):
    """
    Renders a few camera views through the REAL differentiable rasterizer (not the
    point-splat placeholder used elsewhere in this project for CPU-only evaluation)
    and saves rendered-vs-actual-vs-error-map PNGs, PLUS real PSNR/SSIM/LPIPS numbers
    -- reusing the exact same metric code the CPU sandbox uses
    (layer1_cpu_sandbox/evaluation/metrics_visual.py), so a number here means the same
    thing it means everywhere else in this project. When `cameras` is the held-out set
    (the normal case -- see train()'s train/held-out split), these numbers are a
    genuine generalization check, not a training-fit check.
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    from layer1_cpu_sandbox.evaluation.metrics_visual import evaluate_visual_fidelity

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    label = "held-out (novel view)" if is_held_out else "TRAINING view (memorization risk, see console warning)"
    idxs = np.linspace(0, len(cameras) - 1, min(n_views, len(cameras)), dtype=int)
    metrics_this_checkpoint = []
    with torch.no_grad():
        for k, ci in enumerate(idxs):
            cam = cameras[int(ci)]
            rendered, _, _, _ = render_gaussian_view(model, cam, device, active_sh_degree=active_sh_degree)
            rendered_np = (rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_np = (cam.gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            diff = np.abs(rendered_np.astype(np.float32) - gt_np.astype(np.float32)).mean(axis=-1)

            vf = evaluate_visual_fidelity(rendered_np, gt_np)
            metrics_this_checkpoint.append(vf)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(gt_np); axes[0].set_title(f"Actual frame (frame {cam.frame_idx}, {label})"); axes[0].axis("off")
            lpips_str = "n/a" if np.isnan(vf.lpips) else f"{vf.lpips:.3f}"
            axes[1].imshow(rendered_np)
            axes[1].set_title(f"Rendered @ iter {it}\nPSNR={vf.psnr:.1f} SSIM={vf.ssim:.3f} LPIPS={lpips_str}")
            axes[1].axis("off")
            im = axes[2].imshow(diff, cmap="inferno", vmin=0, vmax=80)
            axes[2].set_title("Abs. error"); axes[2].axis("off")
            fig.colorbar(im, ax=axes[2], fraction=0.046)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"render_iter{it}_frame{cam.frame_idx}.png"), dpi=120)
            plt.close(fig)

    mean_psnr = float(np.mean([m.psnr for m in metrics_this_checkpoint]))
    mean_ssim = float(np.mean([m.ssim for m in metrics_this_checkpoint]))
    lpips_vals = [m.lpips for m in metrics_this_checkpoint if not np.isnan(m.lpips)]
    mean_lpips_str = f"{np.mean(lpips_vals):.3f}" if lpips_vals else "n/a"
    with open(os.path.join(out_dir, "eval_history.jsonl"), "a") as f:
        f.write(json.dumps({"iter": it, "is_held_out": is_held_out, "mean_psnr": mean_psnr,
                             "mean_ssim": mean_ssim, "mean_lpips": (float(np.mean(lpips_vals)) if lpips_vals
                                                                     else None)}) + "\n")
    # PER-VIEW breakdown, not just the mean -- added after a mean of [20.8, 16.6, 15.8] PSNR
    # quietly hid that two of three held-out views (the ones nearest the start/end of the
    # extracted flight, structurally the least multi-view-confirmed by construction) were
    # near-total failures while the middle one was fine. The mean alone made that invisible
    # in the console log; you had to separately open all three PNGs to see it.
    per_view_str = "  ".join(f"frame{cam.frame_idx}: PSNR={m.psnr:.1f}/SSIM={m.ssim:.3f}"
                              for cam, m in zip((cameras[int(ci)] for ci in idxs), metrics_this_checkpoint))
    print(f"      [checkpoint @ iter {it}] {label}: mean PSNR={mean_psnr:.2f}  mean SSIM={mean_ssim:.3f}  "
          f"mean LPIPS={mean_lpips_str}  ({len(idxs)} views) -- saved PNGs + full_state_iter{it}.pt "
          f"(reloadable via train_gpu.load_full_state)")
    print(f"         per-view: {per_view_str}")
    mean_lpips = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
    return mean_psnr, mean_ssim, mean_lpips


# ------------------------------------------------------------ dataset bridges --

def build_cameras_from_synthetic(dataset, pipeline_result) -> List[TrainingCamera]:
    cams = []
    for i in pipeline_result.kept_frame_indices:
        frame = dataset.frames[i]
        img = torch.from_numpy(frame.rgb).float().permute(2, 0, 1) / 255.0
        q = pipeline_result.frame_quality_scores[i].overall
        pc = pipeline_result.fused_traj.pose_confidence(i)
        cams.append(TrainingCamera(pose=pipeline_result.fused_traj.camera_pose(i), K=dataset.K,
                                    gt_image=img, frame_quality=q, pose_confidence=pc, frame_idx=i))
    return cams


def build_cameras_from_real(dataset, pipeline_result) -> List[TrainingCamera]:
    # Frame quality already controls Layer 1 selection. Passing it through to
    # Layer 3 prevents motion blur and exposure failures being treated as perfect evidence.
    from layer1_cpu_sandbox.perception.frame_quality import compute_frame_quality
    cams = []
    n_per_frame_K = 0
    for i in pipeline_result.kept_frame_indices:
        frame = dataset.frames[i]
        img = torch.from_numpy(frame.rgb).float().permute(2, 0, 1) / 255.0
        pose_conf = 1.0 - min(frame.position_residual_m / 3.0, 1.0)
        # Per-frame K (zoom lens) when this frame has one, else the shared dataset.K -- see
        # real_dataset_builder.py's build_real_dataset. render_gaussian_view/_camera_to_rasterizer_*
        # already read fx/fy/width/height off whatever TrainingCamera.K they're given, per camera,
        # so no rasterizer-side change is needed for this to take effect.
        frame_K = getattr(frame, "K", None) or dataset.K
        if frame_K is not dataset.K:
            n_per_frame_K += 1
        cams.append(TrainingCamera(pose=frame.pose, K=frame_K, gt_image=img,
                                    frame_quality=compute_frame_quality(frame.rgb).overall,
                                    pose_confidence=pose_conf, frame_idx=frame.idx))
    if n_per_frame_K:
        print(f"[cameras] {n_per_frame_K}/{len(cams)} cameras use their own per-frame intrinsics "
              f"(zoom lens) rather than the shared --camera_hfov_deg assumption.")
    return cams


def filter_downward_mapping_frames(dataset, max_forward_z: float):
    """Keep only views whose optical axis has sufficient downward component.

    A reef/ground model cannot explain a horizon-facing frame: it contains sky
    and effectively infinite-depth water that have no stable correspondence to
    the nadir views.  Rather than forcing a static ground model to average both
    scene types, exclude those frames from this model and report their original
    ids.  ``max_forward_z=-0.5`` means the camera must look at least 30 degrees
    below the horizon; pass a value >= 1 to disable filtering.
    """
    if max_forward_z >= 1.0:
        return dataset, []
    kept = [frame for frame in dataset.frames if float(frame.pose.forward()[2]) <= max_forward_z]
    rejected = [frame.idx for frame in dataset.frames if float(frame.pose.forward()[2]) > max_forward_z]
    if len(kept) < 8:
        raise ValueError(
            f"Downward-view filter retained only {len(kept)} frames; relax --max_forward_z "
            f"(current {max_forward_z}) or disable it with --max_forward_z 1.0.")
    return dataclasses.replace(dataset, frames=kept), rejected


def filter_stable_focal_length(dataset, max_focal_drift_frac: float):
    """train_gpu.py-side wrapper around dji_log_parser.filter_stable_focal_length_frames,
    adapted to the same (dataset, rejected_idx_list) calling convention as
    filter_downward_mapping_frames above. No-op for any dataset whose frames don't carry
    focal_len_mm (i.e. every format except Format D SRT logs).

    Historically this was the ONLY defense against a zoom lens changing focal length
    mid-clip, because the whole repo assumed one shared K -- see this function's git
    history / RUN_WITH_SRT_POSE_FIX.md. Real per-frame intrinsics now exist (see
    real_dataset_builder.py's build_real_dataset), so with the new default
    (--max_focal_drift_frac 1.0) this is a no-op in practice: every frame keeps its own
    K instead of being dropped. Lower --max_focal_drift_frac to re-enable the old
    stable-focal-length-window behavior (e.g. if the DJI Air 3's wide/tele physical
    camera switch, which per-frame K does NOT model, turns out to matter for your scene).
    """
    from layer1_cpu_sandbox.real_data.dji_log_parser import filter_stable_focal_length_frames
    if max_focal_drift_frac >= 1.0:
        return dataset, []
    kept, rejected = filter_stable_focal_length_frames(dataset, max_drift_frac=max_focal_drift_frac)
    if len(kept) < 8:
        raise ValueError(
            f"Stable-focal-length filter retained only {len(kept)} frames; relax "
            f"--max_focal_drift_frac (current {max_focal_drift_frac}) or disable it with 1.0.")
    return dataclasses.replace(dataset, frames=kept), [f.idx for f in rejected]


def _build_layer4_agents(use_agents: bool, use_llm: bool, model: str):
    """Returns (frame_agent, training_agent), both None if `use_agents` is False -- see
    module docstring "AGENT HOOKS". Mirrors layer4_agents/run_agentic_experiment.py's own
    agent-construction pattern exactly, so learning one teaches you the other."""
    if not use_agents:
        return None, None
    if not use_llm:
        from layer4_agents.mock_agents import MockFrameAgent, MockTrainingControlAgent
        return MockFrameAgent(), MockTrainingControlAgent()
    from layer4_agents.llm_agents import LLMFrameAgent, LLMTrainingControlAgent, build_default_client
    client = build_default_client()
    return LLMFrameAgent(client, model=model), LLMTrainingControlAgent(client, model=model)


def _frame_agent_decision_kwargs(frame_agent, frames, out_dir: str) -> dict:
    """Runs the Frame Agent once against the raw frame sequence (same agent, same
    frame_quality.py/frame_overlap.py signals layer4_agents/orchestrator.py already uses
    on Layer 1) and returns {"keep_fraction", "min_overall"} to merge into whichever
    PipelineConfig/RealPipelineConfig kwargs the caller is building. Saves the decision
    to `out_dir`/frame_agent_decision.json so it's part of the run's saved artifacts,
    same as everything else train() already writes to cfg.out_dir."""
    import cv2
    from layer1_cpu_sandbox.perception.frame_quality import compute_frame_quality
    from layer1_cpu_sandbox.perception.frame_overlap import compute_consecutive_overlap
    from layer4_agents.context_builders import build_frame_agent_context

    grays = [cv2.cvtColor(f.rgb, cv2.COLOR_RGB2GRAY) for f in frames]
    quality_scores = [compute_frame_quality(f.rgb) for f in frames]
    overlap = compute_consecutive_overlap(grays)
    ctx = build_frame_agent_context(quality_scores, overlap)
    decision = frame_agent.decide(ctx)
    print(f"[agent] frame selection: keep_fraction={decision.keep_fraction:.2f}, "
          f"min_overall={decision.min_overall:.2f} -- {decision.reasoning}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "frame_agent_decision.json"), "w") as f:
        json.dump(dataclasses.asdict(decision), f, indent=2)
    return {"keep_fraction": decision.keep_fraction, "min_overall": decision.min_overall}


def _float_or_auto(s: str):
    """argparse type for --mesh_min_opacity/--mesh_min_confidence: 'auto' (default) derives
    the threshold from this run's own Gaussian distribution at export time (see _export_mesh);
    any other value is parsed as a float and used as-is, exactly like before this fix."""
    if s == "auto":
        return "auto"
    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # 30ms/iteration was MEASURED at 225,649 Gaussians, 640x480 on a T4 (see chat history) --
    # 20,000 iterations at that rate is ~10 minutes, comfortably inside a Colab session, and
    # real 3DGS/FastGS training commonly runs 15,000-30,000 iterations. The previous default
    # (8000) was a conservative first-run guess, not a considered choice; there was no reason
    # to stay there once we had a real throughput number.
    ap.add_argument("--max_iterations", type=int, default=20000)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--target_fps", type=float, default=2.0,
                    help="Video sampling rate; raise only when adjacent views retain useful parallax.")
    ap.add_argument("--camera_hfov_deg", type=float, default=84.0,
                    help="Measured horizontal FOV after resize; 84 is only a Phantom 4 Pro fallback.")
    ap.add_argument("--position_smoothing_window", type=int, default=5,
                    help="Telemetry smoothing window in log samples.")
    ap.add_argument("--max_forward_z", type=float, default=-0.5,
                    help="For a ground/reef model, keep cameras with optical-axis world-Z <= this value; 1 disables. "
                         "This assumes a near-nadir (looking down) shot -- for an oblique/street-level capture "
                         "with a forward-tilted gimbal, this default may reject most or all frames; check the "
                         "printed exclusion count and raise this (or pass 1 to disable it) if so.")
    ap.add_argument("--assumed_gimbal_pitch_deg", type=float, default=None,
                    help="REQUIRED when --log is a .srt file (that format carries no per-frame gimbal pitch, "
                         "only aircraft heading). -90 = straight down, roughly -30 to -45 = typical oblique/"
                         "forward-looking, closer to 0 = near-horizontal. No default -- check your DJI Fly "
                         "app's flight record or estimate from the footage itself; guessing wrong here produces "
                         "a plausible-looking but geometrically wrong reconstruction, not an obvious failure.")
    ap.add_argument("--assumed_gimbal_roll_deg", type=float, default=0.0,
                    help="Only used with --log .srt. 0 is a safe default -- unlike pitch, a stabilized gimbal "
                         "genuinely holds roll near 0 by design.")
    ap.add_argument("--max_focal_drift_frac", type=float, default=1.0,
                    help="Only meaningful for Format D SRT logs (per-frame focal_len, e.g. a zoom shot): "
                         "keeps only the longest contiguous run of frames whose focal length stays within "
                         "this fraction of that run's own starting value. Real per-frame intrinsics now "
                         "exist (see real_dataset_builder.py / build_cameras_from_real), so the default is "
                         "1.0 (keep every frame, each with its own K) rather than the old shared-K "
                         "workaround. This does NOT model the DJI Air 3 wide/tele pair being two physically "
                         "separate cameras -- frames deep into the tele range still carry a small, "
                         "unmodelled baseline error. Lower this (e.g. 0.1) to fall back to the old "
                         "stable-focal-length-window behavior if that residual error turns out to matter "
                         "more than the extra frames help.")
    ap.add_argument("--pose_source", choices=["telemetry", "visual_odometry"], default="telemetry",
                    help="'telemetry' uses the DJI SRT/CSV pose directly. 'visual_odometry' recovers relative "
                         "rotation and translation direction from consecutive image features, using telemetry only "
                         "to scale each step. Recommended for FrameCnt SRT files, which contain no attitude.")
    ap.add_argument("--vo_min_matches", type=int, default=80,
                    help="Minimum ORB matches required per consecutive frame pair for --pose_source visual_odometry.")
    ap.add_argument("--vo_min_inliers", type=int, default=40,
                    help="Minimum essential-matrix inliers required per pair for --pose_source visual_odometry.")
    ap.add_argument("--vo_max_frame_gap", type=int, default=3,
                    help="When an adjacent visual-odometry pair is weak, try the next 1..N input frames and "
                         "drop only untrackable intermediate frames. 3 is a safe short-clip default.")
    ap.add_argument("--depth_window", type=int, default=2,
                    help="Neighbouring frames on either side used by plane-sweep MVS.")
    ap.add_argument("--n_depths", type=int, default=128,
                    help="Number of logarithmically spaced plane-sweep depth hypotheses.")
    ap.add_argument("--depth_min", type=float, default=3.0)
    ap.add_argument("--depth_max", type=float, default=100.0)
    ap.add_argument("--depth_spacing", choices=["log", "linear"], default="log")
    ap.add_argument("--depth_boundary_reject", action=argparse.BooleanOptionalAction, default=True,
                    help="Discard endpoint depth estimates; prevents out-of-range rays from becoming a frustum.")
    ap.add_argument("--no_dynamic_filter", action="store_true",
                    help="Disable the experimental residual-flow mask for a diagnostic comparison.")
    ap.add_argument("--keep_fraction", type=float, default=1.0,
                    help="Fraction of quality-ranked frames retained for MVS. Keep all initially; reduce only if "
                         "the printed frame-quality report identifies genuinely bad frames.")
    ap.add_argument("--seed_min_observations", type=int, default=2,
                    help="Minimum distinct MVS views that must agree on a voxel before it may seed a Gaussian. "
                         "2 is the minimum evidence for depth; 1 accepts a single-view depth guess.")
    ap.add_argument("--seed_min_consistency", type=float, default=0.25,
                    help="Minimum mean plane-sweep cost-curve consistency for a Gaussian seed.")
    ap.add_argument("--seed_min_confidence", type=float, default=0.12,
                    help="Minimum observation-aware confidence for a Gaussian seed.")
    ap.add_argument("--min_seed_points", type=int, default=500,
                    help="Fail before GPU training when too few multi-view-supported seed points remain.")
    ap.add_argument("--n_eval_views", type=int, default=3,
                    help="Number of evenly spaced held-out frames rendered at each checkpoint; use 999 for all.")
    # Growth PAST here was measured to actively HURT held-out generalization (see chat
    # history: 20,000-iteration real run, best held-out SSIM at iter 6500/246K Gaussians,
    # declining monotonically all the way to iter 20000/829K Gaussians -- classic
    # overfitting once Gaussian count outgrows what 77 training cameras can actually
    # constrain). 15000 was a guess before we had this data; 7000 is not -- it's where the
    # SAME run's own numbers say densification stopped helping.
    ap.add_argument("--densify_until_iter", type=int, default=7000)
    ap.add_argument("--confidence_densify_above", type=float, default=0.40)
    ap.add_argument("--min_densify_observations", type=int, default=2,
                    help="Fresh render observations required before a Gaussian may split again.")
    ap.add_argument("--max_gaussians", type=int, default=150000,
                    help="Hard primitive budget; prevents overfitting sparse single-pass views by exponential splitting.")
    ap.add_argument("--split_scale_thresh", type=float, default=0.01,
                    help="Fraction of scene extent above which a densify candidate splits instead of clones. "
                         "Original 3DGS default (percent_dense) is 0.01; FastGS_QADS uses a more aggressive "
                         "0.001. Only lower this together with --opacity_reset_interval > 0.")
    ap.add_argument("--opacity_reset_interval", type=int, default=0,
                    help="FastGS_QADS-style periodic opacity reset (their default: 3000). 0 disables it. "
                         "On the reef data this did not beat plain training at matched iteration counts -- "
                         "worth re-testing on a new scene, not assuming it'll help here too.")
    ap.add_argument("--opacity_reset_target", type=float, default=0.05,
                    help="Opacity ceiling applied at each reset. Reference implementations use 0.01.")
    ap.add_argument("--max_world_scale_frac", type=float, default=0.1,
                    help="Prune any Gaussian whose world-space scale exceeds this fraction of the "
                         "reconstructed point cloud's own extent (FastGS_QADS's / stock 3DGS's "
                         "big_points_ws floater check).")
    ap.add_argument("--holdout_fraction", type=float, default=0.0,
                    help="Fraction of cameras held out as a genuine novel-view check. 0 (default) trains "
                         "on every available camera with no holdout -- fine when cameras are scarce, but "
                         "checkpoint renders then only show training-view fit, not generalization. Set to "
                         "e.g. 0.15 once you have enough cameras that losing some to holdout doesn't starve "
                         "training.")
    # sh_degree=0 (flat, non-view-dependent color) cannot represent a specular highlight at
    # all, by construction -- not "represents it poorly", structurally CANNOT. The missing
    # sun-glint in the iter-8000 render is exactly what that predicts. Raised to 2 (9 SH
    # coefficients/Gaussian) with a PROGRESSIVE schedule (see TrainConfig.sh_degree_interval)
    # rather than fixed from iteration 1 -- starting directly at high SH degree before coarse
    # geometry is even in place is known to destabilize early optimization in the reference
    # 3DGS implementation, hence the ramp.
    ap.add_argument("--sh_degree", type=int, default=2)
    ap.add_argument("--out_dir", type=str, default="outputs/gpu_train_run1")
    ap.add_argument("--auto_vggt_mesh", action=argparse.BooleanOptionalAction, default=True,
                    help="After successful real-data Gaussian training, automatically run VGGT fusion and the validated Trim001/Clean500 Poisson export. Use --no-auto-vggt-mesh to skip.")
    ap.add_argument("--vggt_max_frames", type=int, default=28,
                    help="Maximum number of evenly spaced views in the post-training VGGT joint pass (6-40; 28 is recommended for a Colab T4).")
    # --- final .obj mesh export -- see TrainConfig / _export_mesh / mesh_export.py ---
    ap.add_argument("--export_mesh", action=argparse.BooleanOptionalAction, default=True,
                    help="Export outputs/<run>/mesh_iter<N>.obj from the final Gaussians when training "
                         "finishes. A failure here is reported but never crashes the run -- see --no-export_mesh "
                         "to skip it outright, or export_obj.py to (re)run it later against any checkpoint.")
    ap.add_argument("--mesh_min_opacity", type=_float_or_auto, default="auto",
                    help="float, or 'auto' (default) for a histogram-valley threshold derived from this "
                         "run's own Gaussians -- see docs/MESH_QUALITY_AUDIT.md (a fixed 0.1 was a "
                         "near-no-op on the checkpoint that prompted this).")
    ap.add_argument("--mesh_min_confidence", type=_float_or_auto, default="auto",
                    help="float, or 'auto' (default). See --mesh_min_opacity.")
    ap.add_argument("--mesh_method", choices=["poisson", "ball_pivoting"], default="poisson")
    ap.add_argument("--mesh_poisson_depth", type=int, default=9)
    ap.add_argument("--mesh_density_trim_quantile", type=float, default=0.03)
    ap.add_argument("--mesh_use_covariance_normals", action=argparse.BooleanOptionalAction, default=True,
                    help="derive surface normals from each Gaussian's own scale+rotation instead of "
                         "generic point-cloud PCA. See docs/MESH_QUALITY_AUDIT.md.")
    ap.add_argument("--mesh_floater_filter", action=argparse.BooleanOptionalAction, default=True,
                    help="exclude roughly-isotropic, unusually-large Gaussians (haze/sky/background) "
                         "before meshing.")
    ap.add_argument("--mesh_floater_anisotropy_below", type=float, default=3.0)
    ap.add_argument("--mesh_floater_scale_percentile", type=float, default=90.0)
    ap.add_argument("--mesh_min_component_vertices", type=int, default=20,
                    help="drop connected mesh components smaller than this many vertices (debris "
                         "cleanup) after reconstruction; 0 disables. The shipped mesh_iter8000.obj had "
                         "1042 components (97.7%% of vertices in the largest) because this was "
                         "effectively 0 in every automatic export before this fix -- see "
                         "docs/MESH_QUALITY_AUDIT.md.")
    ap.add_argument("--mesh_keep_top_k_components", type=int, default=None,
                    help="keep only the N largest connected mesh components; None disables. Use instead "
                         "of (or with) --mesh_min_component_vertices if you know roughly how many "
                         "separate real structures the scene should have.")
    ap.add_argument("--mesh_voxel_size", type=float, default=None,
                    help="optional pre-reconstruction downsample to homogenize point density before "
                         "Poisson (Gaussian centers are not uniformly spaced). None = no downsampling; "
                         "try ~1-2x the checkpoint's median nearest-neighbor spacing if the mesh looks "
                         "noisy/spiky.")
    # AGENT HOOKS (see module docstring). --use_agents is OFF by default: every flag
    # above (SRT/gimbal, mesh export, quality knobs) behaves exactly as it did before
    # this addition unless it's passed.
    ap.add_argument("--use_agents", action="store_true",
                    help="Enable the Frame Agent (frame selection before training) and Training "
                         "Control Agent (held-out-SSIM-driven early stop / densification freeze "
                         "during training). Uses the deterministic mock agents by default -- see "
                         "--use_llm. NOT executed on a real GPU anywhere in this repo yet; see "
                         "docs/AGENTIC_ARCHITECTURE.md before trusting it on a real run.")
    ap.add_argument("--use_llm", action="store_true",
                    help="Only relevant with --use_agents: use real Claude-backed agents instead "
                         "of the offline deterministic mock agents. Requires network + "
                         "ANTHROPIC_API_KEY and `pip install -r requirements-agents.txt`.")
    ap.add_argument("--agent_model", type=str, default="claude-sonnet-5",
                    help="Only relevant with --use_agents --use_llm. Check docs.claude.com for "
                         "the current recommended model string before a real run.")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "train_gpu.py requires a CUDA device -- run in Environment B (Colab GPU)"
    device = "cuda"
    # Explicit, unambiguous device report -- added after a real question ("am I actually on
    # GPU? I didn't see usage") that the 30ms/iteration-at-225k-Gaussians throughput answered
    # correctly, but only by inference. Never leave that to inference again.
    print(f"Device: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total memory)")

    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

    # DENSE SEEDING CONFIG (added after the first two real GPU runs both converged to a flat,
    # nearly featureless render -- see chat history). `PipelineConfig`/`RealPipelineConfig`'s
    # DEFAULT voxel_size=2.0 / confidence gating were tuned for the CPU sandbox's OWN goal --
    # a small, individually-trustworthy POINT CLOUD suitable as a standalone deliverable and
    # for point-to-point geometric metrics. That goal is actively wrong for seeding GPU
    # Gaussian training, which needs many candidate primitives (even mediocre ones) for the
    # optimizer to refine, split, or prune -- exactly the density vs. reliability tradeoff a
    # 200k-300k-Gaussian FastGS scene resolves by starting dense (COLMAP SfM, thousands of
    # points per image) and letting training sort out what survives. `finer_voxel_size` below
    # gives more DISTINCT starting points from the same input frames; `initialize_from_fused_points`
    # is now called on `result.points`/`result.confidence` (the FULL, pre-gating fused set) rather
    # than `result.final_positions` (the aggressively pruned CPU-sandbox deliverable) -- confidence
    # still controls initial OPACITY (see gaussian_model.py) and still gates densification/pruning
    # during GPU training itself; it just no longer gates whether a point gets to exist as a seed
    # at all.
    dense_seed_kwargs = dict(voxel_size=0.6, min_consistency=0.10)
    frame_agent, training_agent = _build_layer4_agents(args.use_agents, args.use_llm, args.agent_model)

    if args.source == "synthetic":
        from layer1_cpu_sandbox.synthetic.dataset_builder import build_dataset
        from layer1_cpu_sandbox.pipeline import run_pipeline, PipelineConfig
        dataset = build_dataset(seed=args.seed)
        pipeline_kwargs = dict(dense_seed_kwargs)
        if frame_agent is not None:
            pipeline_kwargs.update(_frame_agent_decision_kwargs(frame_agent, dataset.frames, args.out_dir))
        result = run_pipeline(dataset, PipelineConfig(**pipeline_kwargs))
        cameras = build_cameras_from_synthetic(dataset, result)
    else:
        assert args.video and args.log, "--video and --log are required for --source real"
        from layer1_cpu_sandbox.real_data.real_dataset_builder import build_real_dataset
        from layer1_cpu_sandbox.real_data.real_pipeline import run_real_pipeline, RealPipelineConfig
        dataset = build_real_dataset(args.video, args.log, camera_hfov_deg=args.camera_hfov_deg,
                                      target_fps=args.target_fps, max_frames=args.max_frames,
                                      resize_to=(args.width, args.height),
                                      position_smoothing_window=args.position_smoothing_window,
                                      assumed_gimbal_pitch_deg=args.assumed_gimbal_pitch_deg,
                                      assumed_gimbal_roll_deg=args.assumed_gimbal_roll_deg)
        dataset, rejected_zoom_frames = filter_stable_focal_length(dataset, args.max_focal_drift_frac)
        if rejected_zoom_frames:
            print(f"Excluded {len(rejected_zoom_frames)} frames outside the longest stable-focal-length run "
                  f"(zoom lens changing focal length mid-clip -- see --max_focal_drift_frac): "
                  f"{rejected_zoom_frames}")
        if args.pose_source == "visual_odometry":
            from layer1_cpu_sandbox.real_data.visual_odometry import (
                VisualOdometryError, replace_with_visual_odometry,
            )
            try:
                dataset, vo_stats = replace_with_visual_odometry(
                    dataset, min_matches=args.vo_min_matches, min_inliers=args.vo_min_inliers,
                    max_frame_gap=args.vo_max_frame_gap)
            except VisualOdometryError as e:
                raise RuntimeError(
                    "Visual odometry refused to create poses from this clip. Do not silently fall back to "
                    "the guessed SRT attitude: adjust --target_fps or use --pose_source telemetry only if you "
                    "have real gimbal yaw/pitch/roll from a DJI flight-record CSV.\n" + str(e)) from e
            print(f"[visual odometry] {vo_stats.n_pairs} pairs, {vo_stats.total_inliers} total inliers "
                  f"(median {vo_stats.median_inliers:.0f}/pair), telemetry-scaled path "
                  f"{vo_stats.telemetry_path_m:.2f} m; retained {vo_stats.n_frames}/"
                  f"{vo_stats.n_frames_input} frames (skipped {vo_stats.n_frames_skipped} weak frames)"
                  + (f"; {vo_stats.n_cross_zoom_pairs} pair(s) matched across two different focal "
                     f"lengths using each frame's own per-frame K" if vo_stats.n_cross_zoom_pairs else ""))
        dataset, rejected_orientation_frames = filter_downward_mapping_frames(dataset, args.max_forward_z)
        if rejected_orientation_frames:
            print(f"Excluded {len(rejected_orientation_frames)} non-downward frames from this ground/reef model: "
                  f"{rejected_orientation_frames}")
        pipeline_kwargs = dict(dense_seed_kwargs, keep_fraction=args.keep_fraction,
                               depth_window=args.depth_window, n_depths=args.n_depths,
                               depth_min=args.depth_min, depth_max=args.depth_max,
                               depth_spacing=args.depth_spacing,
                               depth_boundary_reject=args.depth_boundary_reject,
                               use_dynamic_filter=not args.no_dynamic_filter)
        if frame_agent is not None:
            pipeline_kwargs.update(_frame_agent_decision_kwargs(frame_agent, dataset.frames, args.out_dir))
        result = run_real_pipeline(dataset, RealPipelineConfig(**pipeline_kwargs))
        cameras = build_cameras_from_real(dataset, result)

    n_pre_gating = result.points.positions.shape[0]
    n_post_gating = result.final_positions.shape[0]
    seed_mask = ((result.points.observation_count >= args.seed_min_observations) &
                 (result.points.mean_consistency >= args.seed_min_consistency) &
                 (result.confidence.confidence >= args.seed_min_confidence))
    n_seed = int(seed_mask.sum())
    one_view = int((result.points.observation_count <= 1).sum())
    print(f"Loaded {len(cameras)} training cameras. Confidence-fused points: {n_pre_gating} raw, "
          f"{n_post_gating} post-gating, {n_seed} evidence-backed Gaussian seeds "
          f"(rejected {one_view} single-view points; obs>={args.seed_min_observations}, "
          f"consistency>={args.seed_min_consistency}, confidence>={args.seed_min_confidence}).")
    if n_seed < args.min_seed_points:
        raise RuntimeError(
            f"Only {n_seed} evidence-backed seeds remain (need {args.min_seed_points}). Refusing to train "
            "from single-view depth guesses because that produces a camera-frustum/pyramid instead of scene "
            "geometry. Improve pose/depth evidence or deliberately lower one seed threshold after inspecting "
            "the printed point counts.")

    model = GaussianModel(GaussianTrainingConfig(sh_degree=args.sh_degree)).to(device)
    model.initialize_from_fused_points(
        torch.from_numpy(result.points.positions[seed_mask]), torch.from_numpy(result.points.colors[seed_mask]),
        torch.from_numpy(result.confidence.confidence[seed_mask]), device=device)

    cfg = TrainConfig(max_iterations=args.max_iterations, out_dir=args.out_dir,
                       densify_until_iter=min(args.densify_until_iter, args.max_iterations - 500),
                       confidence_densify_above=args.confidence_densify_above,
                       n_eval_views=args.n_eval_views,
                       min_densify_observations=args.min_densify_observations,
                       max_gaussians=args.max_gaussians,
                       split_scale_thresh=args.split_scale_thresh,
                       opacity_reset_interval=args.opacity_reset_interval,
                       opacity_reset_target=args.opacity_reset_target,
                       max_world_scale_frac=args.max_world_scale_frac,
                       holdout_fraction=args.holdout_fraction,
                       export_mesh=args.export_mesh, mesh_min_opacity=args.mesh_min_opacity,
                       mesh_min_confidence=args.mesh_min_confidence, mesh_method=args.mesh_method,
                       mesh_poisson_depth=args.mesh_poisson_depth,
                       mesh_density_trim_quantile=args.mesh_density_trim_quantile,
                       mesh_use_covariance_normals=args.mesh_use_covariance_normals,
                       mesh_floater_filter=args.mesh_floater_filter,
                       mesh_floater_anisotropy_below=args.mesh_floater_anisotropy_below,
                       mesh_floater_scale_percentile=args.mesh_floater_scale_percentile,
                       mesh_min_component_vertices=args.mesh_min_component_vertices,
                       mesh_keep_top_k_components=args.mesh_keep_top_k_components,
                       mesh_voxel_size=args.mesh_voxel_size,
                       mesh_camera_hfov_deg=args.camera_hfov_deg,
                       mesh_frame_width=args.width, mesh_frame_height=args.height,
                       use_agents=args.use_agents)
    train(model, cameras, cfg, device=device, training_agent=training_agent)

    # Post-training geometry pass: preserves the user-facing train command while
    # creating the stronger, independently estimated VGGT + Trim001/Clean500
    # delivery under the same run directory as the Gaussian renders. Free the
    # training tensors before loading VGGT on a Colab T4.
    if args.source == "real" and args.auto_vggt_mesh:
        print("\nGaussian training completed. Starting the automatic VGGT -> Trim001/Clean500 pass...", flush=True)
        postprocess_script = os.path.join(os.path.dirname(__file__), "run_vggt_trim001_clean500.py")
        postprocess_cmd = [
            sys.executable, postprocess_script,
            "--video", args.video, "--log", args.log,
            "--out_dir", args.out_dir,
            "--target_fps", str(args.target_fps),
            "--max_candidate_frames", str(args.max_frames),
            "--max_vggt_frames", str(args.vggt_max_frames),
            "--camera_hfov_deg", str(args.camera_hfov_deg),
            "--assumed_gimbal_pitch_deg", str(args.assumed_gimbal_pitch_deg),
        ]
        del model, cameras, result, dataset
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        post_status_path = os.path.join(args.out_dir, "vggt_postprocess_status.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(post_status_path, "w", encoding="utf-8") as status_file:
            json.dump({"status": "running", "command": postprocess_cmd}, status_file, indent=2)
        try:
            subprocess.run(postprocess_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            with open(post_status_path, "w", encoding="utf-8") as status_file:
                json.dump({"status": "failed", "return_code": exc.returncode,
                           "manual_retry": postprocess_cmd}, status_file, indent=2)
            print(f"Gaussian training is complete, but VGGT postprocessing failed (exit {exc.returncode}). "
                  f"See {post_status_path} and rerun the saved command after resolving the reported issue.", flush=True)
            raise
        with open(post_status_path, "w", encoding="utf-8") as status_file:
            json.dump({"status": "complete", "output_dir": os.path.join(args.out_dir, "vggt_trim001_clean500")},
                      status_file, indent=2)
        print(f"Automatic VGGT postprocess complete. Status: {post_status_path}", flush=True)


if __name__ == "__main__":
    main()
