"""
gaussian_model.py
LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, NOT GPU-executed
(needs a CUDA device + the external `diff-gaussian-rasterization` package,
neither available in Environment A). Structurally reviewed and syntax-checked
here; must be validated in Environment B before trusting any output.

The real (not simplified/2D) 3D Gaussian parameterization: position, scale,
rotation, opacity, spherical-harmonic color -- the same PARAMETER SHAPES
every 3DGS-family method uses, because that shape is dictated by the
external `diff-gaussian-rasterization` package's public API (position (N,3),
scale (N,3), rotation quaternion (N,4), opacity (N,1), SH coefficients
(N,K,3)), not by any one implementation's design choice. This file's
STRUCTURE, densify/prune/clone/split logic, and confidence integration are
original -- written fresh for this project, not copied from the user's own
prior FastGS-QADS project (see confidence_propagation.py's module docstring
for exactly which mechanisms were design-inspired by it and how).

Confidence integration point: `GaussianModel` owns one
`GaussianConfidenceStats` (reused directly from confidence_gaussian_model.py
-- our own code) and one `ObservationConfidencePropagator`
(confidence_propagation.py -- our own code) per instance, kept index-aligned
with the Gaussian arrays across every densify/prune/clone/split call. This
is the piece that turns "a differentiable Gaussian renderer" into "OUR
confidence-aware Gaussian renderer".
"""
from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn

from confidence_gaussian_model import GaussianConfidenceStats, blended_confidence, gate_densify_and_prune
from confidence_propagation import ObservationConfidencePropagator, PropagationConfig


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def _quat_identity(n: int, device="cpu") -> torch.Tensor:
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0  # (w, x, y, z), identity rotation
    return q


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotates vectors `v` (N,3) by unit quaternions `q` (N,4) in (w,x,y,z) order."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    qv = torch.stack([x, y, z], dim=-1)
    uv = torch.cross(qv, v, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return v + 2 * (w.unsqueeze(-1) * uv + uuv)


def _resize_confidence_stats(stats: GaussianConfidenceStats, keep_mask, n_new: int) -> GaussianConfidenceStats:
    """Resizes a GaussianConfidenceStats' buffers to match a prune (keep_mask) and/or
    append (n_new) -- it's a plain accumulator struct with no resize logic of its own,
    so GaussianModel (which owns the index bookkeeping) does it here."""
    fields = ["obs_count", "angle_min", "angle_max", "consistency_sum", "quality_sum", "pose_conf_sum", "n_accum"]
    device = stats.obs_count.device
    if keep_mask is not None:
        for f in fields:
            setattr(stats, f, getattr(stats, f)[keep_mask])
    if n_new > 0:
        for f in fields:
            fill = float("inf") if f == "angle_min" else (float("-inf") if f == "angle_max" else 0.0)
            pad = torch.full((n_new,), fill, device=device)
            setattr(stats, f, torch.cat([getattr(stats, f), pad]))
    return stats


@dataclasses.dataclass
class GaussianTrainingConfig:
    sh_degree: int = 0                  # 0 = flat color only; raise once basic training is validated
    position_lr: float = 1.6e-4
    feature_dc_lr: float = 2.5e-3
    feature_rest_lr: float = 2.5e-3 / 20.0
    opacity_lr: float = 5e-2
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    opacity_init: float = 0.5
    scale_init_from_nn_dist: bool = True   # initial scale ~ distance to nearest other Gaussian


class GaussianModel(nn.Module):
    def __init__(self, cfg: GaussianTrainingConfig = None):
        super().__init__()
        self.cfg = cfg or GaussianTrainingConfig()
        n_sh = (self.cfg.sh_degree + 1) ** 2
        self._n_sh = n_sh
        # Empty until initialize_from_fused_points(); this is intentional -- a GaussianModel
        # with zero Gaussians is a valid, checkable state (see tests/test_layer3_gaussian_model.py).
        self.xyz = nn.Parameter(torch.zeros(0, 3))
        self.features_dc = nn.Parameter(torch.zeros(0, 1, 3))
        self.features_rest = nn.Parameter(torch.zeros(0, max(n_sh - 1, 0), 3))
        self.log_scaling = nn.Parameter(torch.zeros(0, 3))
        self.rotation = nn.Parameter(torch.zeros(0, 4))
        self.opacity_logit = nn.Parameter(torch.zeros(0, 1))

        self.confidence_stats = GaussianConfidenceStats(0)
        self.propagator = ObservationConfidencePropagator(0, PropagationConfig())
        self.register_buffer("prior_confidence", torch.zeros(0))
        self.optimizer: torch.optim.Adam = None
        self._grad_accum = torch.zeros(0)
        self._grad_denom = torch.zeros(0)

    # ------------------------------------------------------------- shape props --
    @property
    def n_gaussians(self) -> int:
        return self.xyz.shape[0]

    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.log_scaling)

    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.rotation, dim=-1)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    def get_features(self) -> torch.Tensor:
        return torch.cat([self.features_dc, self.features_rest], dim=1)

    def confidence(self, online_weight: float = 0.5) -> torch.Tensor:
        """Same blended-confidence logic as ConfidenceAwareGaussianModel (see
        blended_confidence() in confidence_gaussian_model.py) -- shared, not
        duplicated, so both models can never silently drift apart."""
        return blended_confidence(self.confidence_stats, self.prior_confidence, online_weight)

    def densify_and_prune(self, grad_accum: torch.Tensor, opacity_thresh: float = 0.05,
                           confidence_prune_below: float = 0.12, grad_thresh: float = 0.0002,
                           confidence_densify_above: float = 0.60,
                           propagated_confidence: torch.Tensor = None,
                           optimizer_struggle_signal: torch.Tensor = None,
                           struggle_thresh: float = 3.0,
                           extent: float = None, max_world_scale_frac: float = 0.1,
                           max_floater_frac: float = 0.2):
        """Same three-signal control logic as ConfidenceAwareGaussianModel -- see
        gate_densify_and_prune()'s docstring in confidence_gaussian_model.py for
        the full design rationale. Returns (prune_mask, densify_mask, confidence).

        extent / max_world_scale_frac: FastGS_QADS's `densify_and_prune_fastgs`
        (and stock 3DGS) also unconditionally prune any Gaussian whose
        world-space scale exceeds max_world_scale_frac * extent -- their
        big_points_ws check. This model had no equivalent: a large, badly
        triangulated "floater" Gaussian can hold perfectly reasonable opacity
        and confidence (it's genuinely reducing loss on the training views that
        happen to see it edge-on or averaged-over) while still being
        geometrically wrong, and shows up as a streak/smudge specifically from
        held-out angles that see it face-on instead. Confidence/opacity gating
        alone will never catch that. Pass extent=None to disable.

        SAFETY (added after a run this destroyed): `extent` must describe the
        reconstructed SCENE, not the camera trajectory -- pass the seeded/current
        Gaussian cloud's own bounding-box diagonal, NOT a camera-position-based
        extent. Camera-position spread is fine for clone_and_split's
        split-vs-clone threshold (being off just shifts which branch a candidate
        takes), but is not safe here, where being off deletes Gaussians outright.
        For a single-pass/narrow-baseline flight the camera trajectory can be much
        smaller than the ground footprint it's actually photographing, which
        silently produces exactly that failure. As a hard backstop regardless of
        what extent turns out to be: if this check would flag more than
        max_floater_frac of the population in one pass, that's it being
        miscalibrated, not a real floater problem (real floaters are a small
        minority) -- skipped for that pass with a printed warning instead of
        applied.
        """
        conf = self.confidence()
        opacity = self.get_opacity().squeeze(-1)
        prune_mask, densify_mask = gate_densify_and_prune(
            conf, opacity, grad_accum, opacity_thresh, confidence_prune_below, grad_thresh,
            confidence_densify_above, propagated_confidence, optimizer_struggle_signal, struggle_thresh)
        if extent is not None:
            big_points_ws = self.get_scaling().max(dim=-1).values > max_world_scale_frac * extent
            flagged_frac = float(big_points_ws.float().mean()) if big_points_ws.numel() else 0.0
            if flagged_frac > max_floater_frac:
                print(f"      [floater_prune] SKIPPED: would flag {flagged_frac:.1%} of "
                      f"{self.n_gaussians} Gaussians as oversized (cap is {max_floater_frac:.0%}) -- "
                      f"extent={extent:.4g} is almost certainly miscalibrated for this scene, not a "
                      f"real floater problem. Not applying this pass; fix the extent being passed in "
                      f"rather than raising this cap.")
            else:
                prune_mask = prune_mask | big_points_ws
                densify_mask = densify_mask & ~big_points_ws  # never split/clone something just condemned
        return prune_mask, densify_mask, conf

    def reset_opacity(self, target: float = 0.05):
        """FastGS_QADS-style periodic opacity reset (their arguments.py:
        opacity_reset_interval=3000; Kerbl et al. 2023 reset_opacity()) -- NOT part
        of this project's original design (see train_gpu.py's module docstring on
        WHAT THIS DELIBERATELY DOES NOT DO), added as an opt-in knob (TrainConfig.
        opacity_reset_interval, default 0/off) for runs where held-out quality
        matters more than keeping the confidence-gating ablation clean.

        Clamps every Gaussian's opacity down to at most `target` and -- matching
        the reference implementation, not just approximating it -- zeroes Adam's
        momentum for the opacity parameter specifically via `_swap_optimizer_param`'s
        existing resize_fn hook. Skipping that second part would leave stale
        momentum fighting the sudden value change, so the reset would barely move
        anything for several iterations. Gaussians that don't deserve their opacity
        back (i.e. aren't actually reducing loss where they sit) drift back toward
        0 and get caught by the next opacity_thresh prune; Gaussians that DO deserve
        it climb back up under the rendering loss's own gradient. This is what
        actually holds density accountable over time -- confidence-gated pruning
        alone only screens a Gaussian once, at birth.
        """
        with torch.no_grad():
            new_opacity = torch.clamp(self.get_opacity(), min=1e-4, max=target)
            new_logit = _inverse_sigmoid(new_opacity)
        self.opacity_logit = self._swap_optimizer_param("opacity", new_logit, resize_fn=torch.zeros_like)

    # ---------------------------------------------------------------- init --
    def initialize_from_fused_points(self, positions: torch.Tensor, colors: torch.Tensor,
                                      confidence: torch.Tensor, device: str = "cuda"):
        """
        Bridges Layer 1's confidence-fused point cloud (CPU sandbox's
        `PrototypePointRepresentation` + `confidence.py`, or the real-data
        equivalent in `real_pipeline.py`) into this Gaussian model: one
        Gaussian per fused point, seeded with that point's own observation
        confidence as `prior_confidence` (this is the direct continuation of
        the "observation-aware" thesis into Layer 3 -- a point the CPU
        pipeline was already unsure about starts life as a Gaussian the GPU
        pipeline is ALSO unsure about, rather than every Gaussian starting
        from a blank slate).

        positions: (N,3) float, colors: (N,3) float in [0,1], confidence: (N,) float in [0,1].
        """
        n = positions.shape[0]
        positions = positions.to(device).float()
        colors = colors.to(device).float()
        confidence = confidence.to(device).float()

        self.xyz = nn.Parameter(positions.clone())
        # SH DC term: standard 3DGS convention is (color - 0.5) / SH_C0, SH_C0 = 0.28209479177387814
        sh_c0 = 0.28209479177387814
        dc = ((colors - 0.5) / sh_c0).unsqueeze(1)  # (N,1,3)
        self.features_dc = nn.Parameter(dc)
        self.features_rest = nn.Parameter(torch.zeros(n, max(self._n_sh - 1, 0), 3, device=device))

        if self.cfg.scale_init_from_nn_dist and n > 1:
            with torch.no_grad():
                # Nearest-neighbor distance via a KD-tree (O(N log N), O(N) memory), NOT
                # torch.cdist (O(N^2) memory -- fine at the few-thousand-point scale this
                # was originally tested at, but a real problem once dense seeding (see
                # train_gpu.py's main()) pushes N into the tens of thousands: a 30,000-point
                # cdist matrix alone is ~3.6GB in float32, before any of the actual training
                # tensors. Caught before it became a real Colab OOM, not after.
                from scipy.spatial import cKDTree
                pos_np = positions.detach().cpu().numpy()
                tree = cKDTree(pos_np)
                dists, _ = tree.query(pos_np, k=2)  # k=1 is the point itself (distance 0)
                nn_dist = torch.from_numpy(dists[:, 1]).float().to(device).clamp(min=1e-6)
        else:
            nn_dist = torch.full((n,), 0.5, device=device)
        init_scale = torch.log(nn_dist).unsqueeze(-1).repeat(1, 3)
        self.log_scaling = nn.Parameter(init_scale)
        self.rotation = nn.Parameter(_quat_identity(n, device))

        # Lower-confidence points start at lower initial opacity: an unconfirmed observation
        # shouldn't immediately render as if it were solid, confident geometry.
        opacity_init = (self.cfg.opacity_init * (0.3 + 0.7 * confidence)).clamp(1e-4, 1 - 1e-4)
        self.opacity_logit = nn.Parameter(_inverse_sigmoid(opacity_init).unsqueeze(-1))

        self.confidence_stats = GaussianConfidenceStats(n, device=device)
        self.propagator = ObservationConfidencePropagator(n, PropagationConfig(), device=device)
        self.register_buffer("prior_confidence", confidence.clone())
        self._grad_accum = torch.zeros(n, device=device)
        self._grad_denom = torch.zeros(n, device=device)

    # ------------------------------------------------------------ optimizer --
    def setup_optimizer(self):
        """Per-parameter-group learning rates -- standard 3DGS practice (position needs a much
        smaller LR than opacity/scale/rotation because it directly moves geometry); this
        grouping convention predates FastGS/FastGS-QADS and belongs to no one's IP."""
        groups = [
            {"params": [self.xyz], "lr": self.cfg.position_lr, "name": "xyz"},
            {"params": [self.features_dc], "lr": self.cfg.feature_dc_lr, "name": "features_dc"},
            {"params": [self.features_rest], "lr": self.cfg.feature_rest_lr, "name": "features_rest"},
            {"params": [self.opacity_logit], "lr": self.cfg.opacity_lr, "name": "opacity"},
            {"params": [self.log_scaling], "lr": self.cfg.scaling_lr, "name": "scaling"},
            {"params": [self.rotation], "lr": self.cfg.rotation_lr, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        return self.optimizer

    # ------------------------------------------------------- gradient stats --
    def accumulate_view_gradient(self, screenspace_points: torch.Tensor, visibility_mask: torch.Tensor):
        """Call once per training iteration, right after `loss.backward()`, with the rasterizer's
        screen-space position tensor (its `.grad` is the standard 3DGS view-space-gradient
        densification signal) and the per-Gaussian visibility mask this view produced."""
        grad_norm = torch.norm(screenspace_points.grad[visibility_mask, :2], dim=-1)
        self._grad_accum[visibility_mask] += grad_norm
        self._grad_denom[visibility_mask] += 1

    def mean_view_gradient(self) -> torch.Tensor:
        denom_safe = self._grad_denom.clamp(min=1)
        return self._grad_accum / denom_safe

    def _swap_optimizer_param(self, name: str, new_tensor: torch.Tensor, resize_fn=None) -> nn.Parameter:
        """
        Replaces the `name`d parameter group's tensor with `new_tensor`, AS A NEW
        `nn.Parameter` OBJECT, and -- critically -- updates `group['params'][0]` to
        point at that new object too, transferring (and resizing, via `resize_fn`)
        Adam's `exp_avg`/`exp_avg_sq`/`step` state from the old object to the new one.

        Why this exists: reassigning `self.xyz = nn.Parameter(...)` alone leaves the
        optimizer's param_groups referencing the OLD Parameter object -- Adam would
        then keep computing updates for a stale, wrong-sized tensor that is no longer
        connected to the model at all, while `self.xyz` silently never gets optimized
        again after the first prune/densify call. This is a real, easy-to-miss bug
        class in any dynamic-parameter-count optimizer setup (caught here by actually
        running it on CPU, not just reading the code -- see
        tests/test_layer3_gaussian_model.py::test_optimizer_state_survives_prune).
        """
        new_param = nn.Parameter(new_tensor)
        if self.optimizer is None:
            return new_param
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            old_param = group["params"][0]
            if old_param in self.optimizer.state:
                old_state = self.optimizer.state.pop(old_param)
                new_state = {"step": old_state.get("step", torch.tensor(0.0))}
                if resize_fn is not None:
                    if "exp_avg" in old_state:
                        new_state["exp_avg"] = resize_fn(old_state["exp_avg"])
                    if "exp_avg_sq" in old_state:
                        new_state["exp_avg_sq"] = resize_fn(old_state["exp_avg_sq"])
                else:
                    new_state.update({k: v for k, v in old_state.items() if k != "step"})
                self.optimizer.state[new_param] = new_state
            group["params"][0] = new_param
            break
        return new_param

    # ------------------------------------------------- densify / prune / clone / split --
    def _resize_all_buffers(self, keep_mask: torch.Tensor = None, n_new: int = 0,
                             new_confidence: torch.Tensor = None):
        if keep_mask is not None:
            self.prior_confidence = self.prior_confidence[keep_mask]
            self._grad_accum = self._grad_accum[keep_mask]
            self._grad_denom = self._grad_denom[keep_mask]
        if n_new > 0:
            nc = new_confidence if new_confidence is not None else \
                torch.zeros(n_new, device=self.prior_confidence.device)
            self.prior_confidence = torch.cat([self.prior_confidence, nc])
            self._grad_accum = torch.cat([self._grad_accum, torch.zeros(n_new, device=self._grad_accum.device)])
            self._grad_denom = torch.cat([self._grad_denom, torch.zeros(n_new, device=self._grad_denom.device)])
        self.confidence_stats = _resize_confidence_stats(self.confidence_stats, keep_mask, n_new)
        self.propagator.resize(keep_mask=keep_mask, n_new=n_new, new_confidence=new_confidence)

    def prune_points(self, prune_mask: torch.Tensor):
        """Removes Gaussians in `prune_mask` from every parameter and every confidence buffer
        in lock-step, correctly re-pointing the optimizer at each new parameter object (see
        `_swap_optimizer_param`) so training can continue seamlessly after this call."""
        keep_mask = ~prune_mask
        self.xyz = self._swap_optimizer_param("xyz", self.xyz.detach()[keep_mask], lambda t: t[keep_mask])
        self.features_dc = self._swap_optimizer_param("features_dc", self.features_dc.detach()[keep_mask],
                                                        lambda t: t[keep_mask])
        self.features_rest = self._swap_optimizer_param("features_rest", self.features_rest.detach()[keep_mask],
                                                          lambda t: t[keep_mask])
        self.log_scaling = self._swap_optimizer_param("scaling", self.log_scaling.detach()[keep_mask],
                                                        lambda t: t[keep_mask])
        self.rotation = self._swap_optimizer_param("rotation", self.rotation.detach()[keep_mask],
                                                     lambda t: t[keep_mask])
        self.opacity_logit = self._swap_optimizer_param("opacity", self.opacity_logit.detach()[keep_mask],
                                                          lambda t: t[keep_mask])
        self._resize_all_buffers(keep_mask=keep_mask)

    def _append_new_gaussians(self, new_xyz, new_dc, new_rest, new_log_scale, new_rot, new_opacity_logit,
                               new_confidence):
        n_new = new_xyz.shape[0]
        pad = lambda t: torch.cat([t, torch.zeros((n_new,) + tuple(t.shape[1:]), device=t.device)])
        self.xyz = self._swap_optimizer_param("xyz", torch.cat([self.xyz.detach(), new_xyz]), pad)
        self.features_dc = self._swap_optimizer_param("features_dc",
                                                        torch.cat([self.features_dc.detach(), new_dc]), pad)
        self.features_rest = self._swap_optimizer_param("features_rest",
                                                          torch.cat([self.features_rest.detach(), new_rest]), pad)
        self.log_scaling = self._swap_optimizer_param("scaling",
                                                        torch.cat([self.log_scaling.detach(), new_log_scale]), pad)
        self.rotation = self._swap_optimizer_param("rotation", torch.cat([self.rotation.detach(), new_rot]), pad)
        self.opacity_logit = self._swap_optimizer_param(
            "opacity", torch.cat([self.opacity_logit.detach(), new_opacity_logit]), pad)
        self._resize_all_buffers(n_new=n_new, new_confidence=new_confidence)

    def clone_and_split(self, densify_mask: torch.Tensor, split_scale_thresh: float, extent: float,
                         split_ratio: float = 1.6):
        """
        Standard 3DGS clone-vs-split branching (small Gaussians get duplicated in place;
        large ones get split into two smaller ones sampled from their own covariance) --
        this specific branching rule predates FastGS/FastGS-QADS (it's from the original
        3DGS paper, Kerbl et al. 2023) and is reimplemented fresh here, not copied from
        the user's fork of it.

        CONFIDENCE-AWARE PART (original to this project): every new Gaussian created here
        inherits its PARENT's propagated observation confidence directly, rather than
        starting at `new_gaussian_confidence` (0.0) like a genuinely new/relocated
        Gaussian would (see confidence_propagation.py). Rationale: a clone/split is not
        new evidence -- it's the SAME evidence covering a smaller region -- so it should
        not have to re-earn observation confidence its parent already has.
        """
        if densify_mask.sum() == 0:
            return
        idx = torch.nonzero(densify_mask, as_tuple=True)[0]
        scales = self.get_scaling()[idx].max(dim=1).values
        is_split = scales > split_scale_thresh * extent
        is_clone = ~is_split

        clone_idx = idx[is_clone]
        if clone_idx.numel() > 0:
            self._append_new_gaussians(
                self.xyz.detach()[clone_idx], self.features_dc.detach()[clone_idx],
                self.features_rest.detach()[clone_idx], self.log_scaling.detach()[clone_idx],
                self.rotation.detach()[clone_idx], self.opacity_logit.detach()[clone_idx],
                self.propagator.propagated.detach()[clone_idx],
            )

        # split_idx values are all < the PRE-clone-append count, and appends only add rows at
        # the end, so they stay valid indices into the (now clone-extended) parameter arrays.
        split_idx = idx[is_split]
        if split_idx.numel() > 0:
            n_split = split_idx.numel()
            parent_scale = self.get_scaling()[split_idx]
            parent_rot = self.get_rotation()[split_idx]
            noise = torch.randn(n_split, 3, device=self.xyz.device) * parent_scale
            offset = _quat_rotate(parent_rot, noise)
            new_xyz = torch.cat([self.xyz.detach()[split_idx] + offset,
                                  self.xyz.detach()[split_idx] - offset])
            new_log_scale = (self.log_scaling.detach()[split_idx] - math.log(split_ratio)).repeat(2, 1)
            self._append_new_gaussians(
                new_xyz, self.features_dc.detach()[split_idx].repeat(2, 1, 1),
                self.features_rest.detach()[split_idx].repeat(2, 1, 1), new_log_scale,
                self.rotation.detach()[split_idx].repeat(2, 1), self.opacity_logit.detach()[split_idx].repeat(2, 1),
                self.propagator.propagated.detach()[split_idx].repeat(2),
            )
            # Now remove the original (pre-split) parent rows -- `self.n_gaussians` is evaluated
            # AFTER the two appends above, so the mask is sized to the current array length; the
            # True positions (split_idx, all < pre-clone-append count) are unaffected by either
            # append and remain the correct rows to delete.
            parent_removal_mask = torch.zeros(self.n_gaussians, dtype=torch.bool, device=self.xyz.device)
            parent_removal_mask[split_idx] = True
            self.prune_points(parent_removal_mask)
