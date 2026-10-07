"""
confidence_gaussian_model.py
LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, CPU-TESTED (see
__main__ block and tests/test_layer3_confidence_gaussian_cpu.py); NOT
GPU-executed (no CUDA device in Environment A). The production path
(full 3-D projection + tile rasterization) is
`layer3_gpu/csrc/confidence_gaussian/` (CUDA) + the external
`diff-gaussian-rasterization` extension (see docs/ARCHITECTURE.md) and has
NOT been compiled or run anywhere yet -- that must happen in Environment B.

WHAT IS ACTUALLY VALIDATED HERE, ON CPU, WITH REAL TORCH EXECUTION:
  1. `GaussianConfidenceStats.confidence()` reproduces the exact formula in
     layer1_cpu_sandbox/reconstruction/confidence.py (weighted geometric
     mean of 5 evidence signals, single-observation cap), but implemented
     with torch scatter/index_add ops suited to a per-iteration training
     loop instead of a one-shot pandas groupby.
  2. `ConfidenceAwareGaussianModel.densify_and_prune()` -- the actual novel
     control logic -- correctly identifies confidence-driven prune/densify
     candidates on a toy Gaussian set with hand-constructed statistics.
  3. `SimplifiedDifferentiableSplatterCPU` -- an intentionally simplified
     2-D differentiable alpha-compositing splatter (NOT the real 3-D tile
     rasterizer -- that would be reimplementing an already-solved, highly
     optimized part of 3DGS/FastGS, which is explicitly out of scope; see
     docs/ARCHITECTURE.md) -- lets us run real gradient descent on
     positions/opacity/color/scale against a toy target image and confirm
     loss actually decreases, i.e. the parameterization and confidence
     bookkeeping survive an actual backward pass, on CPU, today.

The production GPU renderer is swapped in at exactly one call site
(`self.renderer.render(...)` in `ConfidenceAwareGaussianModel.render`);
everything else in this file is renderer-agnostic.
"""
from __future__ import annotations

import math
import dataclasses

import torch
import torch.nn as nn

DEFAULT_WEIGHTS = dict(obs=0.30, angle=0.20, consistency=0.25, quality=0.15, pose=0.10)


class GaussianConfidenceStats:
    """Per-Gaussian running accumulators for the identical 5 evidence signals used in
    layer1_cpu_sandbox/reconstruction/confidence.py. Plain tensors (not nn.Parameters):
    these are observation STATISTICS, accumulated once per training-view pass via
    scatter ops keyed by which Gaussians contributed to that view's render, not
    learned via backprop."""

    def __init__(self, n: int, device="cpu"):
        self.obs_count = torch.zeros(n, device=device)
        self.angle_min = torch.full((n,), float("inf"), device=device)
        self.angle_max = torch.full((n,), float("-inf"), device=device)
        self.consistency_sum = torch.zeros(n, device=device)
        self.quality_sum = torch.zeros(n, device=device)
        self.pose_conf_sum = torch.zeros(n, device=device)
        self.n_accum = torch.zeros(n, device=device)

    @torch.no_grad()
    def update(self, idx: torch.Tensor, view_angle: torch.Tensor, consistency: torch.Tensor,
               quality: float, pose_conf: float):
        """idx: (K,) long -- indices of Gaussians touched by the current view.
        view_angle/consistency: (K,) float, per-touched-Gaussian for this view."""
        ones = torch.ones_like(idx, dtype=torch.float32)
        self.obs_count.index_add_(0, idx, ones)
        self.angle_min.scatter_reduce_(0, idx, view_angle, reduce="amin")
        self.angle_max.scatter_reduce_(0, idx, view_angle, reduce="amax")
        self.consistency_sum.index_add_(0, idx, consistency)
        self.quality_sum.index_add_(0, idx, ones * quality)
        self.pose_conf_sum.index_add_(0, idx, ones * pose_conf)
        self.n_accum.index_add_(0, idx, ones)

    def confidence(self, weights: dict = None, obs_tau: float = 1.5,
                   target_angle_spread_rad: float = math.radians(6.0),
                   single_view_cap: float = 0.55) -> torch.Tensor:
        w = weights or DEFAULT_WEIGHTS
        eps = 1e-6
        obs_score = 1.0 - torch.exp(-(self.obs_count - 1).clamp(min=0) / obs_tau)
        spread = (self.angle_max - self.angle_min).clamp(min=0)
        spread = torch.where(torch.isfinite(spread), spread, torch.zeros_like(spread))
        angle_score = (spread / target_angle_spread_rad).clamp(0, 1)
        n_safe = self.n_accum.clamp(min=1)
        consistency_score = (self.consistency_sum / n_safe).clamp(0, 1)
        quality_score = (self.quality_sum / n_safe).clamp(0, 1)
        pose_score = (self.pose_conf_sum / n_safe).clamp(0, 1)

        log_conf = (w["obs"] * torch.log(obs_score + eps) + w["angle"] * torch.log(angle_score + eps) +
                    w["consistency"] * torch.log(consistency_score + eps) +
                    w["quality"] * torch.log(quality_score + eps) +
                    w["pose"] * torch.log(pose_score + eps))
        confidence = torch.exp(log_conf)

        single_view = self.obs_count <= 1
        confidence = torch.where(single_view, confidence.clamp(max=single_view_cap), confidence)
        return confidence


def blended_confidence(stats: GaussianConfidenceStats, prior_confidence: torch.Tensor,
                        online_weight: float = 0.5) -> torch.Tensor:
    """
    Shared confidence-blending logic used by BOTH `ConfidenceAwareGaussianModel`
    (the CPU-testable, simplified 2-D model below) and `gaussian_model.GaussianModel`
    (the production 3-D model, layer3_gpu/python/gaussian_model.py). Extracted here
    rather than duplicated a third time -- this project already tracks "three
    independent copies of the same confidence formula" as a maintenance risk (see
    the CUDA kernel's header comment and docs/STOP_CONDITIONS.md); this at least
    keeps it at two (Python, CUDA) instead of three.

    Blends the confidence carried over from CPU-sandbox-style fusion at
    initialization with what's been accumulated online during GPU optimization
    (online_weight=0 => trust only the initialization prior; 1 => trust only
    what's been observed during this optimization run).
    """
    online = stats.confidence()
    has_obs = stats.n_accum > 0
    return torch.where(has_obs, online_weight * online + (1 - online_weight) * prior_confidence, prior_confidence)


def gate_densify_and_prune(confidence: torch.Tensor, opacity: torch.Tensor, grad_accum: torch.Tensor,
                            opacity_thresh: float = 0.05, confidence_prune_below: float = 0.12,
                            grad_thresh: float = 0.0002, confidence_densify_above: float = 0.60,
                            propagated_confidence: torch.Tensor = None,
                            optimizer_struggle_signal: torch.Tensor = None, struggle_thresh: float = 3.0):
    """
    THE core novel control logic (see layer2_interfaces/i_gaussian_refine.py for
    why this differs from standard 3DGS densification), shared by both Gaussian
    model classes (see `blended_confidence` above for why this is extracted).
    Three signals combine, each doing a distinct job -- see
    confidence_propagation.py's module docstring for the full design rationale
    and how signals 2-3 relate to (but are not copied from) the author's own
    prior FastGS-QADS project:

      1. `confidence` (this project's core signal, observation-coverage-based) --
         the HARD GATE. Nothing densifies below `confidence_densify_above` on
         this signal, however strong signals 2-3 look, because a struggling
         optimizer or noisy fit in a poorly-observed region is more likely
         "not enough real evidence" than "needs more capacity".
      2. `propagated_confidence` (optional, STCP-inspired temporal+spatial
         smoothing of signal 1 via ObservationConfidencePropagator) -- if
         supplied, THIS is what's actually compared to the thresholds instead
         of the raw per-round `confidence`, so a single noisy round can't flip
         a decision.
      3. `optimizer_struggle_signal` (optional, OSAD-inspired) -- ADDS densify
         candidates beyond the gradient criterion alone (mirrors how OSAD ORs
         with FastGS's gradient trigger), but ONLY among Gaussians that already
         pass the signal-1/2 gate -- it can never override it.
    """
    gate_conf = propagated_confidence if propagated_confidence is not None else confidence
    prune_mask = (opacity < opacity_thresh) | (gate_conf < confidence_prune_below)
    confidence_gate = (gate_conf >= confidence_densify_above) & (~prune_mask)

    densify_trigger = grad_accum > grad_thresh
    if optimizer_struggle_signal is not None:
        densify_trigger = densify_trigger | (optimizer_struggle_signal > struggle_thresh)

    densify_mask = densify_trigger & confidence_gate
    return prune_mask, densify_mask


class ConfidenceAwareGaussianModel(nn.Module):
    """
    3-D Gaussian parameters (production shape: position (N,3), anisotropic scale
    (N,3), rotation quaternion (N,4), opacity (N,1), SH color coefficients). For the
    CPU-testable path here, we keep a simplified 2-D-projected parameterization
    (position2d, isotropic scale, opacity, RGB, depth) sufficient to exercise
    `SimplifiedDifferentiableSplatterCPU`; the full 3-D parameterization and its real
    projection live in the CUDA extension (layer3_gpu/csrc/confidence_gaussian) and
    the external tile rasterizer, both used only in Environment B.
    """

    def __init__(self, init_positions2d: torch.Tensor, init_colors: torch.Tensor,
                 init_confidence: torch.Tensor = None):
        super().__init__()
        n = init_positions2d.shape[0]
        self.positions2d = nn.Parameter(init_positions2d.clone())
        self.log_scales = nn.Parameter(torch.full((n,), math.log(2.0)))
        self.opacity_logits = nn.Parameter(torch.zeros(n))
        self.colors = nn.Parameter(init_colors.clone())
        self.depth = nn.Parameter(torch.linspace(0.9, 1.1, n))
        self.stats = GaussianConfidenceStats(n)
        prior = init_confidence if init_confidence is not None else torch.full((n,), 0.5)
        self.register_buffer("prior_confidence", prior)

    def confidence(self, online_weight: float = 0.5) -> torch.Tensor:
        return blended_confidence(self.stats, self.prior_confidence, online_weight)

    def densify_and_prune(self, grad_accum: torch.Tensor, opacity_thresh: float = 0.05,
                           confidence_prune_below: float = 0.12, grad_thresh: float = 0.0002,
                           confidence_densify_above: float = 0.60,
                           propagated_confidence: torch.Tensor = None,
                           optimizer_struggle_signal: torch.Tensor = None,
                           struggle_thresh: float = 3.0):
        conf = self.confidence()
        opacity = torch.sigmoid(self.opacity_logits)
        prune_mask, densify_mask = gate_densify_and_prune(
            conf, opacity, grad_accum, opacity_thresh, confidence_prune_below, grad_thresh,
            confidence_densify_above, propagated_confidence, optimizer_struggle_signal, struggle_thresh)
        return prune_mask, densify_mask, conf

    def render(self, H: int, W: int, renderer=None):
        renderer = renderer or simplified_splat_cpu
        return renderer(self.positions2d, self.log_scales, self.opacity_logits, self.colors, self.depth, H, W)


def simplified_splat_cpu(positions2d: torch.Tensor, log_scales: torch.Tensor, opacity_logits: torch.Tensor,
                          colors: torch.Tensor, depth: torch.Tensor, H: int, W: int):
    """
    `SimplifiedDifferentiableSplatterCPU` -- back-to-front alpha-compositing of
    isotropic 2-D Gaussian kernels. NOT the production renderer (see module
    docstring). O(N * H * W) via a Python loop over Gaussians, which is fine for
    the toy N~O(100), H,W~O(32) correctness check this exists for, and would be
    unacceptably slow at real scene/image scale -- that performance gap is exactly
    why production uses the tile-based CUDA rasterizer instead.
    """
    device = positions2d.device
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                             torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    order = torch.argsort(depth, descending=True)  # back-to-front for the "over" compositing operator
    canvas = torch.zeros(H, W, 3, device=device)
    alpha_acc = torch.zeros(H, W, device=device)

    sigma = torch.exp(log_scales).clamp(min=0.6)
    opacity = torch.sigmoid(opacity_logits)

    for i in order:
        mu = positions2d[i]
        dx = (xs - mu[0]) / sigma[i]
        dy = (ys - mu[1]) / sigma[i]
        g = torch.exp(-0.5 * (dx ** 2 + dy ** 2))
        a = opacity[i] * g
        weight = (1 - alpha_acc) * a
        canvas = canvas + weight.unsqueeze(-1) * colors[i]
        alpha_acc = alpha_acc + weight
    return canvas, alpha_acc


if __name__ == "__main__":
    torch.manual_seed(0)
    H, W, N = 32, 32, 60

    # Toy target: a soft red blob left-of-center, a soft blue blob right-of-center.
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    target = torch.zeros(H, W, 3)
    for (cx, cy, col) in [(10, 16, torch.tensor([0.9, 0.15, 0.15])), (22, 16, torch.tensor([0.15, 0.2, 0.9]))]:
        g = torch.exp(-0.5 * (((xs - cx) / 4.0) ** 2 + ((ys - cy) / 4.0) ** 2))
        target += g.unsqueeze(-1) * col

    init_pos = torch.rand(N, 2) * torch.tensor([W, H])
    init_col = torch.rand(N, 3) * 0.3 + 0.3
    init_conf = torch.rand(N) * 0.5  # deliberately mediocre priors, to exercise gating below

    model = ConfidenceAwareGaussianModel(init_pos, init_col, init_conf)
    opt = torch.optim.Adam(model.parameters(), lr=0.15)

    print("Optimizing a toy scene with SimplifiedDifferentiableSplatterCPU (CPU, real torch execution)...")
    losses = []
    for step in range(150):
        opt.zero_grad()
        rendered, alpha = model.render(H, W)
        loss = torch.mean((rendered - target) ** 2)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 30 == 0 or step == 149:
            print(f"  step {step:3d}  MSE loss = {loss.item():.5f}")

    assert losses[-1] < losses[0] * 0.5, "Optimization did not meaningfully reduce loss -- CPU correctness check FAILED"
    print(f"PASS: loss reduced {losses[0]:.5f} -> {losses[-1]:.5f} over {len(losses)} steps.")

    # --- exercise the confidence-gated densify/prune logic on hand-built statistics ---
    print("\nExercising confidence-gated densify/prune with synthetic per-Gaussian stats...")
    stats = model.stats
    # Gaussian 0: well-observed from 4 well-spread views, high quality/pose confidence -> should be densify-eligible
    idx0 = torch.tensor([0, 0, 0, 0])
    stats.update(idx0, torch.tensor([0.05, 0.15, 0.25, 0.35]), torch.tensor([0.8, 0.85, 0.9, 0.82]), 0.9, 0.9)
    # Gaussian 1: single observation only -> must be capped, never "densify-eligible" regardless of opacity
    idx1 = torch.tensor([1])
    stats.update(idx1, torch.tensor([0.1]), torch.tensor([0.95]), 0.95, 0.95)
    with torch.no_grad():
        model.opacity_logits[0] = 3.0   # high opacity
        model.opacity_logits[1] = 3.0   # also high opacity, but single-view
        # Set an unambiguous, reasonable prior for the two Gaussians under test (rather
        # than the deliberately-noisy random init used for the rest of the toy scene),
        # so this check isolates the online-statistics gating logic being tested here.
        model.prior_confidence[0] = 0.75
        model.prior_confidence[1] = 0.75

    grad_accum = torch.zeros(N)
    grad_accum[0] = 0.001
    grad_accum[1] = 0.001
    prune_mask, densify_mask, conf = model.densify_and_prune(grad_accum)
    print(f"  Gaussian 0 (4-view, wide-baseline): confidence={conf[0]:.3f}, "
          f"densify_eligible={bool(densify_mask[0])}, pruned={bool(prune_mask[0])}")
    print(f"  Gaussian 1 (single-view only):      confidence={conf[1]:.3f}, "
          f"densify_eligible={bool(densify_mask[1])}, pruned={bool(prune_mask[1])}")
    assert bool(densify_mask[0]) is True, "Well-observed Gaussian should be densify-eligible -- FAILED"
    assert bool(densify_mask[1]) is False, "Single-view Gaussian must be capped below densify threshold -- FAILED"
    print("PASS: single-observation confidence cap correctly blocks densification despite high gradient+opacity.")
