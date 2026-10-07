"""
i_gaussian_init.py / i_gaussian_refine.py -- LAYER 2 INTERFACE CONTRACTS
STATUS: implemented (interface only) -- these have NO CPU prototype (see
docs/MISSING_COMPONENTS.md: Gaussian optimization is GPU-only by nature,
there is no meaningful "classical CPU Gaussian splatting" to validate
logic against beyond the point-fusion analogy already covered by
prototype_point_repr.py + confidence.py).

GAUSSIAN INITIALIZATION
GPU IMPLEMENTATION:   initialize one Gaussian per fused, confidence-passing
                       point from `PrototypePointRepresentation`-equivalent
                       fusion (position = point position, initial scale from
                       local point spacing, initial opacity from confidence,
                       color from observed RGB / spherical harmonics DC term)
                       -- i.e. Layer 1's fusion logic IS the initializer;
                       only the primitive type changes (point -> Gaussian).
CANDIDATE:             standard 3DGS/FastGS SfM-point initialization,
                       adapted to seed from our confidence-fused points
                       instead of COLMAP sparse SfM points.
RESEARCH NOVELTY:      LOW (standard initialization, adapted input source).

GAUSSIAN REFINEMENT / DENSIFICATION -- THIS IS WHERE THE CORE NOVELTY LIVES
GPU IMPLEMENTATION:   layer3_gpu/csrc/confidence_gaussian/ (CUDA kernel) +
                       layer3_gpu/python/confidence_gaussian_model.py
                       (training loop). Confidence, accumulated per-Gaussian
                       from the SAME five evidence signals as confidence.py
                       (multi-view redundancy, angle spread, photometric/
                       depth consistency, frame quality, pose confidence),
                       gates: (a) which Gaussians are candidates for
                       split/clone densification, (b) which are pruned for
                       low opacity AND low confidence (vs. standard 3DGS,
                       which prunes on opacity alone), (c) how much of the
                       fixed Gaussian budget goes to a region.
WHY THIS IS DIFFERENT FROM STANDARD 3DGS/FastGS DENSIFICATION:
  Standard 3DGS densifies based on view-space positional gradient magnitude
  averaged over training -- a purely photometric-optimization signal, blind
  to WHY a region has high gradient (could be genuinely under-reconstructed
  detail, OR could be a region with little multi-view support that the
  optimizer is fighting to fit from too few constraints, i.e. overfitting a
  single view). Confidence-gating adds an independent, geometry/evidence-
  based signal that lets the system tell those two cases apart --
  concretely: don't spend Gaussian budget aggressively densifying a region
  the flight path only weakly observed just because its photometric
  gradient is high; that is more likely overfitting than genuine detail.
RESEARCH NOVELTY:      HIGH (this is the paper's central claimed
                       contribution) -- CONDITIONAL on the ablation in
                       docs/EXPERIMENT_PLAN.md actually showing
                       confidence-gated densification beats
                       gradient-only densification on held-out views under
                       the single-pass constraint. This must be measured on
                       GPU, not asserted.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class IGaussianInitializer(ABC):
    @abstractmethod
    def initialize(self, fused_points):
        """fused_points: PrototypePointRepresentation-equivalent fused evidence.
        Returns initial Gaussian parameters (positions, scales, rotations, opacities,
        SH color coefficients, and the carried-over per-primitive confidence)."""
        raise NotImplementedError


class IGaussianRefiner(ABC):
    @abstractmethod
    def step(self, gaussians, render_grad_stats, confidence_stats, iteration: int):
        """One optimization-loop refinement step: given accumulated view-space
        gradient statistics (standard 3DGS signal) AND accumulated confidence
        statistics (this project's signal), return updated Gaussians after
        confidence-gated split/clone/prune."""
        raise NotImplementedError
