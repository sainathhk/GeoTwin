"""
i_occlusion_completion.py / i_confidence_estimator.py -- LAYER 2 INTERFACE CONTRACTS

OCCLUSION / SURFACE COMPLETION
STATUS: NOT IMPLEMENTED anywhere yet (CPU or GPU) -- deliberately deferred,
see docs/STOP_CONDITIONS.md item 1. Layer 1's occlusion.py is a coverage
DIAGNOSTIC (eval-only, tells you what's missing and why), not a completion
method (it never invents geometry for a gap).
GPU CANDIDATE:         a learned surface-completion / inpainting prior
                       (e.g. a 3D diffusion prior, or a simpler symmetry-
                       and-regularity prior specific to buildings: assume
                       vertical facades are extruded rooflines, complete a
                       partially-seen rectilinear building via a fitted
                       primitive rather than a generic generative model).
CONDITION TO BUILD:    only if docs/EXPERIMENT_PLAN.md's completeness
                       ablation shows confidence-aware pruning alone
                       (i.e. honestly reporting gaps rather than filling
                       them) scores worse on a completeness-weighted metric
                       than stakeholders actually need for the target
                       applications (disaster assessment, inspection) --
                       otherwise, per the brief's own instruction to treat
                       "insufficient observation" as a legitimate, honest
                       output, this may not be worth building at all.
RESEARCH NOVELTY:      HIGH if built with an explicit, visualized
                       confidence distinction between "observed" and
                       "completed" geometry (never silently merging the
                       two) -- this is listed as a Future Research
                       Direction, not a Day-1 must-have.

CONFIDENCE ESTIMATION (GPU VERSION OF THE CORE NOVELTY)
STATUS: CPU prototype implemented and validated (confidence.py, Pearson
r=-0.16 to -0.23 significant correlation with geometric error on the
synthetic sandbox -- see outputs/run1/confidence_vs_error.png). GPU version
NOT YET implemented (CUDA kernel design only, see
layer3_gpu/csrc/confidence_gaussian/).
GPU IMPLEMENTATION:    per-Gaussian accumulation of the same five signals,
                       fused via the SAME weighted-geometric-mean formula
                       as confidence.py by default, with the weights
                       promoted to LEARNABLE parameters (see
                       i_gaussian_refine.py) as a concrete research
                       extension once real GT/RTK checkpoints are available
                       to supervise them.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class ISurfaceCompletion(ABC):
    @abstractmethod
    def complete(self, gaussians_or_points, coverage_report):
        """NOT IMPLEMENTED. See module docstring: build only if the experiment
        plan's completeness ablation justifies it, and never merge completed
        geometry with observed geometry without a persistent confidence-band
        distinction in the output."""
        raise NotImplementedError


class IConfidenceEstimator(ABC):
    @abstractmethod
    def compute(self, primitives_with_observations, weights: dict = None):
        """primitives_with_observations: per-primitive (point or Gaussian) observation
        ledger matching PrototypePointRepresentation's fields. Returns per-primitive
        confidence in [0,1] plus a High/Moderate/Low band label. GPU implementations
        MUST preserve the single-observation cap behaviour from confidence.py unless
        a specific experiment justifies removing it."""
        raise NotImplementedError
