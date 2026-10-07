"""
i_representation.py -- LAYER 2 INTERFACE CONTRACT
STATUS: implemented (interface only); CPU side is a deliberately non-differentiable
stand-in, GPU side is the real (unexecuted-here) representation.

CPU PROTOTYPE:        layer1_cpu_sandbox/reconstruction/prototype_point_repr.py
                       `PrototypePointRepresentation` -- a plain fused point
                       cloud with per-point observation statistics. NOT
                       differentiable, NOT splatted/rendered with any
                       learned optimization. Exists purely to validate the
                       confidence/fusion LOGIC cheaply.
GPU REPLACEMENT:       layer3_gpu/ -- 3D Gaussian Splatting (position,
                       anisotropic covariance, opacity, spherical-harmonic
                       color per Gaussian), rendered with the standard
                       differentiable tile rasterizer used by 3DGS/FastGS,
                       optimized by gradient descent against the captured
                       frames. See docs/ARCHITECTURE.md section
                       "3-D REPRESENTATION CHOICE" for the point cloud vs
                       mesh vs NeRF vs 3DGS comparison and why 3DGS wins
                       for this problem.
RESEARCH NOVELTY:      The representation itself (3DGS) is NOT novel --
                       FastGS/3DGS are existing methods. What plugs into
                       this interface as OUR contribution is
                       `ConfidenceAwareGaussianModel`
                       (layer3_gpu/python/confidence_gaussian_model.py):
                       confidence is tracked per-Gaussian and gates
                       densification/pruning/opacity regularization.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class ISceneRepresentation(ABC):
    @abstractmethod
    def fuse_observations(self, depth_estimates, dynamic_masks, poses, rgb_frames,
                           frame_quality_scores, pose_confidences, K):
        """Consume per-frame depth/quality/pose evidence, return the representation's
        own primitive set (points, or Gaussians) with per-primitive observation stats."""
        raise NotImplementedError

    @abstractmethod
    def render(self, pose, K, width: int, height: int):
        """Render the representation from an arbitrary camera pose. The CPU prototype
        uses a non-differentiable splat (point_renderer.py, eval-only); the GPU
        representation uses the differentiable tile rasterizer (train + eval)."""
        raise NotImplementedError
