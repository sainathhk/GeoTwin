"""
i_renderer.py -- LAYER 2 INTERFACE CONTRACT
STATUS: implemented (interface only), CPU-tested (against point_renderer.py)

CPU PROTOTYPE:        layer1_cpu_sandbox/reconstruction/point_renderer.py
                       (non-differentiable nearest-point z-buffer splat,
                       eval-only)
GPU REPLACEMENT:       the standard differentiable tile-based Gaussian
                       rasterizer used by 3DGS/FastGS (external dependency,
                       not reimplemented -- see docs/ARCHITECTURE.md
                       "why we do not reimplement the rasterizer"). Used
                       for BOTH training (gradient flows back into Gaussian
                       parameters) and final evaluation rendering.
RESEARCH NOVELTY:      NONE -- deliberately a reused, well-optimized
                       existing component, not part of this project's
                       contribution surface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class IRenderer(ABC):
    @abstractmethod
    def render(self, primitives, pose, K, width: int, height: int):
        """Return (rgb (H,W,3), coverage_or_alpha (H,W)). GPU implementations
        additionally support backpropagation through this call."""
        raise NotImplementedError
