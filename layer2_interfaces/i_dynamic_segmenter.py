"""
i_dynamic_segmenter.py -- LAYER 2 INTERFACE CONTRACT
STATUS: implemented (interface only), CPU-tested (against dynamic_object_filter.py)

CPU PROTOTYPE:       layer1_cpu_sandbox/perception/dynamic_object_filter.py
                      (depth-aware rigid-flow residual thresholding --
                      MEASURED to have low precision at 160x120 synthetic
                      resolution for small objects, see that file's
                      docstring for the actual numbers)
GPU REPLACEMENT:      a learned video/motion segmentation network -- either
                      (a) a lightweight motion-aware segmentation head
                      (e.g. a small temporal U-Net over stacked
                      flow+RGB), or (b) SAM2/SAM-style promptable
                      segmentation where the classical residual-flow mask
                      from Layer 1 is reused as a weak PROMPT (point/box)
                      rather than the final answer -- turning a noisy
                      classical heuristic into a strong learned one without
                      throwing away the geometric reasoning
PRETRAINED CANDIDATE: SAM2 (video, promptable) or YOLOv9/RT-DETR for
                      vehicle/pedestrian detection, fused with the
                      classical residual-flow mask as a consistency check
CUSTOM CANDIDATE:     fine-tune a lightweight segmentation head on this
                      project's synthetic vehicle-injection generator
                      (degradation.make_vehicle_tracks already produces
                      perfect training labels: dynamic_mask_gt per frame)
RESEARCH NOVELTY:     LOW-MODERATE on its own; positioned here mainly to
                      fix a documented, measured Layer-1 weakness rather
                      than as a headline contribution.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class IDynamicSegmenter(ABC):
    @abstractmethod
    def segment(self, frame_t, frame_t_next, depth_t, pose_t, pose_t_next, K):
        """Return a DynamicMaskResult-like object with `.mask` (H,W) bool True=dynamic."""
        raise NotImplementedError
