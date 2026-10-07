"""
i_depth_estimator.py -- LAYER 2 INTERFACE CONTRACT
STATUS: implemented (interface only), CPU-tested (against depth_estimation.py)

CPU PROTOTYPE:       layer1_cpu_sandbox/reconstruction/depth_estimation.py
                      (classical plane-sweep MVS, ZNCC patch cost, cost-curve
                      peakiness as a consistency/confidence signal)
GPU REPLACEMENT:      a learned MVS cost-volume network (MVSNet / CasMVSNet /
                      TransMVSNet-style architecture: 2D feature extraction,
                      differentiable homography warping into a cost volume,
                      3D-CNN regularization, soft-argmin depth regression),
                      OR a monocular relative-depth network (e.g. a
                      Depth-Anything-style ViT) whose per-frame output is
                      scale-aligned using the GPS/IMU-fused poses and then
                      fused across views
PRETRAINED CANDIDATE: Depth Anything V2 (monocular prior) + classical
                      multi-view scale/consistency fusion; or a pretrained
                      MVSNet checkpoint fine-tuned on aerial/oblique imagery
CUSTOM CANDIDATE:     an MVS cost-volume network trained/fine-tuned on this
                      project's synthetic dataset generator's *exact*
                      degradation distribution (motion blur, JPEG,
                      illumination) -- directly addresses PS challenge #2/#3
RESEARCH NOVELTY:     MODERATE-HIGH if the network is trained to predict
                      the SAME cost-curve-peakiness confidence signal the
                      classical version derives geometrically -- i.e. a
                      network with a confidence head supervised by
                      multi-view geometric consistency, not just depth L1
                      loss. This directly extends the core novelty into
                      the depth stage itself, not just the point/Gaussian
                      fusion stage.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class IDepthEstimator(ABC):
    @abstractmethod
    def estimate(self, ref_frame, neighbor_frames: List, ref_pose, neighbor_poses: List, K):
        """Return a DepthEstimate-like object: `.depth` (H,W), `.consistency` (H,W in [0,1]).
        `.consistency` MUST be a genuine per-pixel reliability signal (not a constant),
        since confidence.py depends on it having real discriminative power -- see
        tests/test_layer1_depth.py for the correlation check this is held to."""
        raise NotImplementedError
