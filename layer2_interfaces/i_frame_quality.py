"""
i_frame_quality.py -- LAYER 2 INTERFACE CONTRACT
STATUS: implemented (interface only, Python ABC), CPU-tested (against the
concrete CPU implementation in layer1_cpu_sandbox/perception/frame_quality.py)

CPU PROTOTYPE:      layer1_cpu_sandbox/perception/frame_quality.py
                     (Laplacian-variance blur, histogram exposure, DCT-grid
                     compression estimate -- classical, explainable, fast)
GPU REPLACEMENT:     a small learned CNN quality-regressor (e.g. a
                     MobileNet/EfficientNet-Lite backbone fine-tuned to
                     predict a scalar "usefulness for reconstruction" score),
                     OR a distilled version of an existing blind
                     image-quality-assessment model (e.g. MUSIQ, HyperIQA)
PRETRAINED CANDIDATE: HyperIQA / MUSIQ (drop-in scalar quality score)
CUSTOM CANDIDATE:    a tiny CNN trained on this project's own synthetic
                     degradation pairs (rgb_clean vs rgb, with known
                     blur_px/jpeg_quality as regression targets) -- the
                     synthetic dataset generator already produces exactly
                     that supervised pair for free (DroneFrame.rgb vs
                     DroneFrame.rgb_clean/blur_px/jpeg_quality)
RESEARCH NOVELTY:    LOW on its own; the novelty is in how the resulting
                     score feeds confidence.py, not in the quality model itself.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class IFrameQualityModel(ABC):
    @abstractmethod
    def score(self, rgb: np.ndarray) -> float:
        """Return an overall quality score in [0,1] for a single RGB frame."""
        raise NotImplementedError

    @abstractmethod
    def score_batch(self, rgb_batch) -> np.ndarray:
        """Return (N,) scores for a batch. GPU implementations should batch this
        on-device rather than looping `score()` per frame."""
        raise NotImplementedError
