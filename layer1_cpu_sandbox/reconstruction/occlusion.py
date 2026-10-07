"""
occlusion.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested (EVALUATION-SIDE MODULE)

Answers a question the reconstruction pipeline itself cannot honestly
answer from its own output alone: of the scene surface that was NOT
reconstructed with high confidence, how much of that is because a single
flight path geometrically could never have seen it (a true single-pass
blind spot -- module 9, "single-pass visibility analysis" -- and PS
challenge #1/#7), versus how much was theoretically visible but got
dropped by our own frame-quality selection or dynamic-object filtering
(a pipeline weakness we can actually go fix)?

This module is deliberately EVAL-ONLY: it reads the renderer's
per-triangle visibility trace (ground truth), which the reconstruction
pipeline never sees. Its output is what turns "our coverage is only 61%"
from a bare number into an honest, actionable diagnosis.
"""
from __future__ import annotations

import dataclasses
import numpy as np

from ..synthetic.scene_generator import SyntheticScene

NEVER_OBSERVABLE = "never_observable"          # 0 visibility in ANY captured frame -- true single-pass blind spot
OBSERVABLE_BUT_DROPPED = "observable_but_dropped"  # visible in >=1 raw frame, but 0 in the KEPT (post-quality-filter) set
OBSERVED = "observed"                           # visible in >=1 kept frame


@dataclasses.dataclass
class CoverageReport:
    per_triangle_status: np.ndarray   # (n_triangles,) dtype=object
    per_triangle_max_view_angle_quality: np.ndarray  # (n_triangles,) best (smallest) view angle seen, rad; inf if never observed
    fraction_never_observable: float
    fraction_observable_but_dropped: float
    fraction_observed: float
    by_semantic: dict                  # {semantic_name: {status: fraction}}


def analyze_coverage(scene: SyntheticScene, view_angle_by_frame_all: list,
                      view_angle_by_frame_kept: list) -> CoverageReport:
    n_tri = len(scene.triangles)
    seen_all = np.zeros(n_tri, dtype=bool)
    seen_kept = np.zeros(n_tri, dtype=bool)
    best_angle = np.full(n_tri, np.inf, dtype=np.float32)

    for frame_va in view_angle_by_frame_all:
        for ti, ang in frame_va.items():
            if ti < n_tri:
                seen_all[ti] = True
                if ang < best_angle[ti]:
                    best_angle[ti] = ang
    for frame_va in view_angle_by_frame_kept:
        for ti in frame_va.keys():
            if ti < n_tri:
                seen_kept[ti] = True

    status = np.full(n_tri, NEVER_OBSERVABLE, dtype=object)
    status[seen_all & ~seen_kept] = OBSERVABLE_BUT_DROPPED
    status[seen_kept] = OBSERVED

    frac_never = float((status == NEVER_OBSERVABLE).mean())
    frac_dropped = float((status == OBSERVABLE_BUT_DROPPED).mean())
    frac_observed = float((status == OBSERVED).mean())

    from ..synthetic.scene_generator import SemanticClass
    inv_sem = {v: k for k, v in SemanticClass.items()}
    by_semantic = {}
    sem_ids = np.array([t.semantic for t in scene.triangles])
    for sid in np.unique(sem_ids):
        m = sem_ids == sid
        name = inv_sem.get(int(sid), str(sid))
        total = max(1, m.sum())
        by_semantic[name] = {
            NEVER_OBSERVABLE: float((status[m] == NEVER_OBSERVABLE).sum() / total),
            OBSERVABLE_BUT_DROPPED: float((status[m] == OBSERVABLE_BUT_DROPPED).sum() / total),
            OBSERVED: float((status[m] == OBSERVED).sum() / total),
        }

    return CoverageReport(per_triangle_status=status, per_triangle_max_view_angle_quality=best_angle,
                           fraction_never_observable=frac_never, fraction_observable_but_dropped=frac_dropped,
                           fraction_observed=frac_observed, by_semantic=by_semantic)
