"""
baseline_framework.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Named `PipelineConfig` variants. Every entry here is a real, runnable
configuration of the SAME pipeline code (pipeline.py) -- "baselines" are
not separate re-implementations, they are documented ways of disabling one
or more of our contributions, which is what makes the resulting comparison
an honest ablation rather than a rigged one.

Note on scope (see also docs/EXPERIMENT_PLAN.md): classical photogrammetry
/ COLMAP, a real FastGS run, and a NeRF baseline are NOT implemented here
-- they require GPU/third-party toolchains unavailable in Environment A.
`docs/EXPERIMENT_PLAN.md` specifies exactly how to run them in Environment
B for the full baseline comparison table; the CPU sandbox instead isolates
what its own modules individually contribute.
"""
from __future__ import annotations

import dataclasses

from ..pipeline import PipelineConfig


def get_ablation_configs():
    """
    One config per row of the requested ablation table. Each config differs
    from `full_method` by exactly one toggle (or, for `bare_minimum`, all of
    them), so any metric delta can be attributed to a single module.
    """
    return [
        PipelineConfig(name="bare_minimum",
                       use_quality_filter=False, use_sensor_fusion=False,
                       use_dynamic_filter=False, use_confidence_gating=False),
        PipelineConfig(name="+quality_filter",
                       use_quality_filter=True, use_sensor_fusion=False,
                       use_dynamic_filter=False, use_confidence_gating=False),
        PipelineConfig(name="+sensor_fusion",
                       use_quality_filter=True, use_sensor_fusion=True,
                       use_dynamic_filter=False, use_confidence_gating=False),
        PipelineConfig(name="+dynamic_filter",
                       use_quality_filter=True, use_sensor_fusion=True,
                       use_dynamic_filter=True, use_confidence_gating=False),
        PipelineConfig(name="full_method (+confidence_gating)",
                       use_quality_filter=True, use_sensor_fusion=True,
                       use_dynamic_filter=True, use_confidence_gating=True),
    ]


def get_baseline_configs():
    """
    Single-factor comparisons against the full method, for the "does this
    ONE module matter, holding everything else fixed" table (as opposed to
    the cumulative build-up above).
    """
    full = PipelineConfig(name="full_method")
    return [
        full,
        dataclasses.replace(full, name="no_sensor_fusion (raw GPS)", use_sensor_fusion=False),
        dataclasses.replace(full, name="no_quality_filter (all frames)", use_quality_filter=False),
        dataclasses.replace(full, name="no_dynamic_filter", use_dynamic_filter=False),
        dataclasses.replace(full, name="no_confidence_gating (keep everything)", use_confidence_gating=False),
        dataclasses.replace(full, name="fine_voxel (0.5m, noise-mismatched)", voxel_size=0.5),
    ]
