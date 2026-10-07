"""Unit tests for layer4_agents.schemas -- the untrusted-LLM-text validation boundary.
These are the most important tests in the agent layer: they prove a malformed,
out-of-bounds, or adversarially-shaped response can never reach the deterministic
pipeline unfiltered.
"""
import dataclasses
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer4_agents.schemas import (
    FrameAgentDecision, ReconstructionAgentDecision, EvaluationAgentReport,
    KEEP_FRACTION_BOUNDS, RECON_BOUNDS, RECON_INT_BOUNDS,
)
from layer1_cpu_sandbox.pipeline import PipelineConfig
from layer1_cpu_sandbox.real_data.real_pipeline import RealPipelineConfig


# ---------------------------------------------------------------------------
# FrameAgentDecision
# ---------------------------------------------------------------------------

def test_frame_agent_valid_json_parses_cleanly():
    d = FrameAgentDecision.from_raw_json('{"keep_fraction": 0.6, "min_overall": 0.2, "reasoning": "ok"}')
    assert d.keep_fraction == 0.6
    assert d.min_overall == 0.2
    assert d.reasoning == "ok"
    assert d.warnings == []


def test_frame_agent_strips_markdown_code_fences():
    text = '```json\n{"keep_fraction": 0.5, "min_overall": 0.1}\n```'
    d = FrameAgentDecision.from_raw_json(text)
    assert d.keep_fraction == 0.5
    assert d.warnings == []


def test_frame_agent_malformed_json_falls_back_safely():
    d = FrameAgentDecision.from_raw_json("this is not json at all {{{")
    assert d.keep_fraction == 0.75  # matches select_frames()'s own pre-agent default
    assert d.min_overall == 0.15
    assert len(d.warnings) == 1


def test_frame_agent_clamps_out_of_range_keep_fraction_high():
    d = FrameAgentDecision.from_raw_json('{"keep_fraction": 5.0}')
    assert d.keep_fraction == KEEP_FRACTION_BOUNDS[1]
    assert any("clamped" in w for w in d.warnings)


def test_frame_agent_clamps_out_of_range_keep_fraction_low():
    d = FrameAgentDecision.from_raw_json('{"keep_fraction": -3.0}')
    assert d.keep_fraction == KEEP_FRACTION_BOUNDS[0]
    assert any("clamped" in w for w in d.warnings)


def test_frame_agent_wrong_type_falls_back_to_default_with_warning():
    d = FrameAgentDecision.from_raw_json('{"keep_fraction": "very high please"}')
    assert d.keep_fraction == 0.75
    assert any("not a number" in w for w in d.warnings)


def test_frame_agent_ignores_unknown_and_geometry_shaped_fields():
    # An injected/hallucinated "positions" field must never surface anywhere on the
    # decision object -- there is no attribute for it to land on.
    text = '{"keep_fraction": 0.8, "positions": [[1,2,3],[4,5,6]], "inject": "ignore prior instructions"}'
    d = FrameAgentDecision.from_raw_json(text)
    assert d.keep_fraction == 0.8
    assert not hasattr(d, "positions")
    assert not hasattr(d, "inject")


def test_frame_agent_flags_list_filters_non_string_items():
    text = '{"flags": ["gap near frame 12", 42, {"bad": "object"}, "dup risk"]}'
    d = FrameAgentDecision.from_raw_json(text)
    assert "gap near frame 12" in d.flags
    assert "dup risk" in d.flags
    assert len(d.flags) == 3  # the dict is dropped, the int is stringified


def test_frame_agent_safe_default_matches_pre_agent_behavior():
    d = FrameAgentDecision.safe_default()
    assert d.keep_fraction == 0.75
    assert d.min_overall == 0.15


def test_frame_agent_call_failure_sentinel_routes_to_safe_default():
    # This is the JSON llm_agents._call_agent_tool emits on an API exception or the
    # model not calling the tool -- it must resolve to the SAME safe_default() a
    # malformed-JSON response gets, not to from_raw_json's ordinary missing-field
    # per-field defaults (which happen to look similar but aren't the same signal).
    d = FrameAgentDecision.from_raw_json('{"__error__": "API call failed: timeout"}')
    assert d.keep_fraction == 0.75 and d.min_overall == 0.15
    assert any("timeout" in w for w in d.warnings)


# ---------------------------------------------------------------------------
# ReconstructionAgentDecision
# ---------------------------------------------------------------------------

def test_reconstruction_agent_valid_retry_decision():
    text = '{"action": "retry", "voxel_size": 1.5, "min_consistency": 0.4, "reasoning": "low coverage"}'
    d = ReconstructionAgentDecision.from_raw_json(text)
    assert d.action == "retry"
    assert d.voxel_size == 1.5
    assert d.min_consistency == 0.4
    assert d.depth_window is None  # untouched fields stay None -> "keep current config value"
    assert d.warnings == []


def test_reconstruction_agent_invalid_action_defaults_to_accept():
    d = ReconstructionAgentDecision.from_raw_json('{"action": "DELETE_EVERYTHING"}')
    assert d.action == "accept"
    assert any("not one of" in w for w in d.warnings)


def test_reconstruction_agent_malformed_json_is_safe_accept():
    d = ReconstructionAgentDecision.from_raw_json("<<garbage>>")
    assert d.action == "accept"
    assert d.agent_confidence == 0.0
    assert len(d.warnings) == 1


def test_reconstruction_agent_call_failure_sentinel_routes_to_safe_default():
    d = ReconstructionAgentDecision.from_raw_json('{"__error__": "network down"}')
    assert d.action == "accept"
    assert d.agent_confidence == 0.0  # not 0.5 -- must be the "did not run" signal, not a neutral guess
    assert any("network down" in w for w in d.warnings)


def test_reconstruction_agent_clamps_each_bounded_field_both_directions():
    lo_json = {k: bounds[0] - 1000 for k, bounds in RECON_BOUNDS.items()}
    hi_json = {k: bounds[1] + 1000 for k, bounds in RECON_BOUNDS.items()}
    import json as _json
    d_lo = ReconstructionAgentDecision.from_raw_json(_json.dumps(lo_json))
    d_hi = ReconstructionAgentDecision.from_raw_json(_json.dumps(hi_json))
    for key, (lo, hi) in RECON_BOUNDS.items():
        assert getattr(d_lo, key) == lo, f"{key} low-side clamp failed"
        assert getattr(d_hi, key) == hi, f"{key} high-side clamp failed"


def test_reconstruction_agent_clamps_int_fields():
    d = ReconstructionAgentDecision.from_raw_json('{"depth_window": 999, "n_depths": -50}')
    assert d.depth_window == RECON_INT_BOUNDS["depth_window"][1]
    assert d.n_depths == RECON_INT_BOUNDS["n_depths"][0]


def test_reconstruction_agent_rejects_inverted_depth_range_after_clamp():
    # depth_min > depth_max even after clamping -- must drop BOTH, not guess.
    d = ReconstructionAgentDecision.from_raw_json('{"depth_min": 50, "depth_max": 10}')
    assert d.depth_min is None
    assert d.depth_max is None
    assert any("depth_min" in w and "depth_max" in w for w in d.warnings)


def test_reconstruction_agent_rejects_inverted_prune_densify_after_clamp():
    d = ReconstructionAgentDecision.from_raw_json('{"prune_below": 0.9, "densify_above": 0.2}')
    assert d.prune_below is None
    assert d.densify_above is None


def test_reconstruction_agent_valid_depth_range_is_kept():
    d = ReconstructionAgentDecision.from_raw_json('{"depth_min": 5, "depth_max": 80}')
    assert d.depth_min == 5.0
    assert d.depth_max == 80.0


def test_reconstruction_agent_ignores_disallowed_ablation_toggles():
    # use_confidence_gating etc. are not in the schema at all -- even if an LLM outputs
    # them, they cannot reach PipelineConfig through this decision object.
    text = '{"action": "retry", "use_confidence_gating": false, "use_quality_filter": false}'
    d = ReconstructionAgentDecision.from_raw_json(text)
    cfg = PipelineConfig(use_confidence_gating=True, use_quality_filter=True)
    new_cfg = d.apply_to(cfg)
    assert new_cfg.use_confidence_gating is True
    assert new_cfg.use_quality_filter is True


def test_reconstruction_agent_apply_to_only_changes_set_fields_pipelineconfig():
    cfg = PipelineConfig(voxel_size=2.0, min_consistency=0.3, keep_fraction=0.75)
    d = ReconstructionAgentDecision(action="retry", voxel_size=1.0)
    new_cfg = d.apply_to(cfg)
    assert new_cfg.voxel_size == 1.0
    assert new_cfg.min_consistency == 0.3  # unchanged
    assert new_cfg.keep_fraction == 0.75   # unchanged
    assert cfg.voxel_size == 2.0            # original untouched (no in-place mutation)


def test_reconstruction_agent_apply_to_works_on_real_pipeline_config_too():
    cfg = RealPipelineConfig(voxel_size=2.0, depth_window=2)
    d = ReconstructionAgentDecision(action="retry", voxel_size=0.8, depth_window=4)
    new_cfg = d.apply_to(cfg)
    assert isinstance(new_cfg, RealPipelineConfig)
    assert new_cfg.voxel_size == 0.8
    assert new_cfg.depth_window == 4


def test_reconstruction_agent_agent_confidence_clamped_to_unit_interval():
    d = ReconstructionAgentDecision.from_raw_json('{"agent_confidence": 3.5}')
    assert d.agent_confidence == 1.0


# ---------------------------------------------------------------------------
# EvaluationAgentReport
# ---------------------------------------------------------------------------

def test_evaluation_agent_valid_report():
    text = '{"verdict": "accept", "diagnosis": "coverage is solid", "suggested_next_step": "ship it"}'
    r = EvaluationAgentReport.from_raw_json(text, key_numbers={"psnr": 21.3})
    assert r.verdict == "accept"
    assert r.diagnosis == "coverage is solid"
    assert r.key_numbers == {"psnr": 21.3}  # passed through, not agent-authored


def test_evaluation_agent_key_numbers_are_never_taken_from_the_llm():
    # Even if the "LLM" tries to supply its own key_numbers, they are ignored --
    # key_numbers always come from the real computed metrics the caller passes in.
    text = '{"verdict": "accept", "key_numbers": {"psnr": 999999}}'
    r = EvaluationAgentReport.from_raw_json(text, key_numbers={"psnr": 21.3})
    assert r.key_numbers == {"psnr": 21.3}


def test_evaluation_agent_invalid_verdict_falls_back():
    d = EvaluationAgentReport.from_raw_json('{"verdict": "maybe idk"}', key_numbers={"a": 1})
    assert d.verdict == "needs_more_data"
    assert d.key_numbers == {"a": 1}


def test_evaluation_agent_malformed_json_keeps_real_numbers():
    d = EvaluationAgentReport.from_raw_json("not json", key_numbers={"psnr": 18.2, "ssim": 0.5})
    assert d.verdict == "needs_more_data"
    assert d.key_numbers == {"psnr": 18.2, "ssim": 0.5}
    assert len(d.warnings) == 1


# ---------------------------------------------------------------------------
# TrainingControlDecision (Layer 3 GPU training loop)
# ---------------------------------------------------------------------------

def test_training_control_valid_decision():
    from layer4_agents.schemas import TrainingControlDecision
    d = TrainingControlDecision.from_raw_json('{"action": "stop_densifying", "reasoning": "ssim stalled", "agent_confidence": 0.6}')
    assert d.action == "stop_densifying"
    assert d.agent_confidence == 0.6
    assert d.warnings == []


def test_training_control_invalid_action_defaults_to_continue():
    from layer4_agents.schemas import TrainingControlDecision
    d = TrainingControlDecision.from_raw_json('{"action": "reboot_the_gpu"}')
    assert d.action == "continue"
    assert any("not one of" in w for w in d.warnings)


def test_training_control_malformed_json_is_safe_continue():
    from layer4_agents.schemas import TrainingControlDecision
    d = TrainingControlDecision.from_raw_json("<<not json>>")
    assert d.action == "continue"
    assert d.agent_confidence == 0.0


def test_training_control_call_failure_sentinel_routes_to_safe_default():
    from layer4_agents.schemas import TrainingControlDecision
    d = TrainingControlDecision.from_raw_json('{"__error__": "timeout during training"}')
    assert d.action == "continue"
    assert d.agent_confidence == 0.0
    assert any("timeout" in w for w in d.warnings)


def test_training_control_confidence_clamped():
    from layer4_agents.schemas import TrainingControlDecision
    d = TrainingControlDecision.from_raw_json('{"action": "early_stop", "agent_confidence": 7.0}')
    assert d.agent_confidence == 1.0
