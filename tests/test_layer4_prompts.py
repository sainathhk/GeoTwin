"""Sanity tests for layer4_agents.prompts -- the tool schemas are plain data, so these
just confirm the shape is well-formed and self-consistent with schemas.py's bounds."""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer4_agents.prompts import (
    FRAME_AGENT_TOOL, RECONSTRUCTION_AGENT_TOOL, EVALUATION_AGENT_TOOL, build_user_prompt,
    FRAME_AGENT_SYSTEM_PROMPT, RECONSTRUCTION_AGENT_SYSTEM_PROMPT, EVALUATION_AGENT_SYSTEM_PROMPT,
)
from layer4_agents.schemas import RECON_BOUNDS, RECON_INT_BOUNDS, KEEP_FRACTION_BOUNDS, MIN_OVERALL_BOUNDS


def test_all_tools_are_json_serializable_and_well_shaped():
    for tool in (FRAME_AGENT_TOOL, RECONSTRUCTION_AGENT_TOOL, EVALUATION_AGENT_TOOL):
        json.dumps(tool)
        assert "name" in tool and "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        for req in tool["input_schema"]["required"]:
            assert req in tool["input_schema"]["properties"]


def test_reconstruction_tool_bounds_match_schemas_bounds():
    props = RECONSTRUCTION_AGENT_TOOL["input_schema"]["properties"]
    for key, (lo, hi) in RECON_BOUNDS.items():
        assert props[key]["minimum"] == lo
        assert props[key]["maximum"] == hi
    for key, (lo, hi) in RECON_INT_BOUNDS.items():
        assert props[key]["minimum"] == lo
        assert props[key]["maximum"] == hi


def test_frame_tool_bounds_match_schemas_bounds():
    props = FRAME_AGENT_TOOL["input_schema"]["properties"]
    assert props["keep_fraction"]["minimum"] == KEEP_FRACTION_BOUNDS[0]
    assert props["keep_fraction"]["maximum"] == KEEP_FRACTION_BOUNDS[1]
    assert props["min_overall"]["minimum"] == MIN_OVERALL_BOUNDS[0]
    assert props["min_overall"]["maximum"] == MIN_OVERALL_BOUNDS[1]


def test_reconstruction_tool_excludes_ablation_toggles():
    props = RECONSTRUCTION_AGENT_TOOL["input_schema"]["properties"]
    for forbidden in ("use_confidence_gating", "use_quality_filter", "use_sensor_fusion", "use_dynamic_filter"):
        assert forbidden not in props


def test_reconstruction_system_prompt_warns_about_gate_thresholds():
    assert "prune_below" in RECONSTRUCTION_AGENT_SYSTEM_PROMPT
    assert "densify_above" in RECONSTRUCTION_AGENT_SYSTEM_PROMPT


def test_build_user_prompt_embeds_context_as_json():
    ctx = {"a": 1, "nested": {"b": 2.5}}
    prompt = build_user_prompt(ctx)
    assert '"a": 1' in prompt
    assert "Call the tool" in prompt


def test_system_prompts_are_nonempty_strings():
    for p in (FRAME_AGENT_SYSTEM_PROMPT, RECONSTRUCTION_AGENT_SYSTEM_PROMPT, EVALUATION_AGENT_SYSTEM_PROMPT):
        assert isinstance(p, str) and len(p) > 100
