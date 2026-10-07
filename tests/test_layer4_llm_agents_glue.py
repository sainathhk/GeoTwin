"""Tests for the parts of layer4_agents.llm_agents that don't require network access:
the tool-input extraction and error-handling glue. A stub client/message stands in for
the real `anthropic` SDK objects (same shape: `.messages.create(...)` returns something
with a `.content` list of blocks that have `.type`/`.name`/`.input`).

These do NOT prove the real Anthropic API integration works end to end -- only that
`llm_agents.py` handles every response shape it might realistically get back (a good
tool call, no tool call, a thrown exception) without crashing or letting bad data past
the schemas.py validation boundary. See llm_agents.py's own docstring.
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer4_agents.llm_agents import (
    _extract_tool_input, _call_agent_tool, LLMFrameAgent, LLMReconstructionAgent, LLMEvaluationAgent,
)
from layer4_agents.prompts import FRAME_AGENT_SYSTEM_PROMPT, FRAME_AGENT_TOOL


class _StubBlock:
    def __init__(self, type_, name=None, input=None, text=None):
        self.type = type_
        self.name = name
        self.input = input
        self.text = text


class _StubMessage:
    def __init__(self, content):
        self.content = content


class _StubMessagesEndpoint:
    def __init__(self, response=None, exception=None):
        self._response = response
        self._exception = exception
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        if self._exception is not None:
            raise self._exception
        return self._response


class _StubClient:
    def __init__(self, response=None, exception=None):
        self.messages = _StubMessagesEndpoint(response=response, exception=exception)


# ---------------------------------------------------------------------------
# _extract_tool_input
# ---------------------------------------------------------------------------

def test_extract_tool_input_finds_matching_tool_use_block():
    msg = _StubMessage([_StubBlock("text", text="thinking out loud"),
                          _StubBlock("tool_use", name="frame_selection_decision",
                                      input={"keep_fraction": 0.6})])
    result = _extract_tool_input(msg, "frame_selection_decision")
    assert result == {"keep_fraction": 0.6}


def test_extract_tool_input_returns_none_when_tool_not_called():
    msg = _StubMessage([_StubBlock("text", text="I'd rather not call a tool")])
    assert _extract_tool_input(msg, "frame_selection_decision") is None


def test_extract_tool_input_ignores_differently_named_tool():
    msg = _StubMessage([_StubBlock("tool_use", name="some_other_tool", input={"x": 1})])
    assert _extract_tool_input(msg, "frame_selection_decision") is None


def test_extract_tool_input_handles_empty_content():
    msg = _StubMessage([])
    assert _extract_tool_input(msg, "frame_selection_decision") is None


# ---------------------------------------------------------------------------
# _call_agent_tool -- error handling
# ---------------------------------------------------------------------------

def test_call_agent_tool_happy_path_returns_json_string():
    response = _StubMessage([_StubBlock("tool_use", name="frame_selection_decision",
                                          input={"keep_fraction": 0.6, "min_overall": 0.1, "reasoning": "ok"})])
    client = _StubClient(response=response)
    raw = _call_agent_tool(client, "claude-sonnet-5", FRAME_AGENT_SYSTEM_PROMPT, FRAME_AGENT_TOOL, {"a": 1})
    parsed = json.loads(raw)
    assert parsed["keep_fraction"] == 0.6


def test_call_agent_tool_never_raises_on_api_exception():
    client = _StubClient(exception=RuntimeError("connection reset"))
    raw = _call_agent_tool(client, "claude-sonnet-5", FRAME_AGENT_SYSTEM_PROMPT, FRAME_AGENT_TOOL, {"a": 1})
    parsed = json.loads(raw)
    assert "__error__" in parsed


def test_call_agent_tool_handles_model_not_calling_tool():
    response = _StubMessage([_StubBlock("text", text="I decline to call the tool")])
    client = _StubClient(response=response)
    raw = _call_agent_tool(client, "claude-sonnet-5", FRAME_AGENT_SYSTEM_PROMPT, FRAME_AGENT_TOOL, {"a": 1})
    parsed = json.loads(raw)
    assert "__error__" in parsed


def test_call_agent_tool_uses_forced_tool_choice():
    response = _StubMessage([_StubBlock("tool_use", name="frame_selection_decision", input={})])
    client = _StubClient(response=response)
    _call_agent_tool(client, "claude-sonnet-5", FRAME_AGENT_SYSTEM_PROMPT, FRAME_AGENT_TOOL, {"a": 1})
    kwargs = client.messages.last_call_kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "frame_selection_decision"}
    assert kwargs["tools"] == [FRAME_AGENT_TOOL]
    assert kwargs["system"] == FRAME_AGENT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# End-to-end through the full LLM*Agent classes with a stub client -- proves the
# tool-call output correctly reaches schemas.py's validation, exactly like a mock
# agent's hand-written JSON string does in test_layer4_schemas.py.
# ---------------------------------------------------------------------------

def test_llm_frame_agent_end_to_end_with_stub_client_clamps_out_of_range_value():
    response = _StubMessage([_StubBlock("tool_use", name="frame_selection_decision",
                                          input={"keep_fraction": 50.0, "min_overall": 0.1, "reasoning": "x"})])
    agent = LLMFrameAgent(_StubClient(response=response))
    decision = agent.decide({"quality": {}, "raw_capture_overlap": {}})
    assert decision.keep_fraction == 1.0  # clamped, same as the malicious/malformed-input schema tests
    assert any("clamped" in w for w in decision.warnings)


def test_llm_reconstruction_agent_end_to_end_falls_back_safely_on_exception():
    agent = LLMReconstructionAgent(_StubClient(exception=RuntimeError("network down")))
    decision = agent.decide({"current_config": {}, "points": {}, "frames": {},
                               "spatial_diagnostic": {}, "attempt_history": []})
    assert decision.action == "accept"  # the documented fail-safe default
    assert decision.agent_confidence == 0.0


def test_llm_evaluation_agent_end_to_end_preserves_real_key_numbers_not_model_output():
    response = _StubMessage([_StubBlock("tool_use", name="evaluation_report",
                                          input={"verdict": "accept", "diagnosis": "fine",
                                                  "key_numbers": {"psnr": 99999}})])
    agent = LLMEvaluationAgent(_StubClient(response=response))
    report = agent.decide({"key_numbers": {"psnr": 21.3}, "has_ground_truth": True})
    assert report.key_numbers == {"psnr": 21.3}  # real number, not the model's fabricated 99999


# ---------------------------------------------------------------------------
# LLMTrainingControlAgent
# ---------------------------------------------------------------------------

def test_llm_training_control_agent_end_to_end_with_stub_client():
    from layer4_agents.llm_agents import LLMTrainingControlAgent
    response = _StubMessage([_StubBlock("tool_use", name="training_control_decision",
                                          input={"action": "stop_densifying", "reasoning": "ssim stalled",
                                                  "agent_confidence": 0.6})])
    agent = LLMTrainingControlAgent(_StubClient(response=response))
    decision = agent.decide({"held_out_history": [], "best_held_out": {}})
    assert decision.action == "stop_densifying"
    assert decision.agent_confidence == 0.6


def test_llm_training_control_agent_falls_back_safely_on_exception():
    from layer4_agents.llm_agents import LLMTrainingControlAgent
    agent = LLMTrainingControlAgent(_StubClient(exception=RuntimeError("gpu session disconnected")))
    decision = agent.decide({"held_out_history": [], "best_held_out": {}})
    assert decision.action == "continue"  # fail-safe: never abort a healthy run on an agent hiccup
    assert decision.agent_confidence == 0.0


def test_llm_training_control_agent_invalid_action_falls_back_to_continue():
    from layer4_agents.llm_agents import LLMTrainingControlAgent
    response = _StubMessage([_StubBlock("tool_use", name="training_control_decision",
                                          input={"action": "restart_from_scratch", "reasoning": "x"})])
    agent = LLMTrainingControlAgent(_StubClient(response=response))
    decision = agent.decide({"held_out_history": [], "best_held_out": {}})
    assert decision.action == "continue"
