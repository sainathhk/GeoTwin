"""
llm_agents.py
LAYER 4 (AGENTS) -- STATUS: implemented, NOT executed in this sandbox

This sandbox's bash tool has network access disabled (see the environment's
own network_configuration), so nothing in this file has been run here --
only statically reviewed, the same honesty this repo already applies to
Layer 3 (STATUS.md: "implemented, GPU required, not executed in this
sandbox"). Environment here is even narrower than Layer 3's Colab/GPU
requirement: this needs outbound network access AND an ANTHROPIC_API_KEY,
neither of which this sandbox has. Run `tests/test_layer4_mock_agents.py`
and `tests/test_layer4_orchestrator.py` for what IS proven to work
end-to-end (the mock agents implement the identical interfaces.py contracts,
so the orchestration logic these classes plug into is fully tested even
though these specific classes are not).

Before actually running this against your API key: read it once yourself.
Two things especially worth checking against current docs.claude.com before
a live run -- pricing/rate limits for whatever loop size you configure, and
that "claude-sonnet-5" is still the model string you want (this repo's
default at time of writing; see docs/AGENTIC_ARCHITECTURE.md).

Install: pip install -r requirements-agents.txt
Configure: export ANTHROPIC_API_KEY=...
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .interfaces import IFrameAgent, IReconstructionAgent, IEvaluationAgent, ITrainingControlAgent
from .schemas import (
    FrameAgentDecision, ReconstructionAgentDecision, EvaluationAgentReport, TrainingControlDecision,
)
from .prompts import (
    FRAME_AGENT_SYSTEM_PROMPT, RECONSTRUCTION_AGENT_SYSTEM_PROMPT, EVALUATION_AGENT_SYSTEM_PROMPT,
    TRAINING_CONTROL_AGENT_SYSTEM_PROMPT,
    FRAME_AGENT_TOOL, RECONSTRUCTION_AGENT_TOOL, EVALUATION_AGENT_TOOL, TRAINING_CONTROL_AGENT_TOOL,
    build_user_prompt,
)

DEFAULT_MODEL = "claude-sonnet-5"  # see this module's docstring before a live run


def _extract_tool_input(message, tool_name: str) -> Optional[dict]:
    """Pulls the first matching tool_use block's `.input` (already a parsed dict --
    the Anthropic SDK decodes tool arguments for you) out of a Messages API response.
    Returns None if the model didn't call the tool at all (e.g. it responded with plain
    text instead, or a content-policy stop) -- callers must treat that as "no valid
    decision", not raise."""
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            return block.input
    return None


def _call_agent_tool(client, model: str, system_prompt: str, tool: dict,
                       context: Dict[str, Any], max_tokens: int = 1024) -> str:
    """Runs one forced-tool-call turn and returns a JSON string (or a JSON-encoded error
    marker on any failure) -- always a string, so every caller can route it through the
    exact same schemas.py `from_raw_json` the mock-agent tests already exercise.
    Never raises: a network error, an auth error, a malformed SDK response, or the model
    simply not calling the tool all collapse to the same "no valid decision" signal,
    because a live pipeline run must never crash on an agent-layer hiccup.
    """
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": build_user_prompt(context)}],
        )
    except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
        return json.dumps({"__error__": f"API call failed: {e}"})

    tool_input = _extract_tool_input(message, tool["name"])
    if tool_input is None:
        return json.dumps({"__error__": "model did not return the expected tool call"})
    try:
        return json.dumps(tool_input)
    except (TypeError, ValueError) as e:
        return json.dumps({"__error__": f"tool input was not JSON-serializable: {e}"})


class LLMFrameAgent(IFrameAgent):
    def __init__(self, client, model: str = DEFAULT_MODEL):
        """`client`: an `anthropic.Anthropic()` instance (construct with your own
        ANTHROPIC_API_KEY; not created here so this module has zero import-time
        dependency on the `anthropic` package -- only users who actually run this
        need it installed)."""
        self.client = client
        self.model = model

    def decide(self, context: Dict[str, Any]) -> FrameAgentDecision:
        raw = _call_agent_tool(self.client, self.model, FRAME_AGENT_SYSTEM_PROMPT,
                                 FRAME_AGENT_TOOL, context)
        return FrameAgentDecision.from_raw_json(raw)


class LLMReconstructionAgent(IReconstructionAgent):
    def __init__(self, client, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def decide(self, context: Dict[str, Any]) -> ReconstructionAgentDecision:
        raw = _call_agent_tool(self.client, self.model, RECONSTRUCTION_AGENT_SYSTEM_PROMPT,
                                 RECONSTRUCTION_AGENT_TOOL, context)
        return ReconstructionAgentDecision.from_raw_json(raw)


class LLMEvaluationAgent(IEvaluationAgent):
    def __init__(self, client, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def decide(self, context: Dict[str, Any]) -> EvaluationAgentReport:
        raw = _call_agent_tool(self.client, self.model, EVALUATION_AGENT_SYSTEM_PROMPT,
                                 EVALUATION_AGENT_TOOL, context)
        # key_numbers is intentionally re-attached from `context` here, not parsed from
        # the model's response -- from_raw_json() already ignores any "key_numbers" the
        # model might echo back, but passing it explicitly makes that guarantee visible
        # at the call site too, not just inside schemas.py.
        return EvaluationAgentReport.from_raw_json(raw, key_numbers=context.get("key_numbers", {}))


class LLMTrainingControlAgent(ITrainingControlAgent):
    """Layer 3 (train_gpu.py) counterpart to LLMReconstructionAgent above -- same pattern,
    a genuinely different decision space (see schemas.TrainingControlDecision)."""

    def __init__(self, client, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def decide(self, context: Dict[str, Any]) -> TrainingControlDecision:
        raw = _call_agent_tool(self.client, self.model, TRAINING_CONTROL_AGENT_SYSTEM_PROMPT,
                                 TRAINING_CONTROL_AGENT_TOOL, context)
        return TrainingControlDecision.from_raw_json(raw)


def build_default_client():
    """Convenience constructor: `import anthropic; return anthropic.Anthropic()`, reading
    ANTHROPIC_API_KEY from the environment the way the SDK does by default. Kept as a
    function (not run at import time) so importing this MODULE never requires the
    `anthropic` package to be installed -- only calling this function does."""
    import anthropic  # local import: see docstring
    return anthropic.Anthropic()
