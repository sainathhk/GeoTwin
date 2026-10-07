"""
interfaces.py
LAYER 4 (AGENTS) -- CONTRACTS

Same role `layer2_interfaces/*.py` plays for classical-vs-learned CV modules,
one level up: a fixed contract with (at least) two implementations behind
it. Here the two sides are `mock_agents.py` (deterministic, CPU-testable,
no network) and `llm_agents.py` (Anthropic-API-backed, implemented but NOT
executed in this sandbox -- no network here, see docs/AGENTIC_ARCHITECTURE.md).

Every `decide()` takes a plain dict (built by context_builders.py -- numbers
and short strings only, never geometry) and returns one of the typed,
already-clamped decisions from schemas.py. `orchestrator.py` only ever calls
`decide()`; it does not know or care whether the implementation behind it is
a rule table or an LLM call, which is what lets `mock_agents.py` genuinely
substitute for `llm_agents.py` in tests rather than merely resembling it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .schemas import (
    FrameAgentDecision, ReconstructionAgentDecision, EvaluationAgentReport, TrainingControlDecision,
)


class IFrameAgent(ABC):
    """Decides frame-selection PARAMETERS (keep_fraction, min_overall) from a quality +
    overlap summary. Never sees a pixel; never returns frame indices directly -- only
    the parameters that `select_frames()` (untouched, deterministic) will use to compute
    them. See schemas.FrameAgentDecision for the exact, bounded contract."""

    @abstractmethod
    def decide(self, context: Dict[str, Any]) -> FrameAgentDecision:
        raise NotImplementedError


class IReconstructionAgent(ABC):
    """Decides whether to accept, retry, or flag-and-accept a reconstruction, and which
    CONTINUOUS threshold(s) to adjust for a retry. Never sees point coordinates, only
    aggregate counts/means/histograms; never may touch the ablation method-toggle fields
    (use_confidence_gating etc.) -- schemas.RECON_BOUNDS structurally excludes them."""

    @abstractmethod
    def decide(self, context: Dict[str, Any]) -> ReconstructionAgentDecision:
        raise NotImplementedError


class IEvaluationAgent(ABC):
    """Turns an already-computed metrics dict into a natural-language diagnosis plus an
    accept/needs_more_data/reject verdict. Never invents numbers -- `key_numbers` on the
    returned report is always exactly what the caller passed in, never something the
    agent supplied itself (see schemas.EvaluationAgentReport.from_raw_json)."""

    @abstractmethod
    def decide(self, context: Dict[str, Any]) -> EvaluationAgentReport:
        raise NotImplementedError


class ITrainingControlAgent(ABC):
    """Watches train_gpu.py's actual GPU Gaussian-training loop (held-out PSNR/SSIM/LPIPS
    history per checkpoint) and decides whether to keep going, freeze further
    densification, or stop training early. Never sees a rendered image or a Gaussian
    parameter directly -- only the same scalar history a human would read off the
    console log. See schemas.TrainingControlDecision for the narrow, enum-only contract."""

    @abstractmethod
    def decide(self, context: Dict[str, Any]) -> TrainingControlDecision:
        raise NotImplementedError
