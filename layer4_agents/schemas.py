"""
schemas.py
LAYER 4 (AGENTS) -- STATUS: implemented, CPU-tested

The safety boundary of the whole agent layer lives here, not in llm_agents.py.
An LLM's raw output is text: it might be well-formed JSON, malformed JSON,
JSON missing fields, JSON with fields of the wrong type, or JSON with EXTRA
fields it was never asked for (a prompt-injected drone log field, a stray
"positions": [[...]] block, anything). Every one of those has to turn into a
safe, bounded, typed decision before it's allowed anywhere near
`PipelineConfig`/`RealPipelineConfig` -- this module is that turnstile, not
an afterthought around it.

Three rules this module enforces mechanically (not just by convention):
  1. Only fields explicitly named in a *_FIELDS tuple are ever read out of the
     parsed JSON. Anything else -- however it got there -- is silently
     dropped. This is what makes "the agent cannot touch geometry" a
     property of the CODE, not a hope about the prompt: there is no schema
     field an agent could populate with point coordinates or pixel data even
     if it tried.
  2. Every numeric field has a documented (lo, hi) bound from *_BOUNDS. A
     value outside that range is CLAMPED, not rejected outright -- an LLM
     that says voxel_size=999 almost always means "much larger than
     default", and clamping preserves that direction of intent while
     guaranteeing the pipeline never receives a value it can't safely run
     with. Every clamp is recorded in `warnings` so the trail is honest
     about what actually happened to the agent's request.
  3. Anything that cannot be salvaged (unparseable JSON, an action string
     outside the fixed enum with no reasonable default) resolves to the
     SAFEST fallback for that decision type, not to "guess and continue".
     For the Reconstruction Agent that fallback is `action="accept"` --
     stop retrying, keep the last known-good result -- exactly the
     fail-safe posture described in docs/AGENTIC_ARCHITECTURE.md.
"""
from __future__ import annotations

import dataclasses
import json
import re
from typing import Optional, List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text.strip()).strip()


def _try_parse_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """Returns (parsed_dict, None) on success, or (None, error_message) on failure.
    Never raises -- this sits directly on the untrusted-text boundary."""
    cleaned = _strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"could not parse JSON: {e}"
    if not isinstance(obj, dict):
        return None, f"expected a JSON object, got {type(obj).__name__}"
    return obj, None


def _call_failure_message(obj: dict) -> Optional[str]:
    """llm_agents.py encodes an upstream failure (API exception, model didn't call the
    tool, non-serializable tool input) as valid JSON of the shape {"__error__": "..."} --
    this is deliberately still valid JSON so it flows through the same _try_parse_json
    path as everything else, rather than needing a second failure channel. Checking for
    it here means EVERY *_AgentDecision.from_raw_json() routes an upstream call failure to
    the same safe_default() a malformed-JSON response gets, instead of quietly falling
    through to from_raw_json's ordinary per-field defaults (which mostly look safe but
    are not the same thing -- e.g. agent_confidence would default to 0.5, "somewhat
    confident", instead of 0.0, "did not actually run")."""
    if isinstance(obj.get("__error__"), str):
        return obj["__error__"]
    return None


def _coerce_float(raw: dict, key: str, default: float) -> Tuple[float, Optional[str]]:
    if key not in raw:
        return default, None
    try:
        return float(raw[key]), None
    except (TypeError, ValueError):
        return default, f"field '{key}' was not a number ({raw[key]!r}); used default {default}"


def _coerce_int(raw: dict, key: str, default: int) -> Tuple[int, Optional[str]]:
    if key not in raw:
        return default, None
    try:
        return int(round(float(raw[key]))), None
    except (TypeError, ValueError):
        return default, f"field '{key}' was not a number ({raw[key]!r}); used default {default}"


def _coerce_str(raw: dict, key: str, default: str, max_len: int = 800) -> str:
    v = raw.get(key, default)
    if not isinstance(v, str):
        return default
    return v[:max_len]


def _coerce_str_list(raw: dict, key: str, max_items: int = 8, max_len: int = 200) -> List[str]:
    v = raw.get(key, [])
    if not isinstance(v, list):
        return []
    return [str(x)[:max_len] for x in v[:max_items] if isinstance(x, (str, int, float))]


def _clamp(value: float, lo: float, hi: float, field_name: str,
           warnings: List[str]) -> float:
    if value < lo:
        warnings.append(f"'{field_name}'={value} below minimum {lo}; clamped to {lo}")
        return lo
    if value > hi:
        warnings.append(f"'{field_name}'={value} above maximum {hi}; clamped to {hi}")
        return hi
    return value


def _clamp_int(value: int, lo: int, hi: int, field_name: str, warnings: List[str]) -> int:
    return int(_clamp(value, lo, hi, field_name, warnings))


# ---------------------------------------------------------------------------
# Frame Agent decision
# ---------------------------------------------------------------------------
# Bounds intentionally mirror select_frames()'s own floor (`max(3, ...)` kept
# frames) and frame_quality.py's documented ~0-1 score range -- see
# docs/AGENTIC_ARCHITECTURE.md's bounds table for the full rationale and the
# explicit note that these were tuned on Environment A (downsampled) frames
# and should be re-checked on real 1080p/4K footage before being trusted there.

KEEP_FRACTION_BOUNDS = (0.3, 1.0)
MIN_OVERALL_BOUNDS = (0.0, 0.5)


@dataclasses.dataclass
class FrameAgentDecision:
    keep_fraction: float = 0.75
    min_overall: float = 0.15
    reasoning: str = ""
    flags: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)  # populated by validation, not the agent

    @staticmethod
    def safe_default() -> "FrameAgentDecision":
        """The values select_frames() already used before this agent layer existed --
        i.e. "do nothing different" is always a valid fallback."""
        return FrameAgentDecision(keep_fraction=0.75, min_overall=0.15,
                                    reasoning="fallback: could not obtain a valid agent decision")

    @staticmethod
    def from_raw_json(text: str) -> "FrameAgentDecision":
        obj, err = _try_parse_json(text)
        if obj is None:
            d = FrameAgentDecision.safe_default()
            d.warnings.append(err)
            return d
        call_err = _call_failure_message(obj)
        if call_err is not None:
            d = FrameAgentDecision.safe_default()
            d.warnings.append(call_err)
            return d
        warnings: List[str] = []
        keep_fraction, w1 = _coerce_float(obj, "keep_fraction", 0.75)
        min_overall, w2 = _coerce_float(obj, "min_overall", 0.15)
        for w in (w1, w2):
            if w:
                warnings.append(w)
        keep_fraction = _clamp(keep_fraction, *KEEP_FRACTION_BOUNDS, "keep_fraction", warnings)
        min_overall = _clamp(min_overall, *MIN_OVERALL_BOUNDS, "min_overall", warnings)
        return FrameAgentDecision(
            keep_fraction=keep_fraction, min_overall=min_overall,
            reasoning=_coerce_str(obj, "reasoning", ""),
            flags=_coerce_str_list(obj, "flags"),
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Reconstruction Agent decision
# ---------------------------------------------------------------------------
# Deliberately excludes use_quality_filter / use_sensor_fusion / use_dynamic_filter /
# use_confidence_gating: those are the ablation_framework.py method switches this
# repo's novelty numbers are built on (confidence-gated vs not, etc.). An automatic
# retry loop silently flipping one of those to chase a better-looking number would
# contaminate the exact comparison the paper depends on. This agent may only tune
# CONTINUOUS thresholds -- it cannot change which method variant it's running.

ACTIONS = ("accept", "retry", "flag_and_accept")

RECON_BOUNDS: Dict[str, Tuple[float, float]] = {
    "voxel_size": (0.3, 10.0),
    "min_consistency": (0.0, 0.9),
    "depth_min": (0.3, 500.0),
    "depth_max": (1.0, 1000.0),
    "prune_below": (0.0, 0.9),
    "densify_above": (0.1, 1.0),
}
RECON_INT_BOUNDS: Dict[str, Tuple[int, int]] = {
    "depth_window": (1, 5),
    "n_depths": (8, 96),
}


@dataclasses.dataclass
class ReconstructionAgentDecision:
    action: str = "accept"
    voxel_size: Optional[float] = None
    min_consistency: Optional[float] = None
    depth_window: Optional[int] = None
    n_depths: Optional[int] = None
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    prune_below: Optional[float] = None
    densify_above: Optional[float] = None
    reasoning: str = ""
    agent_confidence: float = 0.5  # the AGENT's own self-rated certainty in this decision --
                                     # unrelated to reconstruction/confidence.py's per-point
                                     # geometric confidence; never conflate the two.
    warnings: List[str] = dataclasses.field(default_factory=list)

    @staticmethod
    def safe_default() -> "ReconstructionAgentDecision":
        """Fail-safe posture: stop retrying, keep whatever the last valid run produced,
        rather than loop on a config we can no longer trust."""
        return ReconstructionAgentDecision(
            action="accept", reasoning="fallback: could not obtain a valid agent decision",
            agent_confidence=0.0,
        )

    @staticmethod
    def from_raw_json(text: str) -> "ReconstructionAgentDecision":
        obj, err = _try_parse_json(text)
        if obj is None:
            d = ReconstructionAgentDecision.safe_default()
            d.warnings.append(err)
            return d
        call_err = _call_failure_message(obj)
        if call_err is not None:
            d = ReconstructionAgentDecision.safe_default()
            d.warnings.append(call_err)
            return d

        warnings: List[str] = []
        action = obj.get("action")
        if action not in ACTIONS:
            warnings.append(f"'action'={action!r} not one of {ACTIONS}; defaulted to 'accept'")
            action = "accept"

        floats: Dict[str, Optional[float]] = {}
        for key, (lo, hi) in RECON_BOUNDS.items():
            if key in obj and obj[key] is not None:
                val, w = _coerce_float(obj, key, None)
                if w:
                    warnings.append(w)
                    floats[key] = None
                else:
                    floats[key] = _clamp(val, lo, hi, key, warnings)
            else:
                floats[key] = None

        ints: Dict[str, Optional[int]] = {}
        for key, (lo, hi) in RECON_INT_BOUNDS.items():
            if key in obj and obj[key] is not None:
                val, w = _coerce_int(obj, key, None)
                if w:
                    warnings.append(w)
                    ints[key] = None
                else:
                    ints[key] = _clamp_int(val, lo, hi, key, warnings)
            else:
                ints[key] = None

        # Cross-field sanity: never let a partially-clamped pair invert. If both are
        # present and now backwards, drop BOTH rather than guess which one was "right" --
        # dropping means the pipeline just keeps its current values for these two fields.
        if floats["depth_min"] is not None and floats["depth_max"] is not None \
                and floats["depth_min"] >= floats["depth_max"]:
            warnings.append(f"depth_min ({floats['depth_min']}) >= depth_max ({floats['depth_max']}) "
                              f"after clamping; ignoring both, keeping current config values")
            floats["depth_min"] = None
            floats["depth_max"] = None
        if floats["prune_below"] is not None and floats["densify_above"] is not None \
                and floats["prune_below"] >= floats["densify_above"]:
            warnings.append(f"prune_below ({floats['prune_below']}) >= densify_above "
                              f"({floats['densify_above']}) after clamping; ignoring both, "
                              f"keeping current config values")
            floats["prune_below"] = None
            floats["densify_above"] = None

        agent_confidence, w = _coerce_float(obj, "agent_confidence", 0.5)
        if w:
            warnings.append(w)
        agent_confidence = _clamp(agent_confidence, 0.0, 1.0, "agent_confidence", warnings)

        return ReconstructionAgentDecision(
            action=action,
            voxel_size=floats["voxel_size"], min_consistency=floats["min_consistency"],
            depth_window=ints["depth_window"], n_depths=ints["n_depths"],
            depth_min=floats["depth_min"], depth_max=floats["depth_max"],
            prune_below=floats["prune_below"], densify_above=floats["densify_above"],
            reasoning=_coerce_str(obj, "reasoning", ""),
            agent_confidence=agent_confidence,
            warnings=warnings,
        )

    def apply_to(self, config):
        """Returns a NEW config (dataclasses.replace) with only the fields this decision
        actually set. Never mutates `config` in place -- the caller keeps the
        pre-decision config in its history untouched."""
        updates = {k: v for k, v in {
            "voxel_size": self.voxel_size, "min_consistency": self.min_consistency,
            "depth_window": self.depth_window, "n_depths": self.n_depths,
            "depth_min": self.depth_min, "depth_max": self.depth_max,
            "prune_below": self.prune_below, "densify_above": self.densify_above,
        }.items() if v is not None}
        return dataclasses.replace(config, **updates)


# ---------------------------------------------------------------------------
# Evaluation Agent report
# ---------------------------------------------------------------------------

VERDICTS = ("accept", "needs_more_data", "reject")


@dataclasses.dataclass
class EvaluationAgentReport:
    verdict: str = "needs_more_data"
    diagnosis: str = ""
    key_numbers: Dict[str, Any] = dataclasses.field(default_factory=dict)
    suggested_next_step: str = ""
    warnings: List[str] = dataclasses.field(default_factory=list)

    @staticmethod
    def safe_default(key_numbers: Optional[Dict[str, Any]] = None) -> "EvaluationAgentReport":
        return EvaluationAgentReport(
            verdict="needs_more_data",
            diagnosis="fallback: could not obtain a valid agent report; numbers are still real, "
                       "this text summary just wasn't generated by the model.",
            key_numbers=key_numbers or {},
        )

    @staticmethod
    def from_raw_json(text: str, key_numbers: Optional[Dict[str, Any]] = None) -> "EvaluationAgentReport":
        obj, err = _try_parse_json(text)
        if obj is None:
            d = EvaluationAgentReport.safe_default(key_numbers)
            d.warnings.append(err)
            return d
        call_err = _call_failure_message(obj)
        if call_err is not None:
            d = EvaluationAgentReport.safe_default(key_numbers)
            d.warnings.append(call_err)
            return d
        warnings: List[str] = []
        verdict = obj.get("verdict")
        if verdict not in VERDICTS:
            warnings.append(f"'verdict'={verdict!r} not one of {VERDICTS}; defaulted to 'needs_more_data'")
            verdict = "needs_more_data"
        return EvaluationAgentReport(
            verdict=verdict,
            diagnosis=_coerce_str(obj, "diagnosis", "", max_len=2000),
            key_numbers=key_numbers or {},  # ALWAYS the numbers we computed, never agent-supplied numbers
            suggested_next_step=_coerce_str(obj, "suggested_next_step", "", max_len=500),
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Training Control Agent decision (Layer 3 GPU training loop, NOT Layer 1)
# ---------------------------------------------------------------------------
# A genuinely different parameter space from ReconstructionAgentDecision above: there is
# no voxel_size/depth_window here, because this agent watches train_gpu.py's actual
# Gaussian-splatting optimization loop, not the CPU point-fusion pipeline. Deliberately
# narrow and ENUM-ONLY (no numeric config the agent can set directly) -- see
# docs/AGENTIC_ARCHITECTURE.md's "Training Control Agent" section for why: the two real
# actions available (freeze densification early, stop training early) are both simple,
# one-directional, and can only make training MORE conservative than the human-configured
# cfg.densify_until_iter/max_iterations, never less -- there is no equivalent need for a
# bounded numeric range to clamp.

TRAINING_ACTIONS = ("continue", "stop_densifying", "early_stop")


@dataclasses.dataclass
class TrainingControlDecision:
    action: str = "continue"
    reasoning: str = ""
    agent_confidence: float = 0.5
    warnings: List[str] = dataclasses.field(default_factory=list)

    @staticmethod
    def safe_default() -> "TrainingControlDecision":
        """Fail-safe posture, same shape as ReconstructionAgentDecision.safe_default():
        "continue" is the do-nothing-different action here (train() runs exactly as it
        would with no agent at all), not "stop" -- an agent-layer hiccup should not
        abort a multi-minute-to-multi-hour real GPU run that was otherwise healthy."""
        return TrainingControlDecision(action="continue",
                                         reasoning="fallback: could not obtain a valid agent decision",
                                         agent_confidence=0.0)

    @staticmethod
    def from_raw_json(text: str) -> "TrainingControlDecision":
        obj, err = _try_parse_json(text)
        if obj is None:
            d = TrainingControlDecision.safe_default()
            d.warnings.append(err)
            return d
        call_err = _call_failure_message(obj)
        if call_err is not None:
            d = TrainingControlDecision.safe_default()
            d.warnings.append(call_err)
            return d

        warnings: List[str] = []
        action = obj.get("action")
        if action not in TRAINING_ACTIONS:
            warnings.append(f"'action'={action!r} not one of {TRAINING_ACTIONS}; defaulted to 'continue'")
            action = "continue"

        agent_confidence, w = _coerce_float(obj, "agent_confidence", 0.5)
        if w:
            warnings.append(w)
        agent_confidence = _clamp(agent_confidence, 0.0, 1.0, "agent_confidence", warnings)

        return TrainingControlDecision(
            action=action, reasoning=_coerce_str(obj, "reasoning", ""),
            agent_confidence=agent_confidence, warnings=warnings,
        )
