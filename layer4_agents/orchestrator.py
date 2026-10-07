"""
orchestrator.py
LAYER 4 (AGENTS) -- STATUS: implemented, CPU-tested end-to-end (with mock agents)

This is the "Agent -> Decision -> Tool/API -> Measured result -> Agent" loop
from docs/AGENTIC_ARCHITECTURE.md, made concrete. It never calls into
depth_estimation.py, confidence.py, or any geometry code directly -- it only
calls the SAME `run_pipeline`/`run_real_pipeline` entry points a human would
call by hand, with a config an agent chose. Every iteration's full decision
(including every clamp/warning schemas.py produced) is recorded, so a run's
JSON report is a complete, inspectable trail of what was tried and why --
not just the final result.

Fail-safe behavior, all of it deliberate, none of it "and then it crashes":
  - `max_iterations` bounds the retry loop unconditionally. If an agent keeps
    asking to retry past the budget, the orchestrator stops anyway and
    reports the last result, with a note explaining why.
  - If applying a retry config and re-running the pipeline itself raises
    (an edge case inside the *_BOUNDS-clamped-but-not-exhaustively-verified
    parameter space slips past a numerical assumption somewhere in
    depth_estimation.py/prototype_point_repr.py), the orchestrator catches
    it, reverts to the last known-good config/result, and stops -- it does
    not propagate the exception up and lose a perfectly good prior result.
  - Any change to `prune_below`/`densify_above` (the confidence-gating
    thresholds this project's novelty claim depends on) is recorded under
    an explicit `gate_threshold_changed` flag with before/after values, in
    every iteration record, regardless of which agent (mock or LLM)
    produced it -- see mock_agents.py's module docstring for why this
    matters.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional

import cv2

from .interfaces import IFrameAgent, IReconstructionAgent, IEvaluationAgent
from .schemas import ReconstructionAgentDecision
from .context_builders import (
    build_frame_agent_context, build_reconstruction_agent_context, config_tunable_subset,
)
from layer1_cpu_sandbox.perception.frame_quality import compute_frame_quality
from layer1_cpu_sandbox.perception.frame_overlap import compute_consecutive_overlap
from layer1_cpu_sandbox.evaluation.spatial_diagnostics import compute_spatial_confidence_grid


@dataclasses.dataclass
class AgenticRunReport:
    frame_decision: Dict[str, Any]
    iterations: List[Dict[str, Any]]
    final_config: Dict[str, Any]
    final_action: str
    evaluation_report: Dict[str, Any]
    result: Any = None  # the underlying PipelineResult/RealPipelineResult -- not JSON-serialized by default

    def to_json_safe_dict(self) -> Dict[str, Any]:
        """Everything except `result` (a full dataclass of numpy arrays -- report_generator.py
        already owns turning a result into figures/tables; this report is the agent-decision
        trail alongside it, not a replacement for it)."""
        return {
            "frame_decision": self.frame_decision,
            "iterations": self.iterations,
            "final_config": self.final_config,
            "final_action": self.final_action,
            "evaluation_report": self.evaluation_report,
        }


class AgenticReconstructionOrchestrator:
    def __init__(self, frame_agent: IFrameAgent, reconstruction_agent: IReconstructionAgent,
                  evaluation_agent: IEvaluationAgent, max_iterations: int = 3, cell_size: float = 8.0):
        self.frame_agent = frame_agent
        self.reconstruction_agent = reconstruction_agent
        self.evaluation_agent = evaluation_agent
        self.max_iterations = max_iterations
        self.cell_size = cell_size

    def run(self, dataset, initial_config, run_pipeline_fn: Callable, evaluate_fn: Callable,
             build_eval_context_fn: Callable, n_total_frames: Optional[int] = None) -> AgenticRunReport:
        """`run_pipeline_fn(dataset, config) -> result` and `evaluate_fn(dataset, result) ->
        evaluation` are exactly `run_pipeline`/`evaluate_pipeline_result` (synthetic) or
        `run_real_pipeline`/`evaluate_real_result` (real), pre-bound by the caller to paper
        over their differing extra kwargs (run_name, n_self_consistency_views, ...) -- see
        run_agentic_experiment.py / run_agentic_real_experiment.py for exactly how.
        `build_eval_context_fn` is `build_evaluation_agent_context_synthetic` or `_real`.
        """
        # --- Step 1: Frame Agent decides ONCE, before any reconstruction attempt. ---
        grays = [cv2.cvtColor(f.rgb, cv2.COLOR_RGB2GRAY) for f in dataset.frames]
        quality_scores = [compute_frame_quality(f.rgb) for f in dataset.frames]
        raw_overlap = compute_consecutive_overlap(grays)
        frame_ctx = build_frame_agent_context(
            quality_scores, raw_overlap,
            current_keep_fraction=initial_config.keep_fraction,
            current_min_overall=getattr(initial_config, "min_overall", 0.15),
        )
        frame_decision = self.frame_agent.decide(frame_ctx)
        config = dataclasses.replace(initial_config, keep_fraction=frame_decision.keep_fraction,
                                       min_overall=frame_decision.min_overall)

        # --- Step 2: bounded reconstruction retry loop. ---
        iterations: List[Dict[str, Any]] = []
        last_good_config, last_good_result = None, None
        final_action = "accept"

        for it in range(self.max_iterations):
            try:
                result = run_pipeline_fn(dataset, config)
            except Exception as e:  # noqa: BLE001 -- see module docstring: revert, don't propagate
                iterations.append({
                    "iteration": it, "action": "revert_on_exception",
                    "config_attempted": config_tunable_subset(config),
                    "error": f"{type(e).__name__}: {e}",
                    "note": "pipeline raised on this config; reverted to the last working result",
                })
                if last_good_result is None:
                    raise  # nothing to revert to -- the very first attempt failed, not an agent's fault
                config, result = last_good_config, last_good_result
                final_action = "accept_after_revert"
                break

            last_good_config, last_good_result = config, result
            kept_mask = result.gate.keep_mask
            diag = compute_spatial_confidence_grid(
                result.points.positions[kept_mask], result.confidence.confidence[kept_mask],
                result.confidence.band[kept_mask], cell_size=self.cell_size,
            )
            history_so_far = [{"iteration": r["iteration"], "action": r["action"]} for r in iterations]
            recon_ctx = build_reconstruction_agent_context(
                result, config, diag, attempt_history=history_so_far, n_total_frames=n_total_frames)
            decision = self.reconstruction_agent.decide(recon_ctx)

            gate_changed = decision.prune_below is not None or decision.densify_above is not None
            record = {
                "iteration": it,
                "action": decision.action,
                "config_before_this_attempt": config_tunable_subset(config),
                "decision_reasoning": decision.reasoning,
                "decision_warnings": decision.warnings,
                "agent_confidence": decision.agent_confidence,
                "gate_threshold_changed": gate_changed,
                "n_points_after_gating": int(result.final_positions.shape[0]),
                "mean_confidence_kept": recon_ctx["points"]["mean_confidence_kept"],
                "weak_regions": recon_ctx["spatial_diagnostic"]["weak_regions"],
            }
            if gate_changed:
                record["gate_threshold_change_detail"] = {
                    k: {"from": getattr(config, k, None), "to": getattr(decision, k)}
                    for k in ("prune_below", "densify_above") if getattr(decision, k) is not None
                }
            iterations.append(record)

            if decision.action != "retry":
                final_action = decision.action
                break
            if it == self.max_iterations - 1:
                record["note"] = (f"requested another retry but max_iterations="
                                    f"{self.max_iterations} reached; stopping with this result")
                final_action = "accept_by_budget_exhaustion"
                break
            config = decision.apply_to(config)

        # --- Step 3: Evaluation Agent reads the final result's real metrics. ---
        evaluation = evaluate_fn(dataset, result)
        eval_ctx = build_eval_context_fn(evaluation)
        eval_report = self.evaluation_agent.decide(eval_ctx)

        return AgenticRunReport(
            frame_decision=dataclasses.asdict(frame_decision),
            iterations=iterations,
            final_config=config_tunable_subset(config),
            final_action=final_action,
            evaluation_report=dataclasses.asdict(eval_report),
            result=result,
        )
