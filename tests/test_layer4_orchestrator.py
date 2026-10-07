"""End-to-end tests for layer4_agents.orchestrator, run against a REAL (small, fast)
synthetic dataset and the REAL pipeline.py -- not hand-built fixtures. Mock agents stand
in for LLM agents (identical interface -- see mock_agents.py's docstring), which is what
makes it possible to prove the retry loop, config mutation, history tracking, and
fail-safe behavior all actually work, in this network-disabled sandbox.
"""
import dataclasses
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_cpu_sandbox.synthetic.dataset_builder import build_dataset
from layer1_cpu_sandbox.pipeline import run_pipeline, PipelineConfig
from layer1_cpu_sandbox.evaluate_result import evaluate_pipeline_result

from layer4_agents.orchestrator import AgenticReconstructionOrchestrator
from layer4_agents.mock_agents import MockFrameAgent, MockReconstructionAgent, MockEvaluationAgent
from layer4_agents.interfaces import IReconstructionAgent
from layer4_agents.schemas import ReconstructionAgentDecision
from layer4_agents.context_builders import build_evaluation_agent_context_synthetic


def _default_orchestrator(max_iterations=3):
    return AgenticReconstructionOrchestrator(
        frame_agent=MockFrameAgent(),
        reconstruction_agent=MockReconstructionAgent(),
        evaluation_agent=MockEvaluationAgent(),
        max_iterations=max_iterations,
    )


def _synth_run_kwargs():
    dataset = build_dataset(seed=0, extent=25.0)
    run_fn = lambda ds, cfg: run_pipeline(ds, cfg)
    eval_fn = lambda ds, result: evaluate_pipeline_result(ds, result, run_name="agentic_test")
    return dataset, run_fn, eval_fn


def test_orchestrator_end_to_end_produces_complete_report():
    dataset, run_fn, eval_fn = _synth_run_kwargs()
    orch = _default_orchestrator()
    report = orch.run(dataset, PipelineConfig(), run_fn, eval_fn,
                        build_evaluation_agent_context_synthetic, n_total_frames=len(dataset.frames))

    assert report.result is not None
    assert len(report.iterations) >= 1
    assert report.final_action in ("accept", "flag_and_accept", "accept_by_budget_exhaustion", "accept_after_revert")
    assert "verdict" in report.evaluation_report
    d = report.to_json_safe_dict()
    import json
    json.dumps(d)  # the whole report (minus `result`) must be JSON-safe


def test_orchestrator_frame_decision_is_applied_before_first_pipeline_run():
    dataset, run_fn, eval_fn = _synth_run_kwargs()
    orch = _default_orchestrator()
    report = orch.run(dataset, PipelineConfig(keep_fraction=0.75), run_fn, eval_fn,
                        build_evaluation_agent_context_synthetic)
    # Whatever the frame agent decided must match what iteration 0 actually ran with.
    assert report.iterations[0]["config_before_this_attempt"]["keep_fraction"] == \
           report.frame_decision["keep_fraction"]


def test_orchestrator_respects_max_iterations_bound():
    # A reconstruction agent that ALWAYS asks to retry -- the orchestrator must still stop.
    class AlwaysRetryAgent(IReconstructionAgent):
        def decide(self, context):
            return ReconstructionAgentDecision(action="retry", voxel_size=1.9, agent_confidence=0.5,
                                                 reasoning="testing: always retry")

    dataset, run_fn, eval_fn = _synth_run_kwargs()
    orch = AgenticReconstructionOrchestrator(
        frame_agent=MockFrameAgent(), reconstruction_agent=AlwaysRetryAgent(),
        evaluation_agent=MockEvaluationAgent(), max_iterations=3,
    )
    report = orch.run(dataset, PipelineConfig(), run_fn, eval_fn, build_evaluation_agent_context_synthetic)
    assert len(report.iterations) == 3  # never more than max_iterations pipeline runs
    assert report.final_action == "accept_by_budget_exhaustion"
    assert "note" in report.iterations[-1]


def test_orchestrator_config_actually_changes_between_retry_iterations():
    class OnceRetryThenAccept(IReconstructionAgent):
        def __init__(self):
            self.calls = 0
        def decide(self, context):
            self.calls += 1
            if self.calls == 1:
                return ReconstructionAgentDecision(action="retry", voxel_size=1.0, agent_confidence=0.5,
                                                     reasoning="testing: force one retry")
            return ReconstructionAgentDecision(action="accept", agent_confidence=0.8, reasoning="done")

    dataset, run_fn, eval_fn = _synth_run_kwargs()
    orch = AgenticReconstructionOrchestrator(
        frame_agent=MockFrameAgent(), reconstruction_agent=OnceRetryThenAccept(),
        evaluation_agent=MockEvaluationAgent(), max_iterations=3,
    )
    report = orch.run(dataset, PipelineConfig(voxel_size=2.0), run_fn, eval_fn,
                        build_evaluation_agent_context_synthetic)
    assert len(report.iterations) == 2
    assert report.iterations[0]["config_before_this_attempt"]["voxel_size"] == 2.0
    assert report.iterations[1]["config_before_this_attempt"]["voxel_size"] == 1.0  # actually changed
    assert report.final_config["voxel_size"] == 1.0


def test_orchestrator_never_lets_gate_thresholds_change_silently():
    class GateChangingAgent(IReconstructionAgent):
        def decide(self, context):
            return ReconstructionAgentDecision(action="flag_and_accept", prune_below=0.05,
                                                 agent_confidence=0.4, reasoning="testing: touches the gate")

    dataset, run_fn, eval_fn = _synth_run_kwargs()
    orch = AgenticReconstructionOrchestrator(
        frame_agent=MockFrameAgent(), reconstruction_agent=GateChangingAgent(),
        evaluation_agent=MockEvaluationAgent(), max_iterations=2,
    )
    report = orch.run(dataset, PipelineConfig(prune_below=0.12), run_fn, eval_fn,
                        build_evaluation_agent_context_synthetic)
    rec = report.iterations[0]
    assert rec["gate_threshold_changed"] is True
    assert rec["gate_threshold_change_detail"]["prune_below"] == {"from": 0.12, "to": 0.05}


def test_orchestrator_default_mock_agents_never_touch_gate_thresholds():
    dataset, run_fn, eval_fn = _synth_run_kwargs()
    orch = _default_orchestrator()
    report = orch.run(dataset, PipelineConfig(), run_fn, eval_fn, build_evaluation_agent_context_synthetic)
    assert all(not rec["gate_threshold_changed"] for rec in report.iterations)


def test_orchestrator_reverts_on_pipeline_exception_instead_of_crashing():
    dataset, _, eval_fn = _synth_run_kwargs()
    calls = {"n": 0}

    def flaky_run_fn(ds, cfg):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated pipeline failure on the 2nd attempt")
        return run_pipeline(ds, cfg)

    class AlwaysRetryOnce(IReconstructionAgent):
        def __init__(self):
            self.calls = 0
        def decide(self, context):
            self.calls += 1
            if self.calls == 1:
                return ReconstructionAgentDecision(action="retry", voxel_size=1.0, agent_confidence=0.5,
                                                     reasoning="testing: force the flaky 2nd attempt")
            return ReconstructionAgentDecision(action="accept", agent_confidence=0.8, reasoning="unreachable")

    orch = AgenticReconstructionOrchestrator(
        frame_agent=MockFrameAgent(), reconstruction_agent=AlwaysRetryOnce(),
        evaluation_agent=MockEvaluationAgent(), max_iterations=3,
    )
    report = orch.run(dataset, PipelineConfig(), flaky_run_fn, eval_fn, build_evaluation_agent_context_synthetic)
    assert report.final_action == "accept_after_revert"
    assert report.result is not None  # the last GOOD result, not a crash
    assert any(it.get("action") == "revert_on_exception" for it in report.iterations)


def test_orchestrator_history_grows_and_is_visible_to_later_decisions():
    seen_histories = []

    class HistoryRecordingAgent(IReconstructionAgent):
        def __init__(self):
            self.calls = 0
        def decide(self, context):
            seen_histories.append(list(context["attempt_history"]))
            self.calls += 1
            if self.calls < 3:
                return ReconstructionAgentDecision(action="retry", voxel_size=2.0 - 0.1 * self.calls,
                                                     agent_confidence=0.5, reasoning="testing")
            return ReconstructionAgentDecision(action="accept", agent_confidence=0.8, reasoning="testing")

    dataset, run_fn, eval_fn = _synth_run_kwargs()
    orch = AgenticReconstructionOrchestrator(
        frame_agent=MockFrameAgent(), reconstruction_agent=HistoryRecordingAgent(),
        evaluation_agent=MockEvaluationAgent(), max_iterations=5,
    )
    orch.run(dataset, PipelineConfig(), run_fn, eval_fn, build_evaluation_agent_context_synthetic)
    assert len(seen_histories[0]) == 0    # first call: no history yet
    assert len(seen_histories[1]) == 1    # second call: sees the first attempt
    assert len(seen_histories[2]) == 2    # third call: sees both prior attempts
