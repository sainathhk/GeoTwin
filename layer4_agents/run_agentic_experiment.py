"""
run_agentic_experiment.py
LAYER 4 (AGENTS) -- STATUS: implemented; mock-agent path CPU-tested and CPU-executed
here (see /outputs of a run); --use_llm path implemented but NOT executed in this
sandbox (no network -- see llm_agents.py's docstring).

Builds a synthetic single-pass drone dataset, then runs it through the AGENTIC
pipeline instead of a single fixed PipelineConfig: a Frame Agent sets frame-selection
parameters once, a Reconstruction Agent reviews the result and may request bounded
retries with adjusted settings, and an Evaluation Agent turns the final real metrics
into a verdict + diagnosis. Every decision is saved to agentic_report.json alongside
the same kind of artifacts run_experiment.py already produces (PLY, summary.json),
so this is additive to the existing Layer-1 experiment script, not a replacement --
run both; they answer different questions (raw pipeline numbers vs. what an agent
layer did with them).

Usage (mock agents -- default, no API key needed, this is what's CPU-tested here):
    python3 -m layer4_agents.run_agentic_experiment --out_dir outputs/agentic_run1

Usage (real LLM agents -- requires `pip install -r requirements-agents.txt` and
ANTHROPIC_API_KEY set; NOT executed in this sandbox, run this in your own environment):
    python3 -m layer4_agents.run_agentic_experiment --out_dir outputs/agentic_run1 --use_llm
"""
from __future__ import annotations

import argparse
import json
import os
import time

from layer1_cpu_sandbox.synthetic.dataset_builder import build_dataset
from layer1_cpu_sandbox.pipeline import PipelineConfig, run_pipeline
from layer1_cpu_sandbox.evaluate_result import evaluate_pipeline_result
from layer1_cpu_sandbox.utils import save_ply, ensure_dir

from .orchestrator import AgenticReconstructionOrchestrator
from .mock_agents import MockFrameAgent, MockReconstructionAgent, MockEvaluationAgent
from .context_builders import build_evaluation_agent_context_synthetic


def _build_agents(use_llm: bool, model: str):
    if not use_llm:
        return MockFrameAgent(), MockReconstructionAgent(), MockEvaluationAgent()
    # Deferred import: only users who pass --use_llm need the `anthropic` package
    # installed at all (see llm_agents.py's docstring on why this matters here).
    from .llm_agents import LLMFrameAgent, LLMReconstructionAgent, LLMEvaluationAgent, build_default_client
    client = build_default_client()
    return (LLMFrameAgent(client, model=model), LLMReconstructionAgent(client, model=model),
            LLMEvaluationAgent(client, model=model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="layer4_agents/outputs/agentic_run1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extent", type=float, default=40.0)
    ap.add_argument("--max_iterations", type=int, default=3)
    ap.add_argument("--use_llm", action="store_true",
                      help="Use real Claude-backed agents instead of the deterministic mock agents. "
                           "Requires network + ANTHROPIC_API_KEY. NOT exercised in the sandbox this "
                           "code was written in -- see llm_agents.py.")
    ap.add_argument("--model", type=str, default="claude-sonnet-5",
                      help="Only used with --use_llm. Check docs.claude.com for the current "
                           "recommended model string before a real run.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    t_start = time.perf_counter()

    print(f"[1/4] Building synthetic single-pass drone dataset (seed={args.seed})...")
    dataset = build_dataset(seed=args.seed, extent=args.extent)
    print(f"      {len(dataset.frames)} frames, {len(dataset.scene.triangles)} scene triangles")

    agent_kind = "LLM (Claude, model=" + args.model + ")" if args.use_llm else "mock (deterministic, offline)"
    print(f"[2/4] Running AGENTIC pipeline with {agent_kind} agents "
          f"(max_iterations={args.max_iterations})...")
    frame_agent, recon_agent, eval_agent = _build_agents(args.use_llm, args.model)
    orchestrator = AgenticReconstructionOrchestrator(
        frame_agent=frame_agent, reconstruction_agent=recon_agent, evaluation_agent=eval_agent,
        max_iterations=args.max_iterations,
    )
    report = orchestrator.run(
        dataset, PipelineConfig(name="agentic_full_method"),
        run_pipeline_fn=lambda ds, cfg: run_pipeline(ds, cfg),
        evaluate_fn=lambda ds, result: evaluate_pipeline_result(ds, result, run_name="agentic_run"),
        build_eval_context_fn=build_evaluation_agent_context_synthetic,
        n_total_frames=len(dataset.frames),
    )
    print(f"      frame agent: keep_fraction={report.frame_decision['keep_fraction']:.2f}, "
          f"min_overall={report.frame_decision['min_overall']:.2f}")
    print(f"      reconstruction: {len(report.iterations)} attempt(s), final_action={report.final_action}")
    for it in report.iterations:
        gate_note = " [GATE THRESHOLD CHANGED]" if it["gate_threshold_changed"] else ""
        print(f"        iter {it['iteration']}: action={it['action']}{gate_note} -- {it['decision_reasoning']}")
    print(f"      evaluation verdict: {report.evaluation_report['verdict']} -- "
          f"{report.evaluation_report['diagnosis']}")

    print("[3/4] Saving point cloud...")
    save_ply(os.path.join(args.out_dir, "reconstruction_agentic.ply"),
              report.result.final_positions, report.result.final_colors)

    print("[4/4] Saving agentic_report.json and summary.json...")
    with open(os.path.join(args.out_dir, "agentic_report.json"), "w") as f:
        json.dump(report.to_json_safe_dict(), f, indent=2)

    summary = {
        "seed": args.seed, "extent": args.extent, "agent_kind": agent_kind,
        "max_iterations": args.max_iterations,
        "n_frames_total": len(dataset.frames),
        "n_reconstruction_attempts": len(report.iterations),
        "final_action": report.final_action,
        "final_config": report.final_config,
        "n_final_points": int(report.result.final_positions.shape[0]),
        "evaluation_verdict": report.evaluation_report["verdict"],
        "evaluation_key_numbers": report.evaluation_report["key_numbers"],
        "total_wall_time_s": round(time.perf_counter() - t_start, 2),
        "NOTE": "Agent DECISIONS are recorded in agentic_report.json; the underlying "
                 "reconstruction numbers come from the exact same unmodified pipeline.py "
                 "run_experiment.py uses. This script demonstrates the agent layer, it does "
                 "not change what the deterministic pipeline computes.",
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s. Artifacts written to: {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
