"""
run_agentic_real_experiment.py
LAYER 4 (AGENTS) -- STATUS: implemented, CPU-tested against synthetic mock video+log
pairs (same caveat as real_data/run_real_experiment.py: not yet run against real
downloaded footage in this sandbox -- no network here to fetch one). Run this yourself
once you have a real video+log pair (see docs/GETTING_STARTED.md "Real data" section).

This is the real-footage counterpart to run_agentic_experiment.py: same Frame /
Reconstruction / Evaluation agent loop, but reading your actual drone video + flight
log instead of a synthetic scene, and using the NO-ground-truth evaluation context
(self-consistency + confidence distribution only -- see context_builders.py).

Usage (mock agents -- default, no API key needed):
    python3 -m layer4_agents.run_agentic_real_experiment \\
        --video flight.mov --log flight.csv --out_dir outputs/agentic_real_run1

Usage (real LLM agents -- requires network + ANTHROPIC_API_KEY, NOT executed in this
sandbox, run this in your own environment):
    python3 -m layer4_agents.run_agentic_real_experiment \\
        --video flight.mov --log flight.csv --out_dir outputs/agentic_real_run1 --use_llm
"""
from __future__ import annotations

import argparse
import json
import os
import time

from layer1_cpu_sandbox.real_data.real_dataset_builder import build_real_dataset
from layer1_cpu_sandbox.real_data.real_pipeline import RealPipelineConfig, run_real_pipeline, evaluate_real_result
from layer1_cpu_sandbox.utils import save_ply, ensure_dir

from .orchestrator import AgenticReconstructionOrchestrator
from .mock_agents import MockFrameAgent, MockReconstructionAgent, MockEvaluationAgent
from .context_builders import build_evaluation_agent_context_real


def _build_agents(use_llm: bool, model: str):
    if not use_llm:
        return MockFrameAgent(), MockReconstructionAgent(), MockEvaluationAgent()
    from .llm_agents import LLMFrameAgent, LLMReconstructionAgent, LLMEvaluationAgent, build_default_client
    client = build_default_client()
    return (LLMFrameAgent(client, model=model), LLMReconstructionAgent(client, model=model),
            LLMEvaluationAgent(client, model=model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="layer4_agents/outputs/agentic_real_run1")
    ap.add_argument("--hfov", type=float, default=84.0, help="assumed horizontal FOV, deg (DJI Phantom 4 Pro spec)")
    ap.add_argument("--target_fps", type=float, default=2.0)
    ap.add_argument("--max_frames", type=int, default=80)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--max_iterations", type=int, default=3)
    ap.add_argument("--use_llm", action="store_true",
                      help="Use real Claude-backed agents. Requires network + ANTHROPIC_API_KEY. "
                           "NOT exercised in the sandbox this code was written in -- see llm_agents.py.")
    ap.add_argument("--model", type=str, default="claude-sonnet-5")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    t_start = time.perf_counter()

    print("[1/4] Loading video + flight log...")
    dataset = build_real_dataset(args.video, args.log, camera_hfov_deg=args.hfov,
                                   target_fps=args.target_fps, max_frames=args.max_frames,
                                   resize_to=(args.width, args.height))
    print(f"      video: {dataset.video_info.width}x{dataset.video_info.height} @ "
          f"{dataset.video_info.fps:.1f}fps -- log format: {dataset.log_format_detected}")
    print(f"      extracted {len(dataset.frames)} frames")

    agent_kind = "LLM (Claude, model=" + args.model + ")" if args.use_llm else "mock (deterministic, offline)"
    print(f"[2/4] Running AGENTIC pipeline with {agent_kind} agents "
          f"(max_iterations={args.max_iterations})...")
    frame_agent, recon_agent, eval_agent = _build_agents(args.use_llm, args.model)
    orchestrator = AgenticReconstructionOrchestrator(
        frame_agent=frame_agent, reconstruction_agent=recon_agent, evaluation_agent=eval_agent,
        max_iterations=args.max_iterations,
    )
    report = orchestrator.run(
        dataset, RealPipelineConfig(),
        run_pipeline_fn=lambda ds, cfg: run_real_pipeline(ds, cfg),
        evaluate_fn=lambda ds, result: evaluate_real_result(ds, result),
        build_eval_context_fn=build_evaluation_agent_context_real,
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
    print("      NOTE: no ground truth exists for real footage -- this verdict is based on "
          "self-consistency and confidence distribution only, same caveat as run_real_experiment.py.")

    print("[3/4] Saving point cloud...")
    save_ply(os.path.join(args.out_dir, "reconstruction_agentic_real.ply"),
              report.result.final_positions, report.result.final_colors)

    print("[4/4] Saving agentic_report.json and summary.json...")
    with open(os.path.join(args.out_dir, "agentic_report.json"), "w") as f:
        json.dump(report.to_json_safe_dict(), f, indent=2)

    summary = {
        "video_path": args.video, "log_path": args.log,
        "log_format_detected": dataset.log_format_detected, "agent_kind": agent_kind,
        "max_iterations": args.max_iterations,
        "n_frames_extracted": len(dataset.frames),
        "n_reconstruction_attempts": len(report.iterations),
        "final_action": report.final_action,
        "final_config": report.final_config,
        "n_final_points": int(report.result.final_positions.shape[0]),
        "evaluation_verdict": report.evaluation_report["verdict"],
        "evaluation_key_numbers": report.evaluation_report["key_numbers"],
        "total_wall_time_s": round(time.perf_counter() - t_start, 2),
        "NOTE": "No ground truth exists for real footage. Agent DECISIONS are recorded in "
                 "agentic_report.json; the underlying numbers come from the exact same "
                 "unmodified real_pipeline.py run_real_experiment.py uses.",
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s. Artifacts written to: {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
