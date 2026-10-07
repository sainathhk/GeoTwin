"""
colab_smoke_test.py
STATUS: implemented, NOT YET RUN ANYWHERE. This is the first thing to run in
Environment B after `python3 setup.py build_ext --inplace` succeeds.

What it checks (in order -- stop and fix at the first failure):
  1. The compiled `confidence_gaussian_cuda` extension is importable and a
     CUDA device is actually visible to torch.
  2. `compute_confidence` on a handful of hand-built stat vectors matches
     the CPU reference formula (GaussianConfidenceStats.confidence() /
     layer1_cpu_sandbox/reconstruction/confidence.py) to float32 precision
     -- i.e. the CUDA kernel is not just "runs without crashing" but
     numerically CORRECT against the already-CPU-validated formula.
  3. `gate_densify_prune` reproduces the same single-observation-cap
     behaviour already proven on CPU in confidence_gaussian_model.py's
     __main__ block.
  4. A rough throughput number (Gaussians/sec for compute_confidence at
     N=1,000,000) -- printed, NOT asserted against any target, since no
     real GPU throughput number should be claimed until this has actually
     been measured here.

Run with:
    cd layer3_gpu && python3 setup.py build_ext --inplace && python3 python/colab_smoke_test.py
"""
from __future__ import annotations
import math
import os
import time
import sys

import torch

# `setup.py build_ext --inplace` (from layer3_gpu/) drops the compiled .so directly
# into layer3_gpu/, i.e. ONE DIRECTORY UP from this file's own location
# (layer3_gpu/python/). Python does NOT put the invoked script's parent
# directories on sys.path automatically, so running this as
# `python3 layer3_gpu/python/colab_smoke_test.py` from the repo root previously
# failed with "No module named 'confidence_gaussian_cuda'" even after a
# successful build. Fix: explicitly add layer3_gpu/ to sys.path, relative to
# this file, regardless of the current working directory or how it's invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cpu_reference_confidence(obs_count, angle_min, angle_max, consistency_sum, quality_sum,
                              pose_conf_sum, n_accum, weights, obs_tau, target_spread, single_view_cap):
    eps = 1e-6
    obs_score = 1.0 - torch.exp(-(obs_count - 1).clamp(min=0) / obs_tau)
    spread = (angle_max - angle_min).clamp(min=0)
    spread = torch.where(torch.isfinite(spread), spread, torch.zeros_like(spread))
    angle_score = (spread / target_spread).clamp(0, 1)
    n_safe = n_accum.clamp(min=1)
    consistency_score = (consistency_sum / n_safe).clamp(0, 1)
    quality_score = (quality_sum / n_safe).clamp(0, 1)
    pose_score = (pose_conf_sum / n_safe).clamp(0, 1)
    log_conf = (weights["obs"] * torch.log(obs_score + eps) + weights["angle"] * torch.log(angle_score + eps) +
                weights["consistency"] * torch.log(consistency_score + eps) +
                weights["quality"] * torch.log(quality_score + eps) + weights["pose"] * torch.log(pose_score + eps))
    conf = torch.exp(log_conf)
    single_view = obs_count <= 1
    return torch.where(single_view, conf.clamp(max=single_view_cap), conf)


def main():
    print("[1/5] Checking CUDA availability and extension import...")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False. This script must run on a "
              "Colab GPU runtime (Runtime -> Change runtime type -> GPU).")
        sys.exit(1)
    try:
        import confidence_gaussian_cuda as cgc
    except ImportError as e:
        print(f"FAIL: could not import confidence_gaussian_cuda -- did you run "
              f"`python3 setup.py build_ext --inplace` in layer3_gpu/ first? ({e})")
        sys.exit(1)
    print(f"      OK -- device: {torch.cuda.get_device_name(0)}")

    print("[2/5] Verifying compute_confidence matches the CPU reference formula...")
    device = "cuda"
    weights = dict(obs=0.30, angle=0.20, consistency=0.25, quality=0.15, pose=0.10)
    obs_tau, target_spread, single_view_cap = 1.5, math.radians(6.0), 0.55

    torch.manual_seed(0)
    N = 5000
    obs_count = torch.randint(1, 6, (N,), device=device).float()
    angle_min = torch.rand(N, device=device) * 0.05
    angle_max = angle_min + torch.rand(N, device=device) * 0.3
    consistency_sum = torch.rand(N, device=device) * obs_count
    quality_sum = torch.rand(N, device=device) * obs_count
    pose_conf_sum = torch.rand(N, device=device) * obs_count
    n_accum = obs_count.clone()

    conf_cuda = cgc.compute_confidence(obs_count, angle_min, angle_max, consistency_sum, quality_sum,
                                        pose_conf_sum, n_accum, weights["obs"], weights["angle"],
                                        weights["consistency"], weights["quality"], weights["pose"],
                                        obs_tau, target_spread, single_view_cap)
    conf_ref = cpu_reference_confidence(obs_count, angle_min, angle_max, consistency_sum, quality_sum,
                                         pose_conf_sum, n_accum, weights, obs_tau, target_spread, single_view_cap)
    max_abs_diff = (conf_cuda - conf_ref).abs().max().item()
    print(f"      max |CUDA - reference| = {max_abs_diff:.3e}")
    assert max_abs_diff < 1e-4, "FAIL: CUDA compute_confidence diverges from the CPU reference formula"
    print("      PASS")

    print("[3/5] Verifying gate_densify_prune single-observation cap behaviour...")
    confidence = torch.tensor([0.82, 0.38], device=device)  # matches confidence_gaussian_model.py's CPU test
    opacity = torch.tensor([0.95, 0.95], device=device)
    grad_accum = torch.tensor([0.001, 0.001], device=device)
    prune_mask, densify_mask = cgc.gate_densify_prune(confidence, opacity, grad_accum, 0.05, 0.12, 0.0002, 0.60)
    assert bool(densify_mask[0]) is True and bool(densify_mask[1]) is False, \
        "FAIL: densify gating does not match the CPU-validated behaviour"
    print("      PASS")

    print("[4/5] Verifying the OSAD-inspired struggle signal cannot bypass the observation gate...")
    # Same adversarial case validated on CPU in
    # tests/test_layer3_confidence_gaussian_cpu.py::test_optimizer_struggle_signal_cannot_bypass_observation_gate
    conf3 = torch.tensor([0.85, 0.05, 0.85], device=device)   # G1 is a poorly-observed blind spot
    opacity3 = torch.full((3,), 0.95, device=device)
    grad3 = torch.zeros(3, device=device)                      # no gradient trigger at all
    struggle_on_blindspot = torch.tensor([0.0, 50.0, 0.0], device=device)  # huge struggle signal on G1
    _, densify3 = cgc.gate_densify_prune(conf3, opacity3, grad3, 0.05, 0.12, 0.0002, 0.60,
                                          optimizer_struggle_signal=struggle_on_blindspot, struggle_thresh=3.0)
    assert bool(densify3[1]) is False, \
        "FAIL: a huge optimizer-struggle signal in a poorly-observed region bypassed the confidence gate"
    struggle_on_well_observed = torch.tensor([50.0, 0.0, 0.0], device=device)
    _, densify3b = cgc.gate_densify_prune(conf3, opacity3, grad3, 0.05, 0.12, 0.0002, 0.60,
                                           optimizer_struggle_signal=struggle_on_well_observed, struggle_thresh=3.0)
    assert bool(densify3b[0]) is True, \
        "FAIL: the same struggle signal on a well-observed Gaussian should correctly trigger densification"
    print("      PASS -- struggle signal correctly gated by observation confidence, matching the CPU-validated design")

    print("[5/5] Rough throughput check (compute_confidence @ N=1,000,000)...")
    N_big = 1_000_000
    big = {k: torch.rand(N_big, device=device) for k in
           ["obs_count", "angle_min", "angle_max", "consistency_sum", "quality_sum", "pose_conf_sum", "n_accum"]}
    big["obs_count"] = torch.randint(1, 8, (N_big,), device=device).float()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = cgc.compute_confidence(big["obs_count"], big["angle_min"], big["angle_max"], big["consistency_sum"],
                                    big["quality_sum"], big["pose_conf_sum"], big["n_accum"], weights["obs"],
                                    weights["angle"], weights["consistency"], weights["quality"], weights["pose"],
                                    obs_tau, target_spread, single_view_cap)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    print(f"      {N_big:,} Gaussians in {dt*1000:.2f} ms/call ({N_big/dt:,.0f} Gaussians/sec) "
          f"on {torch.cuda.get_device_name(0)} -- REAL measured number, record it in "
          f"docs/EXPERIMENT_PLAN.md once this has actually been run.")

    print("\nALL CHECKS PASSED. The CUDA extension is numerically consistent with the "
          "already CPU-validated confidence formula and gating logic.")


if __name__ == "__main__":
    main()
