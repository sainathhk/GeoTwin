# Novelty Candidates

Ranked by evidence strength actually gathered in this repository, not by
ambition.

## 1. Confidence-gated Gaussian densification/pruning — HIGH, primary claim
Standard 3DGS/FastGS densification is driven purely by view-space
photometric gradient magnitude — a signal blind to *why* a region has high
gradient (genuine unresolved detail vs. the optimizer overfitting a single
weakly-supported view). Gating densification by an independent,
evidence-based confidence signal (multi-view redundancy, angle baseline,
depth consistency, frame/pose quality) is a real, mechanistically distinct
addition, not a relabeling of existing gradient-based heuristics.

**Extended design (post-FastGS-QADS review):** three independent signals
now feed densification, not one. (1) Observation confidence — this
project's core signal — is the hard gate; nothing densifies below it,
however strong the other signals look. (2) A temporal+spatial propagation
of that signal (`confidence_propagation.py::ObservationConfidencePropagator`,
design-inspired by the author's own prior FastGS-QADS project's STCP
mechanism, reimplemented fresh for a different underlying signal — see that
file's module docstring) stabilizes it across training rounds instead of
using one noisy snapshot. (3) An optimizer-struggle signal
(`optimizer_state_struggle_signal`, design-inspired by FastGS-QADS's OSAD,
same reimplementation discipline) can ADD densify candidates the gradient
criterion alone misses — but, unlike OSAD in its original context, is
explicitly SUBORDINATE to signal 1: it can never bypass the observation
gate. This is the key place this project's design diverges from its
inspiration, and on purpose — FastGS-QADS had no single-pass coverage-gap
problem to design around; this project's whole thesis is that such gaps
must be respected, not optimized past. Verified on CPU
(`tests/test_layer3_confidence_gaussian_cpu.py::test_optimizer_struggle_signal_cannot_bypass_observation_gate`):
a deliberately huge struggle signal in a poorly-observed region does NOT
trigger densification, while the identical signal on a well-observed
Gaussian correctly does.

**Evidence so far:** CPU-tested formula + gating logic (deterministic, unit
tested), confidence-vs-error correlation validated on synthetic ground truth
(Layer 1), and now the full three-signal orchestration (including realistic
prune-then-densify index bookkeeping through a production 3-D Gaussian
model) validated on CPU with real Adam optimizer state.
**Not yet evidence:** does confidence-gated densification actually beat
gradient-only densification on held-out NOVEL views under the single-pass
constraint, on GPU, on either synthetic-with-3DGS or real footage? This is
the one experiment in `docs/EXPERIMENT_PLAN.md` that determines whether
this claim survives contact with the real representation — `train_gpu.py`
exists now to run it, but has not been executed anywhere yet.

## 2. Honest "insufficient observation" reporting via confidence + coverage — HIGH, differentiator
Most reconstruction demos silently interpolate/hallucinate over gaps.
Combining per-primitive confidence (this project's signal) with the
coverage diagnostic (`occlusion.py`'s never-observable /
observable-but-dropped / observed split) lets the system make a specific,
falsifiable claim: "this facade was never in any frame's field of view,"
vs. "this facade was seen but reconstruction is unreliable here," vs.
"reliable." That three-way distinction, visualized, is unusual and directly
serves the PS's explicit requirement to not overclaim reconstruction of
occluded/unseen surfaces.

## 3. GPS-fusion-first georeferencing (no post-hoc alignment) — MODERATE
Fusing GPS into the pose estimate before reconstruction (rather than
reconstructing in an arbitrary SfM frame and aligning afterward) is a
legitimate structural advantage for metric/georeferencing accuracy, but it
is largely standard practice in GPS-tagged drone photogrammetry, not
something this project invented. Positioned as a supporting design decision,
not a headline contribution.

## 4. Confidence-gated sensor fusion (learned visual residual on top of GPS/IMU EKF) — MODERATE, future work
Using the SAME confidence framework to gate how much a learned visual pose
correction is trusted (vs. falling back to the GPS/IMU prior) would extend
the core idea beyond geometry into pose estimation itself. Not implemented;
flagged in `layer2_interfaces/i_pose_estimator.py` as a concrete research
extension once Layer 3 has a working visual-refinement stage to gate.

## 5. Learned confidence weights (vs. hand-tuned) — LOW-MODERATE on its own, but cheap and defensible
The five weights in the confidence formula are currently reasoned defaults.
Learning them (e.g. logistic regression against real RTK/GCP checkpoint
error) is a small, well-scoped extension with a clear evaluation protocol,
useful as a "we thought about this, here's the upgrade path" answer to a
judge's question, but not sufficient as a standalone contribution.

## What is explicitly NOT claimed as novel
- 3D Gaussian Splatting itself (existing method — FastGS, 3DGS).
- The differentiable tile rasterizer (external dependency, unmodified).
- Classical plane-sweep MVS, EKF sensor fusion, residual-flow dynamic
  detection (all standard, textbook techniques used here for CPU-sandbox
  validation, not claimed as research contributions).
