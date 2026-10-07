# Experiment Plan — What Determines Whether This Project's Claims Hold Up

Every experiment below specifies: hypothesis, independent variable, metric,
and what result would FALSIFY the hypothesis. None of these have been run on
GPU yet; numbers do not exist until they do.

## Experiment 1 (PRIMARY): Confidence-gated vs. gradient-only densification
**Hypothesis:** on held-out (novel, never-trained-on) views, confidence-gated
densification achieves equal-or-better PSNR/SSIM at equal-or-lower Gaussian
count than standard gradient-only densification, specifically in
weakly-observed (single-pass blind-spot-adjacent) regions.
**Independent variable:** densification rule (gradient-only vs.
confidence-gated), everything else held fixed (same init, same schedule,
same dataset).
**Metrics:** novel-view PSNR/SSIM/LPIPS, Gaussian count, and — critically —
metrics computed SEPARATELY for high-coverage vs. low-coverage regions
(using `occlusion.py`'s coverage split) rather than only a scene-wide
average, since a scene-wide average could hide the effect this hypothesis is
actually about.
**Falsification:** confidence-gating shows no significant difference, or
underperforms, in the low-coverage-region-specific metrics. If so, report
that honestly — it would mean the confidence signal isn't discriminative
enough at Gaussian-optimization scale, a real and useful negative result for
`docs/NOVELTY_CANDIDATES.md`.

## Experiment 2: Sensor fusion ablation (GPU-scale confirmation of the CPU finding)
**Hypothesis:** GPS/IMU EKF fusion reduces geometric/metric error vs. raw
GPS, consistent with the CPU-sandbox finding (RMSE 7.33m→2.70m on synthetic
data).
**Metrics:** point-to-point RMSE, control-point RMSE, on the GPU
representation.
**Note:** this is the lowest-risk experiment — the CPU result is already
strong and the mechanism (Kalman filtering reduces variance) is
well-established; this is a confirmation run, not a fishing expedition.

## Experiment 3: Baseline comparison
**Method | What it needs | What it tells us**
1. Classical photogrammetry / COLMAP+MVS — public COLMAP binary — the
   "no learning at all" floor.
2. Standard 3DGS (public repo, default densification) — public 3DGS repo —
   isolates the effect of confidence-gating alone (vs. Experiment 1, which
   holds init/schedule fixed; this baseline uses each method's own
   defaults, which is the fairer "as commonly used" comparison).
3. FastGS (public repo) — the speed-oriented reference point, given the
   author's own prior FastGS-QADS experience.
4. A monocular-video/NeRF baseline (e.g. Instant-NGP or a fast NeRF variant)
   — tests whether an implicit representation is competitive under the
   single-pass constraint despite the representation-choice analysis in
   `docs/ARCHITECTURE.md` predicting it will not be.
5. Our method (confidence-gated 3DGS).
**Metrics:** the full suite (visual/geometry/metric-accuracy/completeness/
efficiency/robustness) from `docs/ARCHITECTURE.md`'s evaluation framework,
run identically across all 5.

## Experiment 4: Robustness sweep
**Independent variables (one at a time, holding others at Layer-1-sandbox
defaults):** GPS noise σ, motion blur strength, JPEG quality, illumination
gamma range, dynamic-object count, flight speed (frame overlap), frame
keep-fraction (viewpoint sparsity).
**Metrics:** degradation curves (metric vs. severity) for our method vs.
baseline 2 (standard 3DGS) — the interesting result is not absolute
performance but whether OUR method degrades more gracefully, since
graceful degradation under exactly these conditions is the PS's explicit
concern.
**Infrastructure:** `layer1_cpu_sandbox/synthetic/degradation.py`'s
functions are directly reusable for this — same noise injection code, GPU
representation as the system under test.

## Experiment 5: Learned confidence weights (optional, time-permitting)
**Method:** collect (evidence signals, actual error) pairs from Experiment
3's runs where ground truth exists; fit weights via logistic/linear
regression against error, compare to the hand-tuned defaults.
**Falsification:** if learned weights don't outperform the defaults by a
meaningful margin, keep the defaults and report the (still useful) negative
result — this is explicitly an optional refinement, not load-bearing for the
core novelty claim.

## Dataset strategy for turning multi-view datasets into single-pass ones
Public aerial/photogrammetry datasets (e.g. UAVid, SenseFly example
datasets, Mill19/UrbanScene3D-style captures) typically contain MULTIPLE
overlapping passes. To create a controlled single-pass experiment: sort
captured frames by acquisition trajectory, then keep only ONE contiguous
sub-trajectory (one flight line) as the "single-pass" input, holding out
the remaining passes' images as extra ground-truth viewpoints for
novel-view evaluation (they were never given to the reconstruction
pipeline). This directly mirrors what `layer1_cpu_sandbox/synthetic/`
already does by construction (one diagonal traverse, not a grid) and keeps
the same train/eval separation discipline used throughout this repo
(`_interpolated_novel_pose` in `evaluate_result.py`, held-out control
points in `metrics_metric_accuracy.py`).
