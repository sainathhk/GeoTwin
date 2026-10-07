# Stop Conditions — Do Not Build These Unless an Experiment Justifies Them

Per the explicit instruction to prefer a small number of strong innovations
over many weak modules, and to optimize for scientific validity over module
count:

1. **Surface/occlusion completion (generative filling of unseen geometry).**
   Do not build unless Experiment 1/3's completeness numbers show that
   honest gap-reporting is unacceptable for the target use cases
   (disaster assessment, inspection) AND stakeholders explicitly want
   plausible-but-uncertain fill-in over an honest gap. The PS and this
   project's own philosophy favor "insufficient observation" as a valid,
   reportable output — building a completion module by default would work
   against that. If ever built, it must never merge with confidently-observed
   geometry without a persistent, visualized confidence-band distinction.

2. **Full 6-DOF visual-inertial SLAM / bundle adjustment**, beyond the
   GPS/IMU EKF. Build ONLY if Experiment 2 (or real-footage testing) shows
   the EKF-fused pose accuracy is insufficient for the metric-accuracy
   target — do not build it preemptively "because real systems have it."
   The EKF fusion already shows a large, real, measured improvement over
   raw GPS; whether the marginal gain from full VI-SLAM is worth its
   complexity and failure modes (tracking loss, drift) is an empirical
   question, not a given.

3. **A learned dynamic-object segmenter**, beyond the classical residual-flow
   baseline. Build ONLY if the classical baseline's measured precision
   problem (see `perception/dynamic_object_filter.py`) persists at real
   1080p/4K resolution — it may not (the CPU-sandbox failure mode was tied
   to a small object being comparable in size to the resolution's own
   discretization; more pixels per object may resolve this for free).
   Measure before building.

4. **Learned frame-quality model.** The classical scorer is cheap, fast,
   and already discriminates well enough to matter (see Experiment 1's
   `quality_score` term). Replace only if the confidence-vs-error
   correlation on GPU shows `quality_score` is a weak/noisy contributor
   relative to the other four signals (a diagnostic Layer 3's per-signal
   ablation, not built yet, would need to check).

5. **Higher-degree spherical-harmonic view-dependent color**, multi-scale
   Gaussian LOD, or other 3DGS quality-of-life extensions from the wider
   literature. None of these are this project's contribution; add them only
   if visual quality is the bottleneck AFTER the confidence-gating ablation
   is complete, not before (conflating them would make Experiment 1's
   result impossible to attribute cleanly).

6. **Real-time/streaming optimization** (incremental Gaussian updates as
   video streams in, rather than batch processing a captured video). Big
   engineering investment, PS explicitly accepts "near-real-time," and
   premature optimization here risks the time budget for Experiment 1,
   which is load-bearing for the entire novelty claim. Revisit only after
   Day 4 of `docs/SIH_IMPLEMENTATION_PLAN.md`, if time remains.

7. **Consolidating the three duplicate confidence-formula implementations**
   (CPU/pandas, PyTorch/CPU-testable, CUDA) into one code-generated source
   of truth. Real maintenance risk, flagged in
   `layer3_gpu/csrc/confidence_gaussian/confidence_gaussian_kernel.cu`'s
   header comment, but not worth the engineering time until the formula
   itself has stopped changing (i.e. after Experiment 5, if run).
