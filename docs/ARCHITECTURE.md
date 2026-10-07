# Architecture Rationale

## 1. Why three layers, and why Layer 1 is not "the algorithm downgraded"

Layer 1 exists to answer one question cheaply: **is the pipeline's logic
correct?** Frame selection, sensor fusion, multi-view fusion, confidence
scoring, evaluation metrics, ablation infrastructure — all of that is
representation-independent. Validating it against classical CV / NumPy on a
synthetic scene with known ground truth is orders of magnitude cheaper than
validating it by staring at 3DGS renders on real drone footage and guessing
whether an improvement is real. Layer 1's classical MVS/point-cloud
representation is a **stand-in for a differentiable renderer**, not a claim
that classical MVS is what ships. `PrototypePointRepresentation` is named
that way on purpose — it is not called `GaussianModel`.

## 2. 3-D representation choice: point cloud vs mesh vs NeRF vs 3DGS

| Criterion | Point cloud | Mesh | NeRF (implicit) | 3D Gaussian Splatting |
|---|---|---|---|---|
| Fast reconstruction from sparse/single-pass views | Fast but noisy | Slow (needs clean surface extraction) | Slow to train | Fast (minutes, not hours) |
| Visual quality | Poor (gaps, no shading model) | Good if topology is clean | Excellent | Excellent |
| Metric measurement | OK (direct 3-D coords) | Good (watertight distances) | Poor (implicit field, expensive to query for distances) | Good (explicit 3-D coords, same as points) |
| Georeferencing | Trivial (points carry world coords) | Trivial | Awkward (needs a separate extraction step) | Trivial |
| Handling incomplete/single-pass views | Degrades gracefully (just sparser) | Degrades badly (holes, non-manifold) | Degrades badly (floaters, blur in unseen regions) | Degrades gracefully (low-opacity/absent Gaussians, exactly where confidence.py flags it) |
| Memory | Low | Low-medium | High (network weights + query cost) | Medium (explicit primitives, but far fewer than voxels) |
| Rendering speed | Fast (splat) | Fast (rasterize) | Slow (per-ray network eval) | Very fast (tile rasterizer, real-time) |
| SIH demo feasibility (progressive reconstruction, interactive camera, live confidence overlay) | OK | Poor (offline meshing step breaks "progressive") | Poor (too slow for live interaction) | Excellent |

**Verdict: 3D Gaussian Splatting**, primarily because of two rows that matter
most for THIS problem specifically: it degrades *gracefully and legibly*
under incomplete single-pass coverage (a Gaussian can simply be
low-opacity/absent, which is directly interpretable as "unobserved" — see
confidence-gating), and it renders fast enough for a genuinely interactive
demo. This is a reasoned choice given the single-pass constraint, not a
default because of prior 3DGS/FastGS experience.

**This verdict is about what gets *trained and rendered*, not about every
output format the project ever produces.** The "Mesh" column's "Poor ...
offline meshing step breaks 'progressive'" verdict is specifically about
mesh as the *live, interactive* representation — true, and still the reason
3DGS won this table. It says nothing about producing a mesh as one more
*offline, after-the-fact export* of whatever the trained representation
already contains, purely for interoperability (a `.obj` opens in
Blender/MeshLab/a phone AR viewer; a `.pt` full Gaussian state does not).
`layer1_cpu_sandbox/reconstruction/mesh_export.py` is exactly that: it runs
once, after training/fusion finishes, turns the already-trained positions
into a triangle mesh, and has no bearing on what's trained, optimized, or
rendered live during the demo. Same logic as section 3 below applies to
*why* it's Open3D and not a reimplementation: Poisson/ball-pivoting surface
reconstruction is a solved problem, and reimplementing it would spend
effort on zero-novelty code instead of the confidence framework that
actually is this project's contribution.

## 3. Why we do not reimplement the tile rasterizer

Differentiable Gaussian rasterization (the tile-based forward+backward CUDA
rasterizer used by 3DGS and FastGS) is a solved, extremely well-optimized
problem with a public, mature implementation
(`diff-gaussian-rasterization`). Reimplementing it from scratch would be
several thousand lines of highly-tuned CUDA that contributes **zero** novelty
— it is explicitly the part of the system the brief says not to treat as the
contribution ("do not build a generic COLMAP + 3DGS pipeline" is about the
*overall* pipeline being generic, not about every single component needing
to be reinvented). The correct scoping call is: reuse the rasterizer,
concentrate engineering and research effort on the part that is actually
new — confidence-aware densification/pruning — and integrate at the one
well-defined interface boundary (`layer2_interfaces/i_renderer.py`).

## 4. The confidence framework — final formulation

Implemented identically in three places (by design, so CPU validation
transfers directly to the GPU path):
`layer1_cpu_sandbox/reconstruction/confidence.py` (CPU, pandas/numpy),
`layer3_gpu/python/confidence_gaussian_model.py::GaussianConfidenceStats`
(CPU-testable PyTorch), and
`layer3_gpu/csrc/confidence_gaussian/confidence_gaussian_kernel.cu` (CUDA,
unexecuted). All three compute:

```
confidence = exp( Σ_i  w_i * log(score_i + ε) )        # weighted geometric mean
score_obs         = 1 - exp(-(observation_count - 1) / τ_obs)
score_angle       = clip(view_angle_spread / target_spread, 0, 1)
score_consistency = mean plane-sweep / MVS cost-curve sharpness, per contributing pixel
score_quality     = mean classical/learned frame-quality score
score_pose        = mean pose confidence (1 - normalized GPS/IMU-EKF position uncertainty)

if observation_count <= 1: confidence = min(confidence, single_view_cap)   # hard cap, see below
```

Default weights (`obs=0.30, angle=0.20, consistency=0.25, quality=0.15,
pose=0.10`) are a reasoned CPU-sandbox starting point, not a claimed
optimum — see `docs/EXPERIMENT_PLAN.md` for how to learn them once real
GT/RTK checkpoints exist.

**Why a weighted geometric mean, not an arithmetic mean:** an arithmetic mean
lets one excellent signal compensate for a catastrophically bad one. A
photometrically "perfect" match observed from a single, barely-moved
viewpoint should NOT score as confidently as the arithmetic mean of its
components would suggest — the geometric mean forces every factor to be
simultaneously reasonable, matching how photogrammetrists actually reason
about triangulation reliability.

**Why the single-observation hard cap:** a matching cost curve computed from
exactly one stereo pair can still be a confidently-wrong match (see the
periodic-texture aliasing risk discussed in `synthetic/scene_generator.py`'s
texture-jitter fix). Requiring independent re-confirmation from a second,
sufficiently different viewpoint before a point can be called "High
confidence" mirrors standard geodetic/photogrammetric practice for control
points.

**Validated, not asserted:** on the Layer-1 synthetic sandbox run in
`layer1_cpu_sandbox/outputs/run1/`, confidence has a statistically
significant negative correlation with actual distance-to-ground-truth-surface
error (Pearson r≈-0.16, Spearman ρ≈-0.23, p<1e-67, n=5799). Confidence deciles
show mean error dropping from ~24m (lowest-confidence decile) to ~8m
(highest-confidence decile) — see `confidence_vs_error.png`. These are toy
CPU-sandbox numbers (meter-scale error is a classical-MVS-at-160×120-px
artifact, not a claim about final system accuracy) but they are REAL,
reproducible, and they are exactly the check that would have falsified the
core novelty if it hadn't held up.

## 5. Why the GPS/IMU EKF sits ahead of everything else

Fusing GPS directly into the pose estimate means the reconstruction lands in
an ALREADY approximately-georeferenced coordinate frame — unlike vision-only
SfM, which needs a separate similarity-transform alignment step (and can hide
systematic bias behind a good-looking alignment) before any metric claim is
possible. This is a structural, not cosmetic, advantage for a PS that
explicitly requires georeferenced, metrically accurate output without heavy
reliance on ground control points.
