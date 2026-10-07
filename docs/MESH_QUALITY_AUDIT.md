# Mesh quality audit -- full_state_iter8000.pt, 2026-09

> **UPDATE (second pass, after the input video was provided).** The first pass
> called the depth-estimation stage the root cause and the mesh stage a
> contributing bug. With the actual footage in hand, the diagnosis is now
> *conclusive and stronger*, and one claim below has been corrected:
>
> **The fused point cloud is not a scene at all -- it is the camera's viewing
> frustum, filled with points.** Fitting cross-sectional width against distance
> along the cone axis gives R^2 = 0.9990 with an apex intercept of 0.7% of the
> axis length (a true geometric apex), and implied opening angles of 73-85 deg
> x 56-63 deg against the camera's actual 84.0 x 53.7 deg -- matching the
> frustum to within about 2 degrees. Every pixel was back-projected to an
> essentially arbitrary depth inside the tested range.
>
> **Why the renders look excellent anyway** (PSNR 30.0 / SSIM 0.908, visually
> near-identical to the input): photometric loss constrains only where a
> Gaussian PROJECTS, not where it SITS. A Gaussian anywhere along a pixel's
> viewing ray renders to that same pixel. This is the shape-radiance ambiguity,
> and it means **no image-space metric can detect this failure** -- which is
> exactly why `reconstruction/frustum_check.py` now exists and is wired into
> both `real_pipeline.py` (on the seed cloud, before GPU time is spent) and
> `export_obj.py` (on trained Gaussians, before you trust a mesh).
>
> **Why the footage makes this near-inevitable:** the clip is 14.2 s, 427
> frames, 1280x720, DJI Air 3, and it is an *oblique* shot with the **horizon
> visible in frame** -- true depths therefore run from roughly 80 m at the
> bottom of frame to effectively infinity at the horizon, against a depth sweep
> capped at `depth_max = 100.0`. Most pixels have **no correct hypothesis in the
> cost volume at all**, so `argmin` returns noise. Recovered relative pose
> between frames 280 and 400 also shows motion only 51.7 deg off the optical
> axis (|t_z| = 0.62), i.e. **forward-dominant flight** -- the worst case for
> triangulation, since parallax is radial from the focus of expansion and points
> near the direction of travel have near-zero parallax at any baseline. SIFT
> matching over that 4 s baseline yielded only 65 essential-matrix inliers, all
> within 250 px of the image centre.
>
> **Correction to the first pass:** the first pass reported a median
> triangulation angle spread of 14.7 deg from `confidence_angle_min/max` and
> treated that as evidence of healthy parallax. That reading was wrong -- those
> angles are measured at each Gaussian's *current (incorrect)* position, so for
> points spread through a frustum the metric is partly circular and cannot be
> used to certify parallax. The direct two-view estimate from the video above
> supersedes it.
>
> **Consequence:** no meshing algorithm can fix this checkpoint, and neither can
> re-exporting it. The mesh-stage fixes from the first pass are real and worth
> keeping (they are measurable -- 1049 -> 5 components), but they were never
> going to recover geometry that was never estimated. The fix is upstream, and
> for this particular footage it is partly a *capture* problem, not only a code
> problem. See "What to actually do" below.

## What to actually do (ranked, most effective first)

1. **Re-shoot, or re-crop, so the reconstructable region is bounded.** Mask out
   the sky and far field entirely and keep only the near-ground portion of each
   frame, then set `depth_min`/`depth_max` from real flight altitude and gimbal
   pitch (for a camera at height h and depression angle theta, ground depth is
   about h/sin(theta)). For new capture: steep oblique to nadir (-60 to -90 deg
   gimbal), and a grid, orbit, or lawnmower pattern rather than a single 14 s
   forward pass. Forward translation along the view axis is the single worst
   motion for this.
2. **Read the two new diagnostics before spending GPU time.**
   `real_pipeline.py` now prints the per-frame boundary-hit fraction and runs
   `check_frustum_shape` on the seed cloud; a frustum verdict there means stop
   and fix depth, not train.
3. **Use COLMAP (or a commercial photogrammetry tool) for the SIH deliverable.**
   This is the pragmatic recommendation. A mature SfM+MVS pipeline handles wide
   depth ranges and will *refuse* to reconstruct rather than silently emit a
   frustum, which is the property this project's own pipeline was missing. Feed
   its sparse/dense output in as the Gaussian seed if you still want 3DGS in the
   loop.
4. **Only then** revisit `depth_max`, `n_depths`, and
   `depth_boundary_reject=True` in `RealPipelineConfig`, and retrain.

Requested: take ownership of the Gaussian -> mesh pipeline and fix
`mesh_iter8000.obj` (409,500 verts / 814,047 tris, visually a small,
iridescent, incoherent shard rather than a street scene) so it produces a
geometrically faithful model. This document is the "audit trail" the
original request asked for: what was inspected, what was actually wrong,
what got fixed and validated against the real checkpoint, and what is
diagnosed-but-not-yet-fixed because fixing it needs real video/camera data
that was not available in the environment this audit was performed in.

**Read this before changing depth_min/max/n_depths, mesh export thresholds,
or normal estimation again** -- the numbers below are specific to this
checkpoint; re-derive them (the tools to do so are the point of this audit)
rather than copying them to a different run.

## TL;DR

- The 3DGS training itself is not the primary problem. Photometric fit is
  reasonable (frame 12: PSNR 29.3 / SSIM 0.889 on a TRAINING view, with your
  own eval script's "memorization risk" caveat attached -- frames 0 and 7
  are visibly worse and show ghosting/doubling, consistent with local pose
  or geometric inconsistency in parts of the trajectory).
- `mesh_export.py`'s Poisson path had two real, fixable bugs: it estimated
  normals with generic point-cloud PCA (ignoring each Gaussian's own
  scale+rotation covariance), and its confidence gate default
  (`--min_confidence 0.15`) was a **complete no-op** on this checkpoint --
  100% of Gaussians cleared it, because it had never been checked against a
  real confidence distribution. Both are fixed below, and the fix measurably
  improves mesh topology (debris components: 1049 -> 5; largest-component
  vertex fraction: 97.8% -> 99.7%).
- **Neither of those was the dominant cause of the bad shape.** An ablation
  (below) isolating normals-only and filtering-only fixes showed neither
  fixed the fundamental geometry. Visualizing the raw, trained Gaussian
  *positions* -- before any meshing at all -- already shows a
  regularly-banded, pyramid/frustum-shaped point cloud. That traces to
  `depth_estimation.py`'s plane-sweep MVS: only 32 LINEARLY-spaced depth
  hypotheses over a 3-100m range (world units are real meters --
  `real_data/geo_utils.py`), on a checkpoint whose Gaussians span >150m from
  their centroid. Coarse quantization (regular banding, directly visible in
  the point cloud) plus range clamping (real geometry beyond depth_max
  aliasing onto the boundary) plausibly explains the observed shape, given a
  moving single-pass camera. This needed real video/poses to fully confirm
  and fix at the source; the quantization half of the fix (log spacing, more
  hypotheses) is applied and unit-tested, the range half needs your own
  footage (a new automatic coverage diagnostic is added specifically so you
  can check it).

## Pipeline trace

```
VIDEO
  |
FRAME EXTRACTION + DJI TELEMETRY POSES  <-- raw telemetry, NOT bundle-adjusted
  |                                          (real_pipeline.py docstring says so explicitly)
PLANE-SWEEP MVS DEPTH                    <-- ROOT CAUSE: 32 linear bins, 3-100m
  |                                          (depth_estimation.py) -- FIXED (quantization),
  |                                          FLAGGED (range; needs your footage to confirm)
FUSE TO POINT CLOUD                      <-- faithfully reproduces the quantization
  |                                          (prototype_point_repr.fuse_point_cloud)
3DGS TRAINING (8000 iters, RGB loss only)<-- weak position gradient pressure; does not
  |                                          fully correct bad seeding, confirmed by
  |                                          visualizing the TRAINED positions directly
CHECKPOINT (full_state_iter8000.pt)      <-- inspected directly, see "Checkpoint numbers"
  |
GAUSSIAN EXPORT + MESH GENERATION        <-- two real bugs, FIXED + unit-tested
  (mesh_export.py / export_obj.py)           (covariance normals, no-op confidence gate)
  |
mesh_iter8000.obj                        <-- topology measurably cleaner after the fix,
                                              fundamental shape NOT fixed by this stage
                                              alone (see ablation) -- inherited from upstream
```

## Checkpoint numbers (full_state_iter8000.pt, 150,000 Gaussians, sh_degree=2)

```
spatial extent:            bbox diag ~323 units; dist-from-centroid p50=36.7, p99=106.6, max=154.1
opacity:                   p10=0.089  p50=0.817  p90=0.9994   (smooth, no natural split)
confidence (blended):      p10=0.477  p25=0.807  p50=0.870    -- histogram is trimodal, with a
                                                                   clean, near-empty valley around
                                                                   0.51-0.56 (only ~25-30 Gaussians
                                                                   out of 150,000 fall there)
confidence_n_accum:        99.95% nonzero -- NOT the "all-zero / pre-fix checkpoint" case
                                              export_obj.py already warns about
anisotropy (max/min axis): p50=29.6, 85.9% of Gaussians have anisotropy>5 (already disk-like --
                                              real, usable surface-orientation information
                                              mesh_export.py was throwing away)
old keep_mask (opacity>=0.10 & confidence>=0.15): 133,171/150,000 survive (88.8%) -- confirms
                                              confidence>=0.15 filters ~nothing (p1 of confidence
                                              is already 0.375)
new auto-derived thresholds (this audit's valley-finder): min_opacity~0.187 (opacity has no real
  valley -- falls back to p20), min_confidence~0.538 (the real valley) -> 106,163/150,000 survive
  filtering alone (70.8%); + floater filter + statistical outlier removal -> 80,332 points reach
  reconstruction (53.6%) -- a smaller, but much better-evidenced, point set.
```

## Ablation: what actually fixes the shape (and what doesn't)

Four variants, otherwise-identical Poisson depth=9 / 3% density trim, run
against the real checkpoint's Gaussians:

| variant | normals | filtering | components | largest-component % |
|---|---|---|---|---|
| A (reproduces shipped mesh) | naive KNN-PCA | old (no-op confidence) | 1049 | 97.8% |
| B | covariance | old (no-op confidence) | ~1050 | ~97% |
| C | naive KNN-PCA | new (auto thresholds) | ~1600 | ~96% |
| D (mesh_export.py's new defaults) | covariance | new (auto thresholds) | 1004->5 after cleanup | 99.7% |

Component-count/cleanliness improved with better filtering (C, D) and with
the post-hoc debris-removal added to `mesh_export.py`
(`min_component_vertices`). **Swapping normal source alone (B) changed
essentially nothing**, and a shaded render of variant D still shows the same
small, spiky silhouette as the original -- both were checked visually, not
just by the numbers above, specifically to avoid over-claiming a fix that
only looks good in aggregate statistics. Two more targeted experiments
(voxel-downsampling to homogenize density + lower Poisson depth; ball-
pivoting instead of Poisson) also failed to produce a recognizable street
scene -- ball-pivoting in particular did badly (2.9% of vertices in its
largest component) on this non-uniform-density input, consistent with its
own known weakness on uneven density. **The conclusion this points to: the
input point cloud itself, not the reconstruction algorithm, carries the
dominant error.** Visualizing the raw trained Gaussian positions (color =
height) directly confirms this -- a clean, regularly-banded pyramid/frustum
shape, not scene noise.

## What's fixed and validated now (mesh export stage)

- `layer1_cpu_sandbox/reconstruction/gaussian_geometry.py` (new): per-Gaussian
  normals from scale+rotation covariance; a histogram-valley threshold
  finder (`suggest_threshold_from_valley`) so `--min_opacity`/
  `--min_confidence` adapt to whatever a checkpoint's real distribution
  looks like instead of a fixed guess; a floater detector (round + unusually
  large Gaussians). `tests/test_gaussian_geometry.py` -- 7 tests, including
  one that specifically catches a wrong quaternion-order convention (the
  kind of bug that produces a plausible-looking but silently-wrong normal,
  not a crash).
- `layer1_cpu_sandbox/reconstruction/mesh_export.py`: accepts pre-computed
  normals, statistical outlier removal, connected-component debris cleanup,
  and reports every number the original request's "NUMERICAL VALIDATION"
  section asked for (`MeshExportStats` now carries components, degenerate/
  non-manifold/boundary edge counts, surface area, triangle-size
  percentiles). 100% backward compatible -- all 10 pre-existing tests plus
  7 new ones pass unmodified call sites.
- `layer3_gpu/python/export_obj.py`: wires all of the above into the CLI,
  defaults `--min_opacity`/`--min_confidence` to `auto`, and prints the full
  Gaussian-stage / filtering / mesh-stage report on every run.
- Run against the real checkpoint (`validate_real_pipeline.py`, reproduced
  by `tests/`): components 1049->5 (1004 removed as debris), largest-
  component fraction 97.8%->99.7%, non-manifold edges 962->854, boundary
  edges 6265->2622, vertex/triangle count 409,501/814,049 -> 308,779/618,637
  (smaller because the filtering is now real instead of a no-op -- see
  above).

## What's diagnosed but needs your footage to finish (depth estimation stage)

- `layer1_cpu_sandbox/reconstruction/depth_estimation.py`: `plane_sweep_depth`
  now supports `depth_spacing="log"` (new default) vs. `"linear"` (old
  behavior, kept for comparison), plus a new `check_depth_range_coverage()`
  that flags when too many resolved pixels pile up near `depth_max` (the
  range-clamping symptom). `RealPipelineConfig` now defaults to
  `n_depths=128`, `depth_spacing="log"`; `depth_min`/`depth_max` are left at
  3.0/100.0 deliberately -- the right numbers are scene-specific and were
  not re-derived against real footage here. `real_pipeline.py` now prints
  the coverage warning automatically on every run.
- **Action for you**: re-run `real_pipeline.py` (or your training entry
  point) on the real DJI footage and read the printed coverage warning. If
  it fires, raise `depth_max` (a rough starting point: flight altitude AGL +
  expected ground-plane extent) and re-run. Then retrain -- this changes
  what the Gaussians are seeded from, so it needs to go through training
  again, not just re-export.
- `tests/test_depth_estimation.py` -- 7 tests for the new spacing/coverage
  logic specifically (synthetic 2-camera scene + hand-built depth arrays;
  doesn't need real footage to validate the CODE, only to validate the
  CALIBRATION for your scene).

## Gold-standard path: TSDF fusion of rendered depth (implemented, not yet GPU-executed)

Per the original request's preference for "geometry supported by multiple
views over ... one noisy Gaussian": `layer3_gpu/python/render_multiview_depth.py`
renders depth+alpha from the TRAINED Gaussians (reusing the existing
rasterizer call, via the standard colors-precomp-as-depth trick -- no new
CUDA code), and `layer3_gpu/python/fuse_tsdf.py` fuses those into a mesh via
TSDF, which -- unlike Poisson -- only creates surface where multiple views
agree and leaves genuinely unobserved regions as holes rather than
interpolating through them. `fuse_tsdf.py` is unit-tested against a fully
synthetic analytic scene (`tests/test_fuse_tsdf.py`, 5 tests, no GPU needed)
and correctly recovers a known sphere's shape/size from synthetic multi-view
depth. `render_multiview_depth.py` needs CUDA + `diff_gaussian_rasterization`
+ real camera poses -- none available here -- so it is implemented and
carefully matched to this repo's existing rasterizer-call conventions, but
**not yet run against real data**; validate it on Environment B before
trusting it in production, the same "implemented, NOT GPU-executed" label
this project already uses elsewhere. `reconstruct_via_tsdf.py` wires both
together as a CLI; its one placeholder (`_build_cameras`) needs to be
pointed at however you already build the camera list for this checkpoint in
`train_gpu.py`'s `main()` -- left unfilled rather than guessed, since
guessing wrong here would look complete while silently using the wrong
cameras.

## Recommended, not implemented: geometric supervision during training

RGB-only loss under-constrains 3-D position (many wrong point configurations
render correctly-ish from a narrow, single-pass set of viewpoints -- the
classic shape-radiance ambiguity). Two follow-ups worth trying, in
increasing order of effort, neither implemented here because validating
either needs real training runs this environment cannot do:
1. **Bundle-adjust the DJI telemetry poses** before depth estimation, rather
   than trusting raw GPS/gimbal telemetry as-is (`real_pipeline.py`'s own
   docstring already notes poses are used as-is). Consumer GPS/gimbal error
   compounds directly into MVS depth error.
2. **A lightweight depth-consistency regularizer** in `train_gpu.py`'s loss
   (e.g. penalize Gaussian depth deviating from `depth_estimation.py`'s own
   high-consistency pixels) so photometric training actually gets pulled
   toward the plane-sweep evidence where that evidence is trustworthy,
   instead of only being weakly nudged by RGB gradients.

## Tuning guide (mesh export stage)

- **Mesh still noisy/spiky**: lower `--min_confidence` less aggressively
  (i.e. raise it) than auto suggests, or pass `--voxel_size` (try ~1-2x the
  printed p50 nearest-neighbor spacing) to homogenize density before Poisson.
- **Holes appearing where they shouldn't**: lower `--density_trim_quantile`,
  or lower whichever of `--min_opacity`/`--min_confidence` is cutting real
  surface Gaussians (check the printed per-filter removal counts).
- **Objects/thin structures disappearing**: check `--floater_filter` isn't
  excluding real thin-but-large structures (e.g. a long straight railing can
  have high scale and low anisotropy along its own axis in a bad training
  fit) -- try `--no-floater_filter` or raise `--floater_scale_percentile`.
- **Mesh too smooth / losing detail**: lower `--poisson_depth` less (i.e.
  raise it), or check `--voxel_size` isn't over-downsampling.
- **Giant triangles**: raise `--density_trim_quantile`; check for a small
  number of far-away outlier Gaussians surviving your thresholds (see the
  printed distance-from-centroid percentiles -- a max far past p99 is a tell).
- **Model too dense / slow to reconstruct**: `--voxel_size` is the main
  lever; also consider a stricter `--min_confidence`.
- **Debris / disconnected junk**: `--min_component_vertices` (start around
  the size your smallest real disconnected structure could legitimately be
  -- a small isolated tree, for instance -- so you don't delete real content)
  or `--keep_top_k_components` if you know roughly how many separate real
  structures the scene should have.

## How to re-validate any of this

```
python3 -m pytest tests/test_gaussian_geometry.py tests/test_mesh_export.py \
                   tests/test_depth_estimation.py tests/test_fuse_tsdf.py -v
```

173 pre-existing repo tests (all layers) still pass unmodified after every
change in this audit -- run the full suite (`pytest tests/ -q`) if you want
that confirmed again after pulling this in.
