# Module Status Ledger

Status vocabulary (as specified): **implemented** / **statically checked** /
**CPU-tested** / **CUDA-compiled** / **GPU-executed** / **experimentally
validated** / **benchmarked**. A module only gets a status tag if it actually
earned it in this repository — nothing below is aspirational.

## Layer 1 — CPU Sandbox (`layer1_cpu_sandbox/`)

| Module | File | Status |
|---|---|---|
| Synthetic scene generator | `synthetic/scene_generator.py` | implemented, CPU-tested, CPU-executed |
| Single-pass trajectory generator | `synthetic/trajectory_generator.py` | implemented, CPU-tested, CPU-executed |
| Camera model | `synthetic/camera_model.py` | implemented, CPU-tested, CPU-executed |
| Z-buffer renderer (ground truth) | `synthetic/renderer.py` | implemented, CPU-tested, CPU-executed |
| GT surface sampling + control measurements | `synthetic/ground_truth.py` | implemented, CPU-tested, CPU-executed |
| Degradation simulator (GPS/IMU/blur/JPEG/illum/dynamic/dropout) | `synthetic/degradation.py` | implemented, CPU-tested, CPU-executed |
| Dataset builder (orchestration) | `synthetic/dataset_builder.py` | implemented, CPU-tested, CPU-executed |
| Frame quality scoring | `perception/frame_quality.py` | implemented, CPU-tested, CPU-executed |
| GPS/IMU EKF sensor fusion | `perception/sensor_fusion.py` | implemented, CPU-tested, CPU-executed, **experimentally validated** (RMSE 7.33m→2.70m, this repo's run) |
| Dynamic object filter (residual flow) | `perception/dynamic_object_filter.py` | implemented, CPU-tested, CPU-executed, experimentally validated **and found precision-limited** (see file docstring — honestly reported, not hidden) |
| Plane-sweep MVS depth | `reconstruction/depth_estimation.py` | implemented, CPU-tested, CPU-executed, experimentally validated (consistency-filtered subset: median depth error 4.2m→1.8m) |
| Prototype point fusion | `reconstruction/prototype_point_repr.py` | implemented, CPU-tested, CPU-executed |
| **Confidence estimation (core novelty)** | `reconstruction/confidence.py` | implemented, CPU-tested, CPU-executed, **experimentally validated** (Spearman ρ≈-0.23, p<1e-67 vs geometric error) |
| Coverage/occlusion diagnostic | `reconstruction/occlusion.py` | implemented, CPU-tested, CPU-executed |
| Point-cloud eval renderer | `reconstruction/point_renderer.py` | implemented, CPU-tested, CPU-executed |
| Visual metrics (PSNR/SSIM/MS-SSIM proxy) | `evaluation/metrics_visual.py` | implemented, CPU-tested, CPU-executed |
| LPIPS + real MS-SSIM | `evaluation/metrics_visual.py::compute_lpips/compute_ms_ssim_real` | implemented with environment auto-detection: NaN/proxy fallback in Environment A (confirmed), **real values expected in Environment B once `pip install lpips pytorch-msssim` runs there — not yet confirmed with an actual Colab number** |
| Geometry metrics | `evaluation/metrics_geometry.py` | implemented, CPU-tested, CPU-executed |
| Metric/georeferencing accuracy | `evaluation/metrics_metric_accuracy.py` | implemented, CPU-tested, CPU-executed |
| Completeness metrics | `evaluation/metrics_completeness.py` | implemented, CPU-tested, CPU-executed |
| Efficiency tracker | `evaluation/metrics_efficiency.py` | implemented, CPU-tested, CPU-executed |
| Baseline configs | `evaluation/baseline_framework.py` | implemented, CPU-tested, CPU-executed |
| Ablation runner | `evaluation/ablation_framework.py` | implemented, CPU-tested, CPU-executed |
| Report generator (tables + figures) | `evaluation/report_generator.py` | implemented, CPU-tested, CPU-executed |
| Pipeline orchestrator | `pipeline.py` | implemented, CPU-tested, CPU-executed |
| Full evaluation runner | `evaluate_result.py` | implemented, CPU-tested, CPU-executed |
| Experiment CLI | `run_experiment.py` | implemented, CPU-tested, CPU-executed (see `outputs/run1/`); now also exports `reconstruction_full_method.obj` alongside the `.ply` (see mesh export row below) |
| Point cloud/Gaussian-center → mesh (.obj) export | `reconstruction/mesh_export.py` | implemented, CPU-tested, CPU-executed — 10 unit tests (synthetic point clouds, both `poisson` and `ball_pivoting`, confidence-mask filtering, colorless meshes, .obj round-trip) plus an ad hoc run against this repo's own real `outputs/run1/reconstruction_full_method.ply` (944 points → 5021 vertices/9809 triangles) |

Classical photogrammetry/COLMAP, real FastGS, and a NeRF baseline are **NOT
implemented** anywhere (CPU or GPU) — see `docs/EXPERIMENT_PLAN.md` for how to
run them in Environment B for the full baseline table.

## Real-data ingestion (`layer1_cpu_sandbox/real_data/`)

| Module | File | Status |
|---|---|---|
| DJI flight-record CSV parser (2 formats) | `dji_log_parser.py` | implemented, CPU-tested, CPU-executed against synthetic mock CSVs matching both real column schemas — **not yet run against an actual downloaded DJI file** |
| Lat/lon → local ENU conversion | `geo_utils.py` | implemented, CPU-tested, CPU-executed (cross-checked against an independent haversine formula, <1% disagreement) |
| Video frame extractor | `video_loader.py` | implemented, CPU-tested, CPU-executed against a real OpenCV-written/read `.mp4` |
| Real-dataset assembler (log↔video matching, GPS→pose) | `real_dataset_builder.py` | implemented, CPU-tested, CPU-executed end-to-end on mock video+log |
| Real-data pipeline adapter | `real_pipeline.py` | implemented, CPU-tested, CPU-executed end-to-end on mock data |
| Real-data experiment CLI | `run_real_experiment.py` | implemented, CPU-tested, **CPU-executed as the actual command-line entrypoint** (not just its internal functions) on mock video+log, produced real `.ply`/`.png`/`summary.json` output; now also attempts `reconstruction_real.obj` (see Layer 1's mesh export row) — wrapped in try/except since real single-pass footage is more likely to be too sparse for a mesh than the synthetic path |

**Not yet run against a real downloaded file anywhere** — the schema was
confirmed against the public DroneVideoMeasure project's own parsing code,
not guessed, but only an actual run against a real DJI export (e.g. the
Zenodo 3604005 files) counts as full validation. Camera intrinsics are
assumed (84° HFOV, DJI Phantom 4 Pro spec) unless calibrated — no
calibration routine exists yet in this repo.

## Layer 2 — Interfaces (`layer2_interfaces/`)

All 8 interface files: **implemented** (Python ABCs), **statically checked**
(importable, no CPU/GPU execution — they're contracts, not logic). Each file's
docstring names its CPU prototype, GPU replacement, pretrained/custom model
candidates, and a research-novelty estimate (LOW/MODERATE/HIGH).

## Layer 3 — GPU Research Implementation (`layer3_gpu/`)

| Module | File | Status |
|---|---|---|
| Confidence-aware Gaussian model (PyTorch, simplified 2D CPU-testable) | `python/confidence_gaussian_model.py` | implemented, **CPU-tested, CPU-executed** (real Adam optimization, loss 0.047→0.00001; single-observation-cap gating verified with hand-built stats) |
| Shared confidence formula (`blended_confidence`, `gate_densify_and_prune`) | `python/confidence_gaussian_model.py` | implemented, CPU-tested, CPU-executed — extracted so the production model (below) and the CPU-testable model share one formula instead of duplicating it |
| Temporal+spatial confidence propagation (STCP-inspired) | `python/confidence_propagation.py::ObservationConfidencePropagator` | implemented, CPU-tested, CPU-executed (verified: spatial diffusion measurably pulls up an under-evidenced neighbor's confidence; buffer resize correctly handles prune+split) |
| Optimizer-struggle signal (OSAD-inspired) | `python/confidence_propagation.py::optimizer_state_struggle_signal` | implemented, CPU-tested, CPU-executed (verified: distinguishes sustained directional gradient push from oscillation using real Adam state) |
| Three-signal densify/prune gate | `confidence_gaussian_model.py` + `gaussian_model.py` | implemented, CPU-tested, CPU-executed — **verified the key architectural guarantee**: a massive optimizer-struggle signal in a poorly-observed region does NOT bypass the observation-confidence gate |
| Production 3D Gaussian model (position/scale/rotation/opacity/SH, init/optimizer/prune/clone/split) | `python/gaussian_model.py` | implemented, **CPU-tested, CPU-executed** for everything that doesn't need the external rasterizer (init, optimizer-state correctness through structural changes, clone/split arithmetic, confidence inheritance) — rendering itself untested (needs CUDA + `diff-gaussian-rasterization`) |
| FWL-inspired frequency-weighted loss + real MS-SSIM loss | `python/losses.py` | implemented, CPU-tested, CPU-executed (gradient flow confirmed; MS-SSIM falls back to a hand-computed single-scale version when `pytorch-msssim` isn't installed, same discipline as `metrics_visual.py`) |
| GPU training loop | `python/train_gpu.py` | implemented, **CUDA-executed, multiple full real-footage runs** (up to 20,000 iterations, 829,355 Gaussians on a T4). **Real finding, not yet fully resolved**: held-out SSIM peaked at iter 6500 (246K Gaussians) then declined through iter 20000 (829K Gaussians) — overfitting once Gaussian count outgrew what 77 training cameras could constrain. Separately, the two worst-performing held-out views were traced to the chronological first/last extracted frames of the flight — the structurally weakest-covered points of a single-pass trajectory (see `docs/EXPERIMENT_PLAN.md` robustness notes) — not a rendering bug (a real bug would not correlate this cleanly with flight-path position). `densify_until_iter` retuned 15000→7000 based on this data; best-checkpoint tracking (by held-out SSIM) and per-view metric breakdown added so this pattern is visible in the console log directly, not just discoverable by opening every PNG. Optional `--use_agents` hooks (below) layer on top of this unchanged loop and do not alter it when omitted. |
| Final-mesh export hook (opacity+confidence gated, auto-runs at end of `train()`) | `python/train_gpu.py::_export_mesh` | implemented, **NOT GPU-executed** (no CUDA in this environment to finish a real training run against) — calls the CPU-tested `mesh_export.py` (above) on plain numpy arrays, wrapped in try/except so a meshing failure cannot discard a finished run's other artifacts. Confidence/opacity extraction lines mirror `_save_checkpoint`'s already-GPU-executed color formula exactly (factored into a shared `_gaussian_positions_and_colors` helper) — the new code is the `keep_mask`/export call, not a new way of reading the model. |
| Stand-alone re-export from any saved checkpoint | `python/export_obj.py` | implemented, NOT GPU-executed, NOT yet run against a real `full_state_iter*.pt` (none exists in this environment) — exercised only by reading against `load_full_state`'s (below) now-CPU-tested contract |
| `_save_full_state`/`load_full_state` now also persist `confidence_stats` | `python/train_gpu.py` | implemented; new regression test added (`tests/test_layer3_gaussian_model.py::test_full_state_round_trip_preserves_accumulated_confidence_not_just_prior`) but **not yet executed in this session** — torch isn't installable in this sandbox (see below); run it with `pytest tests/test_layer3_gaussian_model.py -v` once torch is available, exactly as `docs/GETTING_STARTED.md` Part D already instructs. Backward-compatible: checkpoints saved before this fix still load (falling back to the old prior-only behavior) rather than raising. |
| Simplified differentiable splatter | `python/confidence_gaussian_model.py::simplified_splat_cpu` | implemented, CPU-tested, CPU-executed. Explicitly NOT the production renderer. |
| CUDA kernel (confidence accumulate/compute/gate, now including the OSAD-inspired struggle-signal gate) | `csrc/confidence_gaussian/confidence_gaussian_kernel.cu` | implemented, **CUDA-compiled — confirmed** (built successfully via `setup.py build_ext --inplace` on a real Colab T4, CUDA 13.0, Python 3.13, in a user session; extended since with the struggle-signal gate, not yet recompiled) |
| C++/pybind11 bindings | `csrc/confidence_gaussian/confidence_gaussian.cpp` | implemented, statically reviewed — extended for the struggle-signal gate, not yet recompiled |
| Build (setup.py, CMakeLists.txt) | `layer3_gpu/setup.py`, `layer3_gpu/CMakeLists.txt` | implemented, previously confirmed to build; unchanged by this round's edits |
| Colab smoke test | `python/colab_smoke_test.py` | implemented; import-path bug from the first Colab run is fixed; **extended with a new check (struggle signal cannot bypass the observation gate) that mirrors the CPU test exactly — not yet run against the recompiled kernel** |
| Real-data-to-GPU bridge (`build_cameras_from_synthetic`/`_from_real`) | `python/train_gpu.py` | implemented, NOT executed — depends on the untested render call |
| Surface completion | *(interface only, see `layer2_interfaces/i_occlusion_completion.py`)* | **NOT implemented anywhere**, deliberately deferred (see `docs/STOP_CONDITIONS.md`) |

**Four real bugs were caught and fixed while building the above, by actually
running the code on CPU rather than only reading it** — recorded here rather
than swept away, since catching them is exactly what the CPU-testable
discipline is for: (1) the optimizer's `param_groups` not being re-pointed
at new `Parameter` objects after prune/densify, meaning Adam would have
silently kept optimizing a stale, disconnected tensor after the first
structural change in real training; (2) a lambda signature mismatch in a
padding helper; (3) confidence inheritance for clones being documented but
not actually wired through (`ObservationConfidencePropagator.resize()`
wasn't accepting an override value); (4) `GaussianModel` initially had no
`confidence()`/`densify_and_prune()` of its own — `train_gpu.py` would have
crashed on the first densification round. All four are now guarded by
regression tests (`tests/test_layer3_gaussian_model.py`).

**Two more real bugs were caught while adding .obj mesh export** (same
discipline, same reason it's worth recording): (5) Open3D's Poisson
reconstructor allocates a `vertex_colors` buffer defaulting to solid black
when the source point cloud has none, so `has_vertex_colors()` reports
`True` and a naive `write_triangle_mesh(..., write_vertex_colors=True)`
would silently write a mesh that renders solid black instead of a viewer's
default shading — caught by actually running `points_to_mesh` on a
colorless point cloud in `tests/test_mesh_export.py`, not by reading the
Open3D docs; now explicitly cleared when the source had no colors. (6)
`_save_full_state` saved every Gaussian parameter needed to re-render a
model, but not `confidence_stats`' raw accumulators — so
`load_full_state(...).confidence()` silently fell back to each Gaussian's
creation-time `prior_confidence` instead of what training actually
accumulated (see `blended_confidence`'s `has_obs = stats.n_accum > 0`
gate), invisible to rendering/PSNR/SSIM (neither touches confidence) and
to the in-process `_export_mesh` call at the end of `train()` (the live
model's stats were never reset) — only surfaced because confidence-gated
export from a *reloaded* checkpoint (`export_obj.py`) specifically needed
it. Fixed; a CPU-runnable regression test exists but has not been executed
in this session (see the Layer 3 table row above) — this one is reasoned
through the code rather than confirmed by actually running it, flagged as
such rather than claimed otherwise.

**No performance, accuracy, or benchmark number for actual GPU-trained
reconstruction exists in this repository.** The CUDA extension build and
its numeric smoke-test check are the only things confirmed to have run on a
real GPU so far. The .obj mesh export path is CPU-tested end-to-end
(`mesh_export.py` itself, plus both CPU-sandbox entry points actually
producing a `.obj` file) but has never run against real GPU-trained
Gaussians — `_export_mesh`/`export_obj.py` are implemented and reasoned
through carefully against the already-proven `_save_checkpoint` pattern,
not yet exercised against one.

## Layer 4 — Agents (`layer4_agents/`)

Optional. Nothing in Layers 1-3 above requires this layer, and every flag it
adds (`--use_agents`, `--use_llm`, `--agent_model`) defaults OFF — omitting
them runs exactly the Layer 1/3 code paths already documented above,
SRT-log parsing, mesh export, and all.

| Module | File | Status |
|---|---|---|
| LLM-output validation/clamping (the safety boundary) | `layer4_agents/schemas.py` | implemented, CPU-tested (32/32, including malformed JSON, out-of-bounds values both directions, injected/hallucinated fields, inverted ranges, an upstream-API-failure sentinel, and the Training Control Agent's decision type) |
| Agent context builders | `layer4_agents/context_builders.py` | implemented, CPU-tested (11/11), integration-checked against real `PipelineResult`/`FullEvaluation`/`RealEvaluation` objects |
| Agent interfaces (contracts) | `layer4_agents/interfaces.py` | implemented, statically checked (same status Layer 2's interfaces carry, for the same reason — contracts, not logic) |
| Mock agents (deterministic, CPU-testable stand-ins) | `layer4_agents/mock_agents.py` | implemented, CPU-tested, CPU-executed (25/25) |
| Prompts + Anthropic tool schemas | `layer4_agents/prompts.py` | implemented, CPU-tested (7/7, shape/bounds consistency with `schemas.py`) |
| Real Claude-backed agents | `layer4_agents/llm_agents.py` | implemented, glue logic CPU-tested with stub SDK objects (14/14) — **the real API path is NOT executed anywhere in this repository**; this sandbox has no network (mirrors Layer 3's GPU-required caveat, one level narrower: network + `ANTHROPIC_API_KEY` required, neither available here) |
| Orchestrator (bounded retry loop, Layer 1) | `layer4_agents/orchestrator.py` | implemented, CPU-tested, CPU-executed end-to-end (8/8) against a real small synthetic dataset — retry bound, config mutation between iterations, exception-revert, and gate-threshold audit trail all verified against real behavior, not mocked assertions |
| Agentic experiment CLI (synthetic) | `layer4_agents/run_agentic_experiment.py` | implemented, CPU-executable command-line entrypoint (mock-agent path); see `layer4_agents/outputs/agentic_run1/` for a real recorded run |
| Agentic experiment CLI (real data) | `layer4_agents/run_agentic_real_experiment.py` | implemented, CPU-executable command-line entrypoint (mock-agent path) — currently builds its dataset via `build_real_dataset()` without the `--assumed_gimbal_pitch_deg`/`--assumed_gimbal_roll_deg` passthrough `run_real_experiment.py` and `train_gpu.py --source real` have; fine for CSV-format logs, raises `run_real_experiment.py`'s own clear error if pointed at a `.srt` log without those set some other way |
| Training Control Agent decision logic (Layer 3, watches held-out PSNR/SSIM/LPIPS, may freeze densification or stop training early) | `layer4_agents/{schemas,context_builders,mock_agents,llm_agents,prompts}.py` (`TrainingControlDecision`/`MockTrainingControlAgent`/`LLMTrainingControlAgent`) | implemented, CPU-tested — `MockTrainingControlAgent` calibrated against and verified to catch this repo's own documented iter-6500-peak-then-decline finding by iteration 8000 (vs. the 27-stale-checkpoint/iteration-20000 point a human actually caught it at) |
| `train_gpu.py` agent-hook integration (`apply_training_agent_at_checkpoint`, `_build_layer4_agents`, `_frame_agent_decision_kwargs`, `TrainConfig.use_agents`, `--use_agents`/`--use_llm`/`--agent_model` CLI flags) | `layer3_gpu/python/train_gpu.py` | implemented; the extracted decision-application function is **CPU-tested against the real file** (7/7, via scoped, self-reversing dependency stubbing — a real `torch` etc. is always preferred and left untouched when present) — see `tests/test_layer3_train_gpu_agent_hooks.py`. Layered on top of this file's already-GPU-executed SRT-log-parsing/mesh-export/quality-knob code without touching any of it: the frame-selection and densify/early-stop hooks only ever choose VALUES for the same `dense_seed_kwargs`/`TrainConfig` fields those features already populate. The full loop under actual CUDA execution with `--use_agents` on has **not been run on a GPU**; every other flag's behavior is unchanged when `--use_agents` is omitted (verified by static review + syntax check + full CPU test suite, not GPU execution) |
| Frame-overlap diagnostics (raw consecutive-frame match ratio, feeds the Frame Agent's context) | `layer1_cpu_sandbox/perception/frame_overlap.py` | implemented, CPU-tested, CPU-executed (8/8) |
| Spatial confidence-grid diagnostics (feeds the Reconstruction Agent's "weak regions" context) | `layer1_cpu_sandbox/evaluation/spatial_diagnostics.py` | implemented, CPU-tested, CPU-executed (8/8) |
| `PipelineConfig`/`RealPipelineConfig.min_overall` (frame-selection quality floor, promoted from `select_frames()`'s hardcoded default to a real, agent-settable knob) | `layer1_cpu_sandbox/pipeline.py`, `layer1_cpu_sandbox/real_data/real_pipeline.py` | implemented, CPU-tested, CPU-executed — default (0.15) unchanged from the prior hardcoded behavior, so this is additive, not a behavior change, when Layer 4 isn't used |

**Integration note (this file specifically merges two previously-diverged
branches):** one branch had Layers 1-3 fully caught up with real DJI `.srt`
telemetry parsing (`real_data/dji_log_parser.py`'s `parse_dji_srt_log` /
`parse_dji_srt_frametelemetry_log`), confidence/opacity-gated `.obj` mesh
export (`reconstruction/mesh_export.py`, `train_gpu.py::_export_mesh`,
`export_obj.py`), and several quality/CLI refinements (`--split_scale_thresh`
exposed, `holdout_fraction` defaulting to 0 for scarce-camera production
runs) but no Layer 4 at all; the other had Layer 4 fully implemented and
wired into `train_gpu.py` (`--use_agents`) but was missing all of the above.
This merge takes the SRT/mesh/quality branch as the base and re-applies the
Layer 4 addition on top of it, adding `min_overall` to both pipeline configs
along the way (required by `orchestrator.py`'s `dataclasses.replace(...,
min_overall=...)`). The full test suite (152 CPU-runnable tests, 2 more
skipped only because `torch` isn't installed in this sandbox) passes against
the merged tree, including both `tests/test_mesh_export.py`/
`tests/test_real_data.py` (the SRT/mesh branch's tests) and
`tests/test_layer3_train_gpu_agent_hooks.py`/`tests/test_layer4_*.py` (the
agents branch's tests) together.
