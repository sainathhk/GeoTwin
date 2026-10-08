# GeoTwin

**Single-pass drone video → georeferenced, confidence-aware 3D twin.**

SIH 2026 · PS SIH26158, *Single-Pass Drone Video to Accurate 3D Model Generation System* · Theme: Robotics & Drones · Team RECURA (ID 192658)

One drone video and the flight log the drone already writes (the SRT file with GPS and altitude) go in. A textured mesh (OBJ/GLB), landmark tags (GeoJSON) and a coverage map that marks what the camera never saw come out. No ground control points, no second flight.

**Core novelty: Observation-Aware Reconstruction Confidence.** Every reconstructed
primitive carries a confidence score built from five independently measurable
evidence signals (multi-view redundancy, viewing-angle baseline, depth consistency,
frame quality, pose confidence), and that confidence *actively controls* the
representation (prune / densify / flag as uncertain) rather than being reported after
the fact. Three more ideas sit around it: the cameras are locked to the drone's GPS
track before training (true metres first), gaps are shown as gaps (inferred areas stay
separate from measured geometry), and the agents only supervise. They never draw geometry.

## Results on the real clip

One pass, 22 frames, about 14 s of video (DJI0004). Each row has a source file and a plain
"what it does not show" note in the evidence folder (see [Evidence](#evidence)).

| What | Result |
|---|---|
| Mesh | 1,488,206 vertices, 2,980,068 triangles (an independent viewer counts 1,488,208 / 2,980,073) |
| Confidence gate | kept 665,792 of 2,439,699 fused points (27.3%) |
| Clean-up | 5,535 loose fragments cut to 11 parts; the largest holds 98.0% of the vertices; non-manifold edges 0.09% |
| Camera-to-GPS fit | median 1.17 m, mean 1.55 m, max 4.42 m. Approximate, not survey-grade |
| Render vs real frame | mean PSNR 25.92 dB, SSIM 0.832, LPIPS 0.148 (our training run's evaluation; the evidence figure shows one training view) |
| Run time | Poisson meshing 161.4 s; about 5-6 min end to end on one Colab T4 (our recorded run). Faster on larger GPUs is expected, not benchmarked |
| Tests | 221 tests; 176 run and pass on a CPU-only machine, the other 45 (4 files) need Open3D or PyTorch |

On the synthetic sandbox (Layer 1, ground truth known), confidence is a statistically
significant but modest predictor of geometric error (Spearman ρ ≈ -0.23, p < 1e-67,
n = 5,799; Pearson r = -0.164), and GPS + IMU fusion cut position RMSE from 7.33 m to
2.70 m. These are logic checks, not claims about real-footage quality; see
`layer1_cpu_sandbox/outputs/run1/`.

## What we have not shown yet

- **Absolute accuracy.** Nothing here is checked against surveyed points or measured
  distances on the real clip. The mesh report's distances compare the mesh with its own
  input points (in-sample), so they say how faithful the meshing is, not how right the
  model is.
- **Georeferencing is approximate.** Cameras are fitted to the drone's own GPS log, so the
  fit is not an independent check. North and heading are not independently verified, and
  the fusion step assumes an 84° field of view rather than a calibrated one.
- **Held-out image quality.** Held-out SSIM peaked at iteration 6,500 (246 K Gaussians) and
  then declined as the Gaussian count outgrew what the 77 training cameras could constrain.
  Densification now stops at iteration 7,000 and the best held-out checkpoint is tracked
  (`best_checkpoint.json`). The means in the table above are as reported by our training
  run's evaluation; we do not present them as held-out numbers.
- **No baseline comparison yet** (COLMAP/MVS or vanilla 3DGS on the same frames).
- **Coverage percentage** on the real clip and **larger-GPU timing** are not measured.
- **Real Claude-backed agents** are implemented but have never been executed in this
  repository. Every agent run so far uses the deterministic mock agents.

## How it works

1. **Clean and lock** (`layer1_cpu_sandbox/`): keep sharp frames, drop moving cars and
   people, one lens setting per batch, fuse GPS (and IMU data when the drone provides it)
   into one smooth track.
2. **One forward pass** (VGGT): camera poses, depth and 3D points for all frames at once.
   The cameras are then snapped onto the GPS track so the scale is in metres.
3. **Confidence tag**: each 3D point is scored from the five signals, and the gate keeps
   the confident ones.
4. **Training loop** (`layer3_gpu/`): confidence-aware 3D Gaussian Splatting. Render,
   compare with the real frame, update, repeat. Growth is gated by evidence ("no proof,
   no growth").
5. **Supervisor loop** (`layer4_agents/`): agents pick frames, set limits and retry a
   bounded number of times. They never create geometry.
6. **Extraction**: Poisson meshing at depth 12, lowest 1% density trimmed, parts under
   500 vertices dropped. Output is OBJ/GLB, plus GeoJSON tags and the coverage map from
   the viewer.

`layer2_interfaces/` holds the contracts that make each module swappable (CPU twin vs
GPU/VGGT implementation). The architecture diagram (titled VANTAGE, the same system) is
in the evidence folder.

## Layers (do not collapse these into one)

| Layer | What it is | Status |
|---|---|---|
| **Layer 1** — `layer1_cpu_sandbox/` | CPU/NumPy/OpenCV digital twin of the full pipeline, plus real-data ingestion (video + SRT). A **scientific testbed**, not the final method. | Implemented, CPU-tested, **CPU-executed** on synthetic ground truth (real numbers in `outputs/run1/`) |
| **Layer 2** — `layer2_interfaces/` | Abstract interface contracts so CPU and GPU implementations are swappable per module. | Implemented (interfaces only) |
| **Layer 3** — `layer3_gpu/` | Confidence-aware 3D Gaussian Splatting, PyTorch training loop, CUDA confidence kernel. | Implemented. The training loop is **CUDA-executed on real footage** (several full runs on a Colab T4, up to 20,000 iterations). The confidence CUDA kernel compiled on a Colab T4; its later struggle-signal extension has not been recompiled yet. Module by module: `STATUS.md` |
| **Real-clip mesh path** — VGGT poses/depth, confidence fusion, Poisson (depth 12, trim 1%, clean 500) | Wired into `train_gpu.py` for real-data runs; see `docs/RUN_GAUSSIAN_THEN_VGGT.md` | **Executed on a Colab T4 on the real clip** (results above) |
| **Layer 4** — `layer4_agents/` | AI agents that decide/retry/diagnose around the pipeline, never generating geometry themselves: Frame Agent, Reconstruction Agent, Evaluation Agent, Training Control Agent, and bounded retry orchestrators (Layer 1 CPU pipeline and, via `train_gpu.py --use_agents`, Layer 3 GPU training). Fully optional: every flag defaults off. | Implemented, CPU-tested, **CPU-executed** end-to-end with mock agents (real numbers in `layer4_agents/outputs/`). The real Claude-backed path is implemented but not executed anywhere in this repo. `train_gpu.py`'s agent hooks are CPU-tested against the real file but not yet run on an actual GPU |

See `STATUS.md` for the exact implemented / CPU-tested / CUDA-compiled / GPU-executed /
benchmarked status of every module, and `docs/ARCHITECTURE.md` for the design rationale
(why 3DGS, why we don't reimplement the rasterizer, why the CPU sandbox is a testbed and
not a downgrade).

## Quickstart — Environment A (this repo, CPU only)

```bash
pip install -r requirements.txt
python3 -m layer1_cpu_sandbox.run_experiment --out_dir layer1_cpu_sandbox/outputs/run1 --seed 0
python3 -m pytest tests/ -v
python3 layer3_gpu/python/confidence_gaussian_model.py   # CPU-testable Layer-3 correctness check
```

Without Open3D or PyTorch, 176 of the 221 tests run and pass. The other 45 are skipped,
not failed: `test_fuse_tsdf.py` and `test_mesh_export.py` need Open3D, and
`test_layer3_gaussian_model.py` and `test_layer3_confidence_gaussian_cpu.py` need
PyTorch. Install those packages to run everything.

The experiment above also writes `reconstruction_full_method.obj` next to the `.ply`: a
triangle mesh reconstructed from the same confidence-fused points, viewable in
Blender/MeshLab/CloudCompare without any project-specific tooling. See
`layer1_cpu_sandbox/reconstruction/mesh_export.py` for how, and `docs/ARCHITECTURE.md`
for why this is a deliverable-format export and not a change to the representation that
is actually trained (3DGS).

## Layer 4 (agents) — mock agents, no API key needed

```bash
python3 -m layer4_agents.run_agentic_experiment --out_dir layer4_agents/outputs/agentic_run1 --seed 0
python3 -m layer4_agents.run_agentic_real_experiment --video flight.mov --log flight.csv \
    --out_dir layer4_agents/outputs/agentic_real_run1
```

The same optional layer works on the GPU training path (Environment B, below) via
`--use_agents` on `train_gpu.py`. Everything else about that command
(`--source real/synthetic`, SRT/gimbal flags, mesh export, quality knobs) behaves
identically whether or not `--use_agents` is passed.

### Real Claude-backed agents (optional)

Requires network + `ANTHROPIC_API_KEY` (not available in the sandbox this was built in,
so CPU-tested with stub SDK objects only, see `STATUS.md`):

```bash
pip install -r requirements-agents.txt
python3 -m layer4_agents.run_agentic_experiment --out_dir layer4_agents/outputs/agentic_run1 --use_llm
# Or on the real GPU training path (same --use_agents flag as the mock path, add --use_llm):
python3 layer3_gpu/python/train_gpu.py --source real --video flight.mov --log flight.csv \
    --out_dir outputs/gpu_run1 --use_agents --use_llm
```

See `docs/AGENTIC_ARCHITECTURE.md` for the agent-loop design and
`layer4_agents/llm_agents.py` for exactly how the Claude calls are made.

## Quickstart — Environment B (Colab GPU, Layer 3)

```bash
git clone <this repo> && cd sih2026-3d-recon
bash colab_bootstrap.sh
```

See `docs/SIH_IMPLEMENTATION_PLAN.md` for the training run, the full baseline comparison
and the robustness experiments (the baseline comparison has not been done yet).
`train_gpu.py` also exports a confidence/opacity-gated `mesh_iter<N>.obj` once training
finishes; `export_obj.py` can (re-)export a mesh from any saved `full_state_iter*.pt`
checkpoint afterwards, without retraining, for example from the best held-out-SSIM
iteration rather than the last one (`train_gpu.py` prints both and
`best_checkpoint.json` records them). Add `--use_agents` (optionally with `--use_llm`) to
let the Frame Agent and Training Control Agent choose frame-selection and
densify/early-stop decisions during the same run.

## Real-video workflow (Colab)

For real-data runs, `train_gpu.py` automatically performs the validated VGGT
confidence-fusion and Poisson Trim001/Clean500 export after Gaussian training ends. The
OBJ and GLB are written beneath the same run output directory in
`vggt_trim001_clean500/`. See
[docs/RUN_GAUSSIAN_THEN_VGGT.md](docs/RUN_GAUSSIAN_THEN_VGGT.md) for Colab installation,
outputs and rerun instructions.

For the one-click version, run `notebooks/GeoTwin_Colab_Run_and_View.ipynb` after
`colab_bootstrap.sh`. It runs Gaussian training and the VGGT Trim001/Clean500 mesh export,
then offers an **Open GeoTwin viewer** button. The viewer embeds directly in Colab and
loads the run GLB plus the matching SRT without a manual server or a download and
re-upload.

Mesh exports also include an in-sample point-to-surface consistency and topology report.
Its internal screening targets are labelled as project-only and are not accuracy claims.

## GeoTwin viewer (`web/`)

`web/geotwin_mvp.html` loads a local GLB or OBJ and, optionally, the matching DJI SRT/CSV.
On Windows, double-click `Open_GeoTwin_MVP.bat` (it runs `python web\start_geotwin_mvp.py`
and opens the page). You can place landmark pins on visible surfaces (each shows its
latitude and longitude), mark gaps, and export `geotwin_annotations.geojson` (WGS84
points and polygons, suitable for GIS inspection; vertical values are relative
`model_z_m`, not absolute elevation).

An amber patch is a view-only overlay for a hole or unseen region. It is explicitly
uncertain, stays a separate layer, and is never counted as measured geometry. It is not a
watertight mesh repair: a single oblique flight cannot establish what is behind
occluders. The GLB does not retain the local ENU-to-WGS84 origin by itself, so ship it
with the GeoJSON and `web/geotwin_origin.json`. Details and limits:
`docs/GEOTWIN_MVP.md`.

## Repository map

```
layer1_cpu_sandbox/     CPU digital twin: synthetic data, perception, reconstruction, evaluation, real-data ingestion
layer2_interfaces/      Python ABCs: CPU prototype <-> GPU replacement contracts, module by module
layer3_gpu/             CUDA kernels + C++ bindings + PyTorch training loop (confidence-aware 3DGS)
layer4_agents/          Agents that decide/retry/diagnose around the pipeline: Frame, Reconstruction,
                        Evaluation and Training Control agents, plus bounded retry orchestrators.
                        Fully optional: every entrypoint above works identically without this layer.
web/                    GeoTwin viewer: GLB/OBJ + SRT, landmark pins, GeoJSON export, inferred-area overlay
tests/                  pytest suite: 221 tests (176 run on CPU alone; 45 need Open3D or PyTorch)
docs/                   architecture rationale (Layers 1-3 and, separately, Layer 4's agents),
                        prototype-vs-final mapping, novelty candidates, missing components,
                        SIH plan, experiment plan, stop conditions, viewer and Colab run guides
STATUS.md               module-by-module ledger of what is implemented, tested, compiled, executed, benchmarked
```

## Evidence

Every number in the results table has a source file and a note on what it does not show.
The evidence folder (the *Proof documents* link on slide 6 of the idea PDF) opens with
`00_START_HERE.pdf`, then `01_Evidence_Pack.pdf`. The raw reports sit next to them,
unedited. The mesh report is labelled in-sample: it compares the mesh with its own input
points and is not an accuracy measurement.

## Honesty guarantees this repo tries to uphold

- Every number in `layer1_cpu_sandbox/outputs/run1/summary.json` was computed by running
  the code in this repo, on synthetic ground truth. It is not fabricated and not copied
  from a paper.
- Nothing here claims GPU execution that didn't happen. `STATUS.md` is blunt about what
  is still just source code.
- PSNR/SSIM are never used as a stand-in for geometric or metric accuracy; see the
  visual/geometry/metric-accuracy/completeness split enforced throughout
  `layer1_cpu_sandbox/evaluation/`.
- In-sample distances (mesh vs its own input points) are never presented as accuracy, and
  the georeferencing is labelled approximate, not survey-grade.
- Layer 4's agents never invent numbers: every metric an Evaluation Agent cites comes from
  the unmodified evaluation code, passed through unchanged (`layer4_agents/schemas.py`
  structurally discards any numbers a model tries to supply itself). Nothing claims a real
  Claude API call happened anywhere in this repo's own test runs.

## Built on

VGGT (Wang et al., CVPR 2025), 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023),
Screened Poisson Surface Reconstruction (Kazhdan & Hoppe, 2013), COLMAP (Schönberger &
Frahm, CVPR 2016), FastGS (Ren et al., CVPR 2026), gsplat (Ye et al., JMLR 2025) and SuGaR
(Guédon & Lepetit, CVPR 2024). FastGS-QADS, our follow-up to FastGS, is a manuscript
submitted for publication.
