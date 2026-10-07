# Single-Pass Drone Video to 3D Model Generation — SIH 2026 PS 26158

**Core novelty: Observation-Aware Reconstruction Confidence** — every reconstructed
primitive carries a confidence score built from five independently-measurable
evidence signals (multi-view redundancy, viewing-angle baseline, depth-consistency,
frame quality, pose confidence), and that confidence *actively controls* the
representation (prune / densify / flag-uncertain) rather than being reported after
the fact. Validated end-to-end on synthetic ground truth: confidence is a
statistically significant predictor of geometric error (Spearman ρ≈-0.23,
p<1e-67, n=5799 — see `layer1_cpu_sandbox/outputs/run1/`).

## Three-layer architecture (do not collapse these into one)

| Layer | What it is | Status |
|---|---|---|
| **Layer 1** — `layer1_cpu_sandbox/` | CPU/NumPy/OpenCV digital twin of the full pipeline. A **scientific testbed**, not the final method. | Implemented, CPU-tested, **CPU-executed** (real numbers in `outputs/run1/`) |
| **Layer 2** — `layer2_interfaces/` | Abstract interface contracts so CPU and GPU implementations are swappable per-module. | Implemented (interfaces only) |
| **Layer 3** — `layer3_gpu/` | The real GPU architecture: confidence-aware 3D Gaussian Splatting, CUDA kernels, PyTorch training loop. | Implemented (real code); **CPU-tested** where torch-on-CPU makes that possible; CUDA kernels **NOT compiled/executed anywhere yet** — Environment B (Colab GPU) required |
| **Layer 4** — `layer4_agents/` | AI agents that decide/retry/diagnose around the pipeline above — never generate geometry themselves. Frame Agent, Reconstruction Agent, Evaluation Agent, Training Control Agent, bounded retry orchestrators (Layer 1 CPU pipeline and, via `train_gpu.py --use_agents`, Layer 3 GPU training). Fully optional — every flag defaults off. | Implemented, CPU-tested, **CPU-executed** end-to-end with mock agents (real numbers in `layer4_agents/outputs/`); real Claude-backed agent path implemented but not executed anywhere in this repo (no network in the environment it was built in); `train_gpu.py`'s agent hooks CPU-tested against the real file but not yet run on an actual GPU |

See `STATUS.md` for the exact implemented / CPU-tested / CUDA-compiled / GPU-executed /
benchmarked status of every single module, and `docs/ARCHITECTURE.md` for the full
design rationale (why 3DGS, why we don't reimplement the rasterizer, why the CPU
sandbox is a testbed and not a downgrade).

## Quickstart — Environment A (this repo, CPU only)

```bash
pip install -r requirements.txt
python3 -m layer1_cpu_sandbox.run_experiment --out_dir layer1_cpu_sandbox/outputs/run1 --seed 0
python3 -m pytest tests/ -v
python3 layer3_gpu/python/confidence_gaussian_model.py   # CPU-testable Layer-3 correctness check
```

The experiment above now also writes `reconstruction_full_method.obj` next to the
existing `.ply` — a triangle mesh reconstructed from the same confidence-fused
points, viewable in Blender/MeshLab/CloudCompare/etc. without any project-specific
tooling. See `layer1_cpu_sandbox/reconstruction/mesh_export.py` for how, and
`docs/ARCHITECTURE.md` for why this is a deliverable-format export and not a
change to the representation that's actually trained (3DGS).

## Layer 4 (agents) -- mock agents, no API key needed:

```bash
python3 -m layer4_agents.run_agentic_experiment --out_dir layer4_agents/outputs/agentic_run1 --seed 0
python3 -m layer4_agents.run_agentic_real_experiment --video flight.mov --log flight.csv \
    --out_dir layer4_agents/outputs/agentic_real_run1
```

Same optional layer on the GPU training path (Environment B, below) via
`--use_agents` on `train_gpu.py` -- everything else about that command
(`--source real/synthetic`, SRT/gimbal flags, mesh export, quality knobs)
behaves identically whether or not `--use_agents` is passed.

## Real Claude-backed agents (Layer 4, optional)

Requires network + `ANTHROPIC_API_KEY` (not available in the sandbox this
was built in -- CPU-tested with stub SDK objects only, see `STATUS.md`):

```bash
pip install -r requirements-agents.txt
python3 -m layer4_agents.run_agentic_experiment --out_dir layer4_agents/outputs/agentic_run1 --use_llm
# Or on the real GPU training path -- same --use_agents flag as the mock path, add --use_llm:
python3 layer3_gpu/python/train_gpu.py --source real --video flight.mov --log flight.csv \
    --out_dir outputs/gpu_run1 --use_agents --use_llm
```

See `docs/AGENTIC_ARCHITECTURE.md` for the full agent-loop design and
`layer4_agents/llm_agents.py` for exactly how the Claude calls are made.

## Quickstart — Environment B (Colab GPU, Layer 3)

```bash
git clone <this repo> && cd sih2026-3d-recon
bash colab_bootstrap.sh
```

See `docs/SIH_IMPLEMENTATION_PLAN.md` for what to run *after* the bootstrap
(the actual Gaussian-optimization training loop, full baseline comparison,
robustness experiments). `train_gpu.py` now also exports a confidence/opacity-gated
`mesh_iter<N>.obj` automatically once training finishes; `export_obj.py` can
(re-)export a mesh from any saved `full_state_iter*.pt` checkpoint afterwards,
without retraining, e.g. from the best held-out-SSIM iteration rather than the
last one (`train_gpu.py` prints both and `best_checkpoint.json` records them).
Add `--use_agents` (optionally with `--use_llm`) to let the Frame Agent and
Training Control Agent (Layer 4) choose frame-selection and densify/early-stop
decisions during this same run -- see the Layer 4 quickstart above and
`docs/AGENTIC_ARCHITECTURE.md`; every other flag behaves identically whether
or not this one is passed.

## Repository map

```
layer1_cpu_sandbox/     CPU digital twin: synthetic data, perception, reconstruction, evaluation
layer2_interfaces/      Python ABCs: CPU prototype <-> GPU replacement contracts, module by module
layer3_gpu/              CUDA kernels + C++ bindings + PyTorch training loop (confidence-aware 3DGS)
layer4_agents/           Agents that decide/retry/diagnose around the pipeline: Frame, Reconstruction,
                         Evaluation, and Training Control agents, plus bounded retry orchestrators
                         (Layer 1 CPU pipeline and, via train_gpu.py --use_agents, Layer 3 GPU training).
                         Fully optional -- every entrypoint above works identically without this layer.
tests/                  pytest suite (175 tests total; 152 run and pass on CPU alone, the remaining 23
                         -- in test_layer3_gaussian_model.py and test_layer3_confidence_gaussian_cpu.py --
                         need `pip install torch` and are skipped -- not failed -- without it)
docs/                   architecture rationale (Layers 1-3 and, separately, Layer 4's agents),
                         prototype-vs-final mapping, novelty candidates, missing components,
                         SIH plan, experiment plan, stop conditions
```

## Honesty guarantees this repo tries to uphold

- Every number in `layer1_cpu_sandbox/outputs/run1/summary.json` was computed by
  running the code in this repo, on synthetic ground truth, in this environment —
  not fabricated, not copied from a paper.
- Nothing here claims GPU execution that didn't happen. `STATUS.md` is blunt about
  what is still just source code waiting for Environment B.
- PSNR/SSIM are never used as a stand-in for geometric or metric accuracy —
  see the visual/geometry/metric-accuracy/completeness split enforced throughout
  `layer1_cpu_sandbox/evaluation/`.
- The `.obj` mesh export is CPU-tested end-to-end and has actually been run
  against this repo's own real point-cloud output, not just synthetic test data —
  but has never run against real GPU-trained Gaussians (no GPU in this
  environment). `STATUS.md` says so explicitly rather than implying otherwise.
- Layer 4's agents never invent numbers: every metric an Evaluation Agent cites
  comes from the unmodified evaluation code above, passed through unchanged
  (`layer4_agents/schemas.py` structurally discards any numbers a model tries to
  supply itself). Nothing claims a real Claude API call happened anywhere in this
  repo's own test runs — `layer4_agents/llm_agents.py` says plainly that it was
  implemented without network access to actually run it, and `--use_agents` on
  `train_gpu.py` specifically has not yet had a real GPU run either way.

## Automatic VGGT post-training deliverable

For real-data runs, `train_gpu.py` now automatically performs the validated VGGT confidence-fusion and Poisson Trim001/Clean500 export after Gaussian training ends. The OBJ and GLB are written beneath the same run output directory in `vggt_trim001_clean500/`. See [docs/RUN_GAUSSIAN_THEN_VGGT.md](docs/RUN_GAUSSIAN_THEN_VGGT.md) for Colab installation, outputs, and rerun instructions.

Mesh exports also include an in-sample point-to-surface consistency and topology report. Its internal screening targets are labeled as project-only and are not absolute accuracy claims; see docs/RUN_GAUSSIAN_THEN_VGGT.md.
## Colab one-click reconstruction and GeoTwin viewer

For the current real-video workflow, run `notebooks/GeoTwin_Colab_Run_and_View.ipynb` after `colab_bootstrap.sh`. It runs Gaussian training and the automatic VGGT Trim001/Clean500 mesh export, then offers an **Open GeoTwin viewer** button. The viewer embeds directly in Colab and loads the run GLB plus matching SRT without a manual server or download/re-upload. Its optional inferred gap fill is an amber-marked, view-only screen overlay; it does not modify mesh geometry. See `docs/RUN_GAUSSIAN_THEN_VGGT.md` for the full workflow and limitations.
