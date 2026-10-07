# Getting Started: Reproduce, Test, Feed Real Data, Verify Results

This doc assumes you have `sih2026-single-pass-3d-recon.zip` extracted somewhere,
e.g. `~/sih2026-3d-recon/`. Everything in Part A/B works on a plain laptop —
no GPU needed. Part C is for Colab. Part D covers real datasets. Part E tells
you exactly how to check that nothing was faked. Part F runs the actual GPU
training loop for the first time — nothing in that section has executed
anywhere yet, so treat it as a real experiment, not a formality.

---

## Part A — Reproduce on your own machine (CPU only)

### A1. Prerequisites
- Python 3.10–3.12 (repo was built/tested on 3.12.3)
- pip
- ~200MB free disk, no GPU required

### A2. Setup
```bash
cd ~/sih2026-3d-recon
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### A3. Run the full experiment
```bash
python3 -m layer1_cpu_sandbox.run_experiment \
    --out_dir layer1_cpu_sandbox/outputs/run1 \
    --seed 0
```
Takes ~50 seconds on a single CPU core. Prints progress `[1/6]` through
`[6/6]`, ends by dumping `summary.json` to your terminal. Re-run with
`--seed 1`, `--seed 2`, etc. — every stage is seeded, so a given seed always
reproduces the same numbers exactly; different seeds give a different random
scene/trajectory/noise draw, which is how you sanity-check that results
aren't a one-off fluke (see Part E3).

### A4. Run the Layer-3 CPU-testable check
```bash
pip install torch   # CPU wheel; ~500MB-2GB download depending on platform/version
python3 layer3_gpu/python/confidence_gaussian_model.py
```
This actually runs gradient descent (not a mock) and prints the loss
dropping, plus the single-observation confidence-cap check. No GPU involved
— this validates the *logic*, not GPU performance.

### A4. Run the Layer-3 CPU-testable checks
```bash
pip install torch   # CPU wheel; ~500MB-2GB download depending on platform/version
python3 layer3_gpu/python/confidence_gaussian_model.py
python3 layer3_gpu/python/confidence_propagation.py
```
Both actually run real computation (not mocks) and print PASS/FAIL for each
check. The first: gradient descent optimizing a toy scene, plus the
single-observation confidence-cap check. The second: temporal+spatial
confidence propagation (spatial diffusion measurably pulling up an
under-evidenced neighbor's confidence) and the OSAD-inspired optimizer-
struggle signal correctly distinguishing sustained directional gradient
push from oscillation, using real Adam optimizer state. No GPU involved —
this validates the *logic*, not GPU performance.

### A5. Exercise the production Gaussian model (still CPU-only)
```bash
python3 -c "
import torch, sys; sys.path.insert(0, 'layer3_gpu/python')
from gaussian_model import GaussianModel, GaussianTrainingConfig
model = GaussianModel(GaussianTrainingConfig(sh_degree=0))
model.initialize_from_fused_points(torch.rand(20,3)*10, torch.rand(20,3), torch.rand(20), device='cpu')
opt = model.setup_optimizer()
print('initialized', model.n_gaussians, 'Gaussians; optimizer has', len(opt.param_groups), 'param groups')
"
```
This is the same `GaussianModel` class `train_gpu.py` uses — position/scale/
rotation/opacity/SH parameterization, prune/clone/split, and the optimizer-
state bookkeeping that has to stay correct through structural changes
(covered by `tests/test_layer3_gaussian_model.py`). Rendering is the only
part that needs a GPU + the external rasterizer; everything else here runs
on a laptop.

---

## Part B — Running the test suite

### B1. Run everything
```bash
pip install pytest      # already in requirements.txt
python3 -m pytest tests/ -v
```
Expect `32 passed, 2 skipped` if torch isn't installed (the two skipped items
are whole files — `test_layer3_confidence_gaussian_cpu.py` and
`test_layer3_gaussian_model.py`, 23 tests between them — skipped via
`pytest.importorskip("torch")`, not failed). With torch installed, expect
`55 passed`. Takes ~10-25 seconds on CPU either way (the synthetic-dataset
and mesh-reconstruction tests render/reconstruct real data but are still
fast at this scale).

### B2. Run one file, or one test
```bash
python3 -m pytest tests/test_layer1_synthetic.py -v              # scene/trajectory/renderer
python3 -m pytest tests/test_layer1_metrics.py -v                 # metrics + confidence formula
python3 -m pytest tests/test_real_data.py -v                      # DJI CSV/video ingestion (mock data)
python3 -m pytest tests/test_mesh_export.py -v                    # point cloud -> .obj mesh export (needs open3d, in requirements.txt)
python3 -m pytest tests/test_layer3_confidence_gaussian_cpu.py -v # needs torch, skips cleanly without it
python3 -m pytest tests/test_layer3_gaussian_model.py -v          # production Gaussian model, needs torch

python3 -m pytest tests/test_layer1_metrics.py::test_confidence_single_observation_is_capped -v
```

### B3. What each file actually checks
| File | Checks |
|---|---|
| `test_layer1_synthetic.py` | Scene gen is deterministic per seed, differs across seeds, trajectory is a single continuous pass (not a loop), renderer produces non-empty frames, degradation actually changes pixels |
| `test_layer1_metrics.py` | PSNR/SSIM behave correctly on known cases (identical images → near-perfect score, more noise → lower PSNR), Chamfer distance is 0 for identical point sets and grows with noise, **confidence is higher with more observations**, **single-observation points are hard-capped**, gating actually prunes low-confidence points |
| `test_real_data.py` | DJI CSV parser (both formats), lat/lon→ENU conversion cross-checked against an independent haversine formula, real video I/O via OpenCV, and the full real-dataset+pipeline path end to end on synthetic mock files matching the real schema |
| `test_mesh_export.py` | Point cloud → `.obj` mesh export: Poisson and ball-pivoting both produce a non-empty mesh whose extent tracks the input, `keep_mask` actually filters before reconstruction, too-few-points-after-filtering raises a clear error, `[0,255]`-vs-`[0,1]` colors are auto-normalized to the same result, a colorless input produces a mesh with no vertex colors (guards the black-mesh bug described in `STATUS.md`), and the written `.obj` is byte-for-byte readable back with matching vertex/triangle counts and colors |
| `test_layer3_confidence_gaussian_cpu.py` | Confidence-cap property in the PyTorch implementation, a toy Gaussian-splat optimization loop reducing loss via real `.backward()`, temporal+spatial confidence propagation, the OSAD-inspired struggle signal, and **the key architectural guarantee**: a massive struggle signal in a poorly-observed region cannot bypass the observation-confidence gate |
| `test_layer3_gaussian_model.py` | The production 3-D Gaussian model: init shapes, low-confidence points get lower initial opacity, **optimizer state correctly survives prune/densify** (this guards a real bug caught during development — see `STATUS.md`), clone/split Gaussian-count arithmetic, clones inherit parent confidence, gradient accumulation only touches visible Gaussians, a saved+reloaded checkpoint is still trainable, and (added alongside mesh export) a reloaded checkpoint's confidence reflects everything training actually accumulated rather than silently falling back to the creation-time prior |

If you want to break something on purpose to see a test catch it: try setting
`single_view_cap=1.0` in `confidence.py`'s `compute_confidence` call and
re-run `test_layer1_metrics.py` — `test_confidence_single_observation_is_capped`
should fail. That's the kind of check that tells you the tests are real.

---

## Part C — Colab (GPU) — brings the CUDA build up and now runs real training

**Be clear on what changed since the last round:** the CUDA kernel's
`gate_densify_prune` signature changed (added the OSAD-inspired
optimizer-struggle-signal gate), and the actual GPU training loop
(`train_gpu.py`) now exists and is ready to run. **If you already built the
extension in an earlier session, you MUST rebuild it — the old `.so` will
not have the new signature and calls to it will fail or silently use stale
behavior.**

### C1. Get the code into Colab
You don't have this in a GitHub repo yet, so easiest path:
1. Open https://colab.research.google.com → New Notebook
2. Runtime → Change runtime type → **T4 GPU** (or better) → Save
3. Upload the zip and extract:
```python
from google.colab import files
uploaded = files.upload()   # pick sih2026-single-pass-3d-recon.zip
!unzip -oq sih2026-single-pass-3d-recon.zip   # -o = overwrite, important if re-uploading
%cd sih2026-3d-recon
```
(If you push this to your own GitHub instead, just `!git clone <your-url>`
and `%cd` into it — then `colab_bootstrap.sh`'s `git clone` line works as-is.)

### C2. Rebuild the CUDA extension (mandatory if you built it before this round)
```python
!rm -rf layer3_gpu/build layer3_gpu/*.so   # clear any stale prior build
!pip install -q -r requirements.txt -r requirements-gpu.txt
!pip install -q "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
%cd layer3_gpu
!python3 setup.py build_ext --inplace
%cd ..
```

### C3. Run the smoke test (now 5 checks, not 4)
```python
!python3 layer3_gpu/python/colab_smoke_test.py
```
Expect `[1/5]` through `[5/5]`. The new `[4/5]` check is the struggle-signal
guarantee — it must print PASS confirming a huge optimizer-struggle signal
in a poorly-observed region does NOT trigger densification, while the same
signal on a well-observed Gaussian does. This is the GPU-side confirmation
of the exact behavior already proven on CPU
(`tests/test_layer3_confidence_gaussian_cpu.py::test_optimizer_struggle_signal_cannot_bypass_observation_gate`).

### C4. What to watch for
- `nvidia-smi` must show a GPU. If not: Runtime → Change runtime type → GPU.
- The `diff-gaussian-rasterization` install compiles CUDA — if it fails,
  it's almost always a CUDA-toolkit/torch-version mismatch; check the error
  for a version number and match `requirements-gpu.txt` to Colab's
  pre-installed torch (`python3 -c "import torch; print(torch.__version__)"`)
  rather than pinning your own.
- `colab_smoke_test.py` step `[2/5]` is the original, still-important check:
  the CUDA kernel's confidence output against the CPU reference formula to
  <1e-4 tolerance. If that fails, do not trust anything built on top of the
  kernel — fix it first (most likely cause: an architecture mismatch in the
  CUDA build, see `CMAKE_CUDA_ARCHITECTURES` in `layer3_gpu/CMakeLists.txt`).

---

## Part D — Datasets

### D1. Right now, you don't need any external dataset
The synthetic generator (`layer1_cpu_sandbox/synthetic/`) creates its own
scene, trajectory, camera frames, and ground truth on every run — that's
the whole point of the CPU sandbox. Nothing in Part A/B needs you to
download anything.

### D2. For validating against real-world-like data before a live demo
Recommended, roughly in order of usefulness for THIS problem (single-pass
oblique aerial, buildings/roads/vegetation, needs known camera poses):

| Dataset | Why it's useful | Link |
|---|---|---|
| **UrbanScene3D / Mill19** | Real large-scale aerial captures with known camera poses, used by several NeRF/3DGS aerial papers — closest match to this project's scene type | https://vcc.tech/UrbanScene3D |
| **WHU MVS/Stereo dataset** | Aerial oblique multi-view stereo with GT DSM (digital surface model) — good for geometric/height-accuracy validation, analogous to your `metrics_metric_accuracy.py` | http://gpcv.whu.edu.cn/data/WHU_MVS_Stereo_dataset.html |
| **ISPRS benchmark datasets** | Classic photogrammetry benchmark with control points/GCPs — good for validating georeferencing error specifically | https://www.isprs.org/education/benchmarks.aspx |
| **Mid-Air** | Synthetic but photorealistic drone-flight dataset with GT depth/pose/IMU — a good stepping stone between this repo's toy synthetic scenes and real footage | https://midair.ulg.ac.be/ |
| **Your own DJI/drone footage** | The most direct real "single pass" input — see D4 below for exactly what to export from the flight controller | n/a |

### D3. Turning a multi-pass dataset into a genuine single-pass experiment
Public datasets usually have overlapping multi-pass coverage (that's how
they get dense ground truth). To test THIS project's actual claim (single
flight line only), sort frames by acquisition order/trajectory and keep
only one contiguous flight line as input; hold out every other pass's
frames purely as extra evaluation viewpoints (never fed to reconstruction).
This is exactly the pattern `evaluate_result.py::_interpolated_novel_pose`
already uses on the synthetic data — same discipline, real data.

### D4. Real deployment input — what you'd actually export from a drone
Matches the PS's mandatory/optional input list directly:

| PS input | Real-world source | Format you'd export |
|---|---|---|
| Drone video (mandatory) | Flight controller / camera SD card | `.mp4`/`.mov`, 1080p or 4K |
| GPS coordinates (mandatory) | Flight log, or DJI's embedded `.SRT` subtitle file | timestamped lat/lon/alt |
| Flight metadata (mandatory) | Mission planner export or DJI flight log (`.DAT`/`.txt`) | commanded gimbal pitch/roll/yaw, altitude, speed |
| IMU (optional) | Flight controller log | timestamped accel/gyro |
| Barometric altitude (optional) | Flight controller log | timestamped altitude |
| Camera intrinsics (optional) | Camera spec sheet, or a quick checkerboard calibration (OpenCV `calibrateCamera`) if unknown | fx, fy, cx, cy (+ distortion if you want to be thorough) |
| RTK/PPK (optional) | RTK-enabled drone, or post-processed PPK correction file | corrected high-precision GPS trace |

**Honest gap:** there is no real-data loader in this repo yet — `pipeline.py`
currently only consumes the synthetic `SyntheticDroneDataset` object. Writing
one is a well-scoped, moderate task (video frame extraction via `cv2.VideoCapture`,
parsing a GPS/SRT log into the same `NoisySensorTrace` shape used in
`synthetic/degradation.py`, and building `DroneFrame` objects from real
frames instead of `render_frame()` output) — it's not built because it
wasn't asked for yet. Say the word and I'll build it against this exact
interface next.

### D5. Running it on real data — step by step (e.g. the Zenodo 3604005 files)

This is now implemented (`layer1_cpu_sandbox/real_data/`), tested against synthetic
mock files matching the real CSV schema exactly, and its CLI has been run
end-to-end (see `STATUS.md`). It has NOT been validated against an actual
downloaded DJI file yet — do that now, following these steps.

**Important first:** if you're using the Zenodo 3604005 dataset specifically,
`DJI_0013.MOV` is a **stationary hover** — the drone stays in place while the
gimbal pitches from forward to down. No translation means no triangulation
baseline, so multi-view depth estimation will be weak-to-nonexistent on it
almost by construction. `DJI_0004.MOV` (same record) is the drone **orbiting**
a point — real camera translation, much better for actually testing
reconstruction quality. Use 0013 only to confirm the ingestion plumbing
works; use 0004 (or your own translating flight) to judge reconstruction
quality.

```bash
# In Colab, after your usual setup:
!wget "https://zenodo.org/records/3604005/files/DJI_0004.MOV?download=1" -O flight.mov
!wget "https://zenodo.org/records/3604005/files/DJIFlightRecord_2018-07-04_%5B11-19-31%5D.csv?download=1" -O flight.csv

!python3 -m layer1_cpu_sandbox.real_data.run_real_experiment \
    --video flight.mov --log flight.csv \
    --out_dir outputs/real_run1 \
    --target_fps 2.0 --max_frames 60 --width 320 --height 240
```

What it prints, in order:
1. Video/log info, which CSV format was auto-detected, and — importantly —
   the **flight footprint span**: if this comes back under ~5m, you've got a
   hover-type clip (like DJI_0013) and reconstruction quality will be poor;
   that's the code telling you *why*, not a bug.
2. Frame counts kept/rejected by the same classical quality filter used on
   synthetic data.
3. Self-consistency numbers (PSNR/SSIM/LPIPS) — see the important caveat
   below.
4. Output file locations.

**What "evaluation" means here, precisely — there is no ground truth for real
footage.** `real_pipeline.py::evaluate_real_result` does the only honest
check available: it re-renders the reconstruction from a pose that WAS
actually captured, and compares that render to the REAL frame the camera
took at that exact pose. That's a genuine "does the reconstruction reproduce
what the camera saw" check — not a geometric accuracy claim (no Chamfer/
height-error/completeness numbers exist for real data, and the code doesn't
pretend otherwise; check `summary.json`'s `NOTE` field, which says this
explicitly every time).

**Camera intrinsics are ASSUMED** (84° horizontal FOV, DJI Phantom 4 Pro's
commonly-cited spec) unless you calibrate. This matters for anything
metric — if you later add real geometric-accuracy claims, calibrate first
using the dataset's own `calibration.MOV` (a standard OpenCV checkerboard
calibration; not yet built into this repo — ask if you want it).

**Viewing the real-data outputs:** same as Part E — `reconstruction_real.ply`
and, when the confidence-gated point cloud has enough surviving points to
reconstruct a surface from, `reconstruction_real.obj` (open3d/MeshLab),
`real_qualitative.png` (actual frame vs. re-rendered reconstruction vs.
error map), `summary.json`.

---

## Part E — Verifying the metrics and rendered images (i.e., checking nothing's faked)


### E1. Where everything lands
After Part A3, look in `layer1_cpu_sandbox/outputs/run1/`:

| File | What it is | How to open |
|---|---|---|
| `summary.json` | Headline numbers | any text editor, or `cat summary.json` |
| `ablation_table.csv` / `.md` | Per-module ablation | Excel/Sheets/Numbers (csv), or any markdown viewer/GitHub (md) |
| `baseline_table.csv` / `.md` | Single-factor comparisons | same as above |
| `confidence_vs_error.png` | THE core-novelty validation figure | any image viewer |
| `trajectory_coverage.png` | Flight path + what was/wasn't observable | any image viewer |
| `qualitative_comparison.png` | GT render vs. reconstruction render vs. error heatmap | any image viewer |
| `reconstruction_full_method.ply` | The actual reconstructed point cloud | see E2 |
| `reconstruction_full_method.obj` | The same points, reconstructed as a triangle mesh | see E2b |
| `ground_truth_surface.ply` | The scene's true surface, for comparison | see E2 |

### E2. Viewing the `.ply` point clouds
Easiest options:
- **MeshLab** (free, Windows/Mac/Linux) — File → Import Mesh → pick the `.ply`
- **CloudCompare** (free) — same idea, more measurement tools (handy since
  this project cares about metric measurement)
- **Blender** — File → Import → Stanford PLY
- **In Python** (works locally or in Colab; `open3d` is already installed by
  Part A's `pip install -r requirements.txt`, no separate install needed):
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("layer1_cpu_sandbox/outputs/run1/reconstruction_full_method.ply")
o3d.visualization.draw_geometries([pcd])   # local machine with a display
```
In Colab (no display), do a quick matplotlib 3D scatter instead:
```python
import numpy as np, matplotlib.pyplot as plt
pts = np.asarray(pcd.points); cols = np.asarray(pcd.colors)
fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(projection='3d')
ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=cols, s=2)
plt.show()
```

### E2b. Viewing the `.obj` mesh
This is a real, watertight-seeking triangle mesh (Poisson surface
reconstruction over the same confidence-fused points as the `.ply` above),
with colors baked in per-vertex — a single file, no companion `.mtl`/texture
needed. Any of the following open it directly:
- **Blender** — File → Import → Wavefront (.obj)
- **MeshLab / CloudCompare** — File → Import Mesh
- **In Python**, same `open3d` already installed for E2:
```python
import open3d as o3d
mesh = o3d.io.read_triangle_mesh("layer1_cpu_sandbox/outputs/run1/reconstruction_full_method.obj")
o3d.visualization.draw_geometries([mesh])   # local machine with a display
```
If it looks bulgier/smoother than the point cloud in E2, that's expected —
Poisson reconstruction is a smooth interpolating fit, not a literal
connect-the-dots over the points (see
`layer1_cpu_sandbox/reconstruction/mesh_export.py`'s docstring for exactly
what it does and does not claim about filling gaps).

### E3. Sanity-checking that a metric isn't fabricated
Three concrete ways, cheapest first:
1. **Re-run with a different seed** (`--seed 1`, `--seed 2`, ...) and check
   the numbers move sensibly (not identical, not wildly inconsistent, same
   rough ballpark and same rank-ordering across ablation configs). Real,
   computed numbers behave like this; hand-typed numbers usually don't
   survive this check cleanly.
2. **Recompute one metric by hand** in a scratch script — e.g. reload the
   saved `.ply` files and independently recompute Chamfer distance:
   ```python
   import numpy as np, open3d as o3d
   from scipy.spatial import cKDTree
   recon = np.asarray(o3d.io.read_point_cloud("layer1_cpu_sandbox/outputs/run1/reconstruction_full_method.ply").points)
   gt = np.asarray(o3d.io.read_point_cloud("layer1_cpu_sandbox/outputs/run1/ground_truth_surface.ply").points)
   d1,_ = cKDTree(gt).query(recon); d2,_ = cKDTree(recon).query(gt)
   print("Chamfer:", 0.5*(d1.mean()+d2.mean()))   # should match summary.json's geometry_chamfer_distance_m
   ```
3. **Read the metric code directly** — every metric lives in
   `layer1_cpu_sandbox/evaluation/metrics_*.py`, each under ~100 lines,
   docstring'd with which section of the brief it answers. Nothing is
   hidden behind a black-box call.

### E4. Known-answer checks already in the test suite
`tests/test_layer1_metrics.py` already encodes several "this metric must
behave sensibly" checks (identical images → near-perfect PSNR, added noise
→ lower PSNR, identical point sets → zero Chamfer, more observations →
higher confidence, single observation → capped confidence). Running
`pytest tests/test_layer1_metrics.py -v` is itself a verification step, not
just a formality — read the `assert` lines to see exactly what's being
guaranteed.

---

## Part F — Running the actual GPU training loop (`train_gpu.py`)

This is the piece that turns "we have a CPU-validated idea" into "we
measured whether it's true." Nothing in this section has been executed
anywhere yet — this is the first real test.

### F1. Start with synthetic data (controlled, known ground truth)
```bash
cd /content/sih2026-3d-recon   # or wherever you extracted it in Colab
python3 layer3_gpu/python/train_gpu.py \
    --source synthetic --seed 0 --max_iterations 1000 \
    --out_dir outputs/gpu_train_synth1
```
What happens, in order: builds the same synthetic single-pass dataset as
`run_experiment.py`, runs the Layer-1 pipeline to get confidence-fused
points, initializes one Gaussian per point, then trains — logging
`n_gaussians`, `loss_total` (and its `l1`/`ssim` components separately, on
purpose — see `losses.py`'s `combined_loss` docstring for why they're kept
separate) every 100 iterations, and saving a checkpoint `.ply` every 500.

**Start small.** 1000 iterations first, not the 3000 default — this is an
unexecuted code path; if something's going to fail, better to find out in
two minutes than twenty. Watch for:
- `n_gaussians` should change at iteration 300, 500, 700, 900 (the
  densification interval) — if it never changes, the confidence gate might
  be rejecting everything (check the printed mean confidence at each
  checkpoint) or densify_from_iter/until_iter might not overlap with your
  `--max_iterations`.
- `loss_total` should trend down, even if noisily. If it's flat or NaN
  immediately, that's a real bug to report back, not something to push
  through.

### F2. Then try real footage
```bash
python3 layer3_gpu/python/train_gpu.py \
    --source real --video flight.mov --log flight.csv \
    --max_iterations 1000 --out_dir outputs/gpu_train_real1
```
Use `DJI_0004.MOV` (the orbit) rather than `DJI_0013.MOV` (the stationary
hover) — see Part D5 above for why translation matters here. Expect this
to be harder to get clean loss curves from than the synthetic case, both
because real footage is real footage and because (per the earlier
diagnosis) water/specular content specifically defeats the classical
Lambertian-consistency assumption several of the confidence signals rely
on — if training struggles here, that's diagnostic information, not
necessarily a bug.

### F3. If it fails
Two classes of failure, treated differently:
- **Import/build errors** (e.g. `diff_gaussian_rasterization` not found,
  shape mismatches in the rasterizer call): almost certainly something in
  `_camera_to_rasterizer_settings`/`render_gaussian_view` in `train_gpu.py`
  — this is the one piece of the whole pipeline that has never run against
  the real external API, only been reviewed against its documented
  interface. Send me the exact traceback.
- **Runs but produces garbage** (loss doesn't decrease, point count
  explodes/collapses to zero): send me the printed per-checkpoint log
  (`n_gaussians`, mean confidence, loss components) — that's usually enough
  to tell whether it's a hyperparameter issue (thresholds in `TrainConfig`
  tuned on nothing but reasoning, never on data) versus something more
  structural.

### F4. Viewing the result
Same as Part E2 — `outputs/gpu_train_synth1/checkpoint_iter1000.ply` opens
in MeshLab/CloudCompare/open3d exactly like the CPU-sandbox point clouds
did. `training_history.json` has the full per-logged-iteration loss curve
if you want to plot it yourself. Once training reaches `--max_iterations`,
look for `mesh_iter<N>.obj` in the same output directory (see Part E2b for
how to view it) — that's the actual deliverable this whole loop is building
toward: an opacity/confidence-gated mesh of the final Gaussians, exported
automatically unless you passed `--no-export_mesh`. If it's missing, check
the console output right after training finishes — a meshing failure is
reported there and does not fail the run; `export_obj.py` can retry it
afterwards with different `--min_opacity`/`--min_confidence`/`--method`
against the `full_state_iter<N>.pt` checkpoint that's already on disk,
without retraining.
