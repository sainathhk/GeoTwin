# Experiment 2 — geometry-first VGGT twin

This experiment answers the project’s main failure question before spending time on another Gaussian run: **do the video frames produce one spatially coherent 3D scene?** It uses VGGT’s joint camera and depth prediction, preserves the full image field of view, excludes mixed focal-length camera blocks, and uses DJI SRT GPS path length to estimate a metric scale. It exports a shared point cloud and preview first. The TSDF mesh is a second step after that preview is inspected.

This is a geometry baseline, not a claim of survey-grade accuracy. GPS anchors approximate scale only; the result is not aligned to geographic north or a map coordinate system. VGGT can still fail on a long flight, zoom changes, low parallax, repeated structures, moving traffic, haze, or frames with weak overlap. The preview is a required checkpoint.

## 1. Prepare Colab and files

In Colab choose **Runtime → Change runtime type → T4 GPU** (or L4/A100 if available). Put the project ZIP, `DJI0004.MP4`, and matching `DJI0004.srt` in Google Drive. Mount Drive and set paths (edit the three paths to match your Drive folders):

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ZIP = '/content/drive/MyDrive/sih2026-3d-recon-zoom-mesh-fix-v6-exp2.zip'
VIDEO = '/content/drive/MyDrive/DJI0004.MP4'
SRT = '/content/drive/MyDrive/DJI0004.srt'
```

## 2. Unpack and install without replacing Colab’s CUDA PyTorch

```bash
!mkdir -p /content/exp2_project
!unzip -q -o "$PROJECT_ZIP" -d /content/exp2_project
%cd /content/exp2_project
!nvidia-smi
!python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

The archive contains a top-level project directory. Locate it once:

```bash
!find /content/exp2_project -maxdepth 3 -name run_exp2_vggt_geometry.py -print
```

If that prints `/content/exp2_project/sih2026-3d-recon-zoom-mesh-fix-v6/layer3_gpu/python/run_exp2_vggt_geometry.py`, change into that project root:

```bash
%cd /content/exp2_project/sih2026-3d-recon-zoom-mesh-fix-v6
```

Install the required runtime packages. **Do not run `pip install torch` or the project’s full GPU requirements file**; that can replace the CUDA-enabled PyTorch supplied by Colab.

```bash
!pip -q install "numpy<2" pandas opencv-python-headless open3d matplotlib pillow einops safetensors huggingface_hub
!pip -q install --no-deps git+https://github.com/facebookresearch/vggt.git
```

If Colab asks for a runtime restart after changing NumPy, restart once, remount Drive, return to the project directory, and continue with the commands below. Confirm CUDA before the long model pass:

```bash
!python -c "import torch, open3d, vggt; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), open3d.__version__)"
```

## 3. Run the shared geometry pass

```bash
!python layer3_gpu/python/run_exp2_vggt_geometry.py \
  --video "$VIDEO" \
  --log "$SRT" \
  --out_dir /content/exp2_geometry \
  --target_fps 4.0 \
  --max_frames 18 \
  --width 640 --height 480 \
  --camera_hfov_deg 84 \
  --assumed_gimbal_pitch_deg -45 \
  --max_focal_drift_frac 0.10 \
  --confidence_threshold 1.5 \
  --top_mask_fraction 0.08 \
  --voxel_size_m 0.40 \
  --max_depth_m 250
```

`--target_fps 4` samples candidate frames densely enough to find the longest stable focal block. The script then selects at most 18 evenly spaced frames from that block for one VGGT pass, which keeps the image set small enough for a typical T4. It will stop if fewer than six frames remain. If it runs out of GPU memory, retry with `--max_frames 12`; keep those frames in one joint pass because separate passes have separate coordinate systems.

The `-45` gimbal pitch is required by the existing SRT parser but is **not** used as VGGT’s camera orientation. VGGT estimates visual poses. The SRT positions are used only to derive the approximate global scale. If the focal block is too short, do not relax the focal drift threshold to combine the DJI wide and tele modules; first review the reported focal values and select a contiguous single-module segment.

## 4. Inspect before surface extraction

The geometry run creates:

```text
/content/exp2_geometry/exp2_geometry_preview.png
/content/exp2_geometry/vggt_geometry_metric.ply
/content/exp2_geometry/exp2_summary.json
/content/exp2_geometry/vggt_depth_cache.npz
```

Show the preview in Colab:

```python
from IPython.display import display, Image
display(Image('/content/exp2_geometry/exp2_geometry_preview.png'))
```

The preview shows three orthographic projections and the estimated camera path. Continue only if the point map has one coherent footprint, broad road/ground surfaces, and raised building masses in compatible positions across the projections. Stop if it looks like a spherical shell, streaks from the camera path, scattered fragments, or disconnected view-wise slabs. In that case save `exp2_summary.json` and the preview; a TSDF cannot repair inconsistent camera/depth geometry.

For a closer inspection, download `vggt_geometry_metric.ply` and open it in CloudCompare or MeshLab. The colored cloud is the primary geometry result; the preview is only a quick diagnostic.

## 5. Build a surface only after the point map passes

```bash
!python layer3_gpu/python/build_exp2_tsdf_mesh.py \
  --cache /content/exp2_geometry/vggt_depth_cache.npz \
  --out_dir /content/exp2_geometry/mesh \
  --voxel_size_m 0.40 \
  --sdf_trunc_m 1.20 \
  --max_depth_m 250
```

This integrates confidence-filtered metric depth maps with the shared VGGT poses. It writes a colored PLY and OBJ, plus component/bounds statistics. A mesh with many small components or a small largest-component fraction is evidence that the geometry still needs work; do not smooth those statistics away and call it a twin.

## 6. Download deliverables

```python
from google.colab import files
files.download('/content/exp2_geometry/exp2_geometry_preview.png')
files.download('/content/exp2_geometry/vggt_geometry_metric.ply')
files.download('/content/exp2_geometry/exp2_summary.json')
# After the mesh step:
files.download('/content/exp2_geometry/mesh/exp2_vggt_tsdf_mesh.ply')
files.download('/content/exp2_geometry/mesh/exp2_vggt_tsdf_mesh.obj')
```

## Why this is the next experiment

The previous runs improved image fitting but did not show that their Gaussian field represented consistent scene geometry; the held-out render degradation also showed that training metrics were not enough. Experiment 2 bypasses 3D Gaussian fitting and its current extraction path for the first geometry decision. VGGT jointly predicts camera poses and depth across the selected video frames; DJI GPS supplies only a rough metric length scale. If the point cloud is coherent, TSDF gives a direct surface baseline. If it is not coherent, the failure is upstream (pose/depth/coverage), and another mesh or Gaussian tuning pass would not address it.

VGGT references: [official repository](https://github.com/facebookresearch/vggt), [official package setup](https://github.com/facebookresearch/vggt/blob/main/docs/package.md), [official image preprocessing implementation](https://github.com/facebookresearch/vggt/blob/main/vggt/utils/load_fn.py).

## Follow-up: refine the successful VGGT point-fusion quicklook

The successful quicklook used 22 frames, kept 665,792 points after confidence gating, and made 146,869 vertices / 292,673 triangles at Poisson depth 9. Its 94.9% largest-component fraction is encouraging. The attached statistics also report a 1.17 m median and 4.42 m maximum pose-to-telemetry residual; these are meaningful limits on geometric detail. A finer meshing depth can make the surface representation denser, but cannot recover detail absent from or inconsistent in the fused points.

Before the Colab runtime is reset, save the in-memory fused point cloud and confidence mask:

```python
import numpy as np
np.savez_compressed(
    '/content/exp2_fused_cloud.npz',
    positions=fused.positions.astype(np.float32),
    colors=fused.colors.astype(np.float32),
    keep_mask=decision.keep_mask.astype(bool),
    confidence=conf.confidence.astype(np.float32),
)
```

Upload the updated project ZIP, unzip it over the existing project, and run a depth-10 variant first:

```bash
!python layer3_gpu/python/mesh_exp2_poisson_sweep.py \
  --input_npz /content/exp2_fused_cloud.npz \
  --out_dir /content/exp2_poisson_sweep \
  --depths 10 \
  --density_trim_quantile 0.03 \
  --also_save_glb
```

If depth 10 is stable and Colab has enough RAM, try depth 11 in a separate run. Compare the roof and road edges, floating surfaces, component count, boundary edges, and triangle-area statistics. A higher face count alone is not an improvement. Keep the point confidence gate and input cloud fixed for this comparison; changing fusion and mesh parameters together would make it hard to tell what helped.

This Poisson path is the current useful result; the earlier TSDF empty surface is an extraction failure for this depth/pose cache. Do not keep retrying TSDF on the same cache. The OBJ cannot be made more geometrically detailed after the fact: refinement needs the saved fused point cloud. Vertex colors are carried on the mesh; a true UV texture atlas is a later appearance step and will not correct geometric misalignment.

## Active Experiment 2 route and latest trim comparison

The currently successful route is **VGGT pose + VGGT depth + VGGT intrinsics → DJI telemetry similarity alignment → this repository's point fusion and confidence gate → Poisson surface**. The adapter is `layer1_cpu_sandbox/real_data/vggt_pose_depth_source.py`. Use its pose/depth/intrinsics together; do not mix VGGT depth with the old SRT-intrinsics or plane-sweep path. The direct VGGT-to-TSDF script above is retained as a diagnostic branch; its earlier empty surface makes it a poor current candidate.

The latest depth-12 runs used the same 665,792 confidence-gated points:

| Poisson density trim | Vertices | Triangles | Components | Largest-component share | Boundary edges | Surface area |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 1,499,631 | 2,973,141 | 5,556 | 90.9% | 11,445 | 99,190.4 |
| 0.01 | 1,530,563 | 3,042,252 | 5,535 | 95.3% | 4,190 | 127,218.8 |

The 1% trim is the better current candidate: the reported boundary edges fell by about 63% and the largest component share increased. It still has thousands of components, so this is not the final cleaned asset. The 0.01 run's larger surface area and triangle-area p99 (0.516 vs 0.176) also mean some low-density bridges may have been restored; inspect these regions for false sheets. **Both runs were trimmed**; `--density_trim_quantile 0.0` is the true no-density-trim comparison.

Next, hold the frames, fused points, confidence mask, Poisson depth, and all other parameters fixed. First test whether Poisson's density trim itself is causing the remaining holes:

```bash
!python layer3_gpu/python/mesh_exp2_poisson_sweep.py \
  --input_npz /content/exp2_fused_cloud.npz \
  --out_dir /content/exp2_depth12_trim000 \
  --depths 12 \
  --density_trim_quantile 0.0 \
  --also_save_glb
```

This keeps the existing confidence gate; it does not restore points rejected by that gate. If the same holes remain, they come from sparse/missing observations or confidence filtering, and require additional overlapping viewpoints or a measured change to the confidence threshold. Don't fill them with smoothing and report them as observed geometry.

Then make a separate debris-cleanup candidate from the 1% trim result, removing only tiny mesh components:

```bash
!python layer3_gpu/python/mesh_exp2_poisson_sweep.py \
  --input_npz /content/exp2_fused_cloud.npz \
  --out_dir /content/exp2_depth12_trim001_clean500 \
  --depths 12 \
  --density_trim_quantile 0.01 \
  --min_component_vertices 500 \
  --also_save_glb
```

Inspect the removed-component count and compare façades/roof edges before choosing this version; a threshold can remove a genuinely observed but isolated small structure. The final deliverable is a cleaned, vertex-colored GLB/OBJ of the captured, observed urban corridor, plus the confidence/coverage report and camera path. It is a partial-scene twin: unseen backs and surfaces outside this single flight's coverage remain unknown. Do not call the approximate GPS-aligned scale survey-grade.

## Latest decision: stop the Poisson parameter loop

Two more depth-12 runs were completed on the same 665,792 gated points:

| Candidate | Vertices / triangles | Components | Largest share | Boundary edges | Surface area | Visual assessment |
|---|---:|---:|---:|---:|---:|---|
| trim 0.0 | 1,546,031 / 3,077,217 | 5,520 | 96.9% | 155 | 378,679.9 | Reject: boundary count improves by bridging broad unsupported regions into a large smooth sheet. |
| trim 0.01 + remove components under 500 vertices | 1,488,206 / 2,980,075 | 11 | 98.0% | 3,713 | 125,170.3 | Best current observed-surface candidate: recognizable road/buildings/trees, fewer floating specks, gaps remain visible. |

Use `trim001_clean500` as the current candidate for review. Do not interpret the low boundary count of `trim000` as better reconstruction: its screenshot shows the surface has bridged missing areas. The 500-vertex component filter removed 5,524 components, but a final visual check is still needed to make sure no small real structure was removed. Neither file is a complete or watertight digital twin.

### Next stages (bounded, geometry-first)

1. **Freeze this geometry candidate.** Keep the fused cloud, camera poses, confidence mask, summary JSON, current GLB/OBJ, and screenshots together. No more depth/trim sweeps unless a specific visible defect has a testable parameter hypothesis.
2. **Validate the candidate against the source views.** Reproject the cloud/mesh into several input frames across the clip, including held-out frames if available. Check road alignment, building footprints/roof edges, drift across time, and scale consistency. Record what is observed, uncertain, and unobserved. The current GPS similarity fit is approximate (reported pose residual median 1.17 m, max 4.42 m); do not claim survey-grade dimensions or georeferencing.
3. **Make the delivery asset.** If reprojection is acceptable, export the reviewed vertex-colored GLB/OBJ, camera path, preview images, and a short limitations/coverage report. A UV texture atlas is optional polish; it cannot fix geometry. Keep holes where the footage provides no support.
4. **Show the method's contribution with a controlled ablation.** Using the exact same VGGT predictions and poses, compare confidence-gated and ungated fusion (or two predeclared confidence thresholds), then report coverage, alignment, and geometry artifacts. Do not present Poisson component cleanup as a new reconstruction method. VGGT is the pretrained geometry backbone; the project contribution can be the drone-video adaptation and telemetry alignment, confidence-aware fusion, and a measured failure/coverage workflow, if the ablation supports those claims.
5. **Only if the validation exposes systematic gaps/drift, improve the capture/data.** A second overlapping crossing/top-down pass is the most direct way to add missing views. A single oblique flight cannot recover hidden backs or surfaces never seen by the camera. More Gaussian training is not the next action for this geometry-first target; it can later help appearance/novel-view rendering after geometry is accepted.

### Practical schedule estimate

Assuming the saved fused cloud/cache and run outputs are still available in Colab:

- Candidate freeze and visual review: about 30–60 minutes.
- Reprojection/coverage checks and a single controlled confidence ablation: about 2–4 hours of active work, plus Colab runtime time.
- Package a reviewable GLB/OBJ, camera/coverage figures, and SIH evidence: roughly half a day to one day.

This is an estimate for the current-footage MVP, not a guarantee that unseen geometry can be recovered. If the checks show incorrect poses/depth or substantial uncovered surfaces, a new capture or a changed geometry stage will be needed; schedule then depends on obtaining that data. The Poisson exploration phase is complete for now: the next decision is validation, not another open-ended trim loop.

## Submission MVP: georeferenced labels and unknown-coverage layer

For the presentation, use the local ENU coordinate frame from the same `dataset` used by the successful telemetry-aligned fusion. The viewer in `web/geotwin_mvp.html` converts X=east and Y=north from the matching DJI telemetry origin to approximate WGS84 labels. The current `web/geotwin_origin.json` is prefilled from the first GPS sample of `DJI0004.srt`; a user may also select a matching SRT/CSV together with a new model, in which case its first valid latitude/longitude pair supplies the origin. No coordinate typing is needed. The WGS84 output is approximate because the current camera-to-telemetry alignment residual is metre-scale, not survey control.

The viewer supports manually selected building centroids/corners and road/tree landmarks from either OBJ or GLB. A matching SRT/CSV can be selected alongside the model; the log is optional because this project's current DJI0004 origin is bundled as a fallback. It checks the loaded model bounds against the current clean500 candidate and reports a mismatch. It also lets the presenter place amber planar overlays over gaps, labelled `Inferred fill — not camera observed`. These are display-only unknown-coverage annotations, exported separately from the observed surface as GeoJSON polygons and restorable from a session JSON. They make the MVP understandable without claiming that hidden building backs or missing ground were recovered. See `docs/GEOTWIN_MVP.md` for the one-click launcher and limits.

For this deadline, highlight a curated set of important buildings manually. Automated building labeling and fully watertight repair are outside this MVP and would require semantic segmentation plus defensible inference/validation.
