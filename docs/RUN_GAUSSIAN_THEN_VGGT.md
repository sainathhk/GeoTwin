# Automatic Gaussian + VGGT prototype run (Colab)

`train_gpu.py` now runs the validated VGGT geometry pass automatically after a successful real-data Gaussian training run. The existing training command and render outputs stay in place; the final VGGT exports are placed under:

```text
<out_dir>/vggt_trim001_clean500/
  vggt_poisson_depth12.glb       # load this in GeoTwin MVP / 3D viewer
  vggt_poisson_depth12.obj
  exp2_fused_cloud.npz            # confidence-gated cloud used for meshing
  vggt_fusion_summary.json
  poisson_depth_sweep.json        # mesh topology + consistency statistics
  vggt_poisson_depth12_mesh_accuracy.txt   # readable quality/consistency report
  vggt_poisson_depth12_mesh_accuracy.json # all measured values + internal targets
  geotwin_viewer_assets/DJI0004.srt  # copied by the Colab viewer notebook for direct load
```

The automatic sequence is the one from the successful notebook experiment:

1. Re-extract candidate frames from the same video and DJI SRT/CSV using the run's HFOV, sampling rate, maximum candidate count, and assumed SRT gimbal pitch.
2. Select up to 28 evenly spaced views and estimate their cameras and depths together with the pretrained `facebook/VGGT-1B` model. A joint pass is needed to keep a single shared coordinate system; on smaller GPUs reduce `--vggt_max_frames` if needed.
3. Align the VGGT camera-path scale to the telemetry, fuse the depth with the project's existing voxel fusion and confidence logic (`voxel_size=0.4`, confidence gate `0.12`).
4. Export Poisson depth 12 with `density_trim_quantile=0.01`, remove components under 500 vertices, and save both OBJ and GLB.

The GLB/OBJ is a textured, approximate scene reconstruction, not a survey-grade, north-aligned georeferenced twin. SRT path length provides approximate metric scale for this pass; the separate GeoTwin viewer still handles geographic labels/origin metadata.

## Interpreting the mesh metrics

The automatic mesher now writes a readable TXT and machine-readable JSON report for the OBJ/GLB. It reports sampled point-to-surface distances (mean, RMSE, P50/P90/P95 and the fraction of fused confidence-gated points within 0.4 m, 1 m and 2 m), plus mesh topology (components, largest-component fraction, boundary/non-manifold edges, degenerate triangles, area and bounds). The report is also embedded in `poisson_depth_sweep.json`.

The report includes **internal screening targets only** for this project: P50 distance ≤0.40 m, P90 ≤1.00 m, ≥90% of sampled points within 1 m, largest component ≥90%, and approximate non-manifold-edge fraction ≤0.1%. The 0.4 m and 1 m distance targets are tied to the selected 0.4 m fusion voxel; they are not a published standard or SIH acceptance threshold. The output compares the generated mesh to a deterministic sample from the same VGGT fused cloud used to build it, so these values describe **in-sample surface consistency, not independent or absolute physical accuracy**. The SRT path provides approximate scale; absolute accuracy requires a separate surveyed point cloud, control points, or independently measured dimensions.

## Colab install

Use a fresh Colab GPU runtime (T4 is supported). From the extracted project root:

```bash
bash colab_bootstrap.sh
```

The bootstrap pins NumPy to `1.26.1` (the current VGGT upstream requirement) and installs SciPy, OpenCV, Open3D, VGGT, and its supporting packages. It leaves Colab's CUDA-matched PyTorch/torchvision in place. If Colab asks for a runtime restart after installing packages, restart, then rerun the bootstrap once.

Do not run the old notebook cells that uninstall NumPy or install NumPy 2.0. The repo also uses `requirements-gpu.txt` for Gaussian dependencies and `requirements-vggt.txt` for the official VGGT GitHub package.

## Training command

Use the normal training command. The postprocess is on by default for `--source real`:

```bash
python layer3_gpu/python/train_gpu.py \
  --source real \
  --video "/content/DJI0004.MP4" \
  --log "/content/DJI0004.srt" \
  --assumed_gimbal_pitch_deg -45 \
  --camera_hfov_deg 84 \
  --target_fps 1.5 \
  --max_frames 120 \
  --max_focal_drift_frac 0.10 \
  --pose_source visual_odometry \
  --vo_min_inliers 20 \
  --vo_max_frame_gap 3 \
  --keep_fraction 1.0 \
  --depth_min 30 \
  --depth_max 250 \
  --n_depths 128 \
  --depth_boundary_reject \
  --max_iterations 8000 \
  --densify_until_iter 6500 \
  --max_gaussians 200000 \
  --out_dir outputs/srt_visual_odometry_run
```

After the 8,000-iteration training and its exports complete, the same process launches VGGT and the Trim001/Clean500 mesh pass. Progress appears in the same Colab cell. The final status is also recorded at `outputs/srt_visual_odometry_run/vggt_postprocess_status.json`. To intentionally skip VGGT on a later run, pass `--no-auto_vggt_mesh`. To lower the VGGT frame count for limited GPU memory, pass `--vggt_max_frames 18` (all selected frames still go through one shared VGGT pass).

## Open the viewer directly in Colab

After running the bootstrap, open `notebooks/GeoTwin_Colab_Run_and_View.ipynb`. Set `PROJECT_DIR`, `VIDEO_PATH`, and `SRT_PATH` at the top of its code cell, then run that cell. It runs the configured Gaussian training and automatic VGGT/Clean500 export. When complete, it asks **Open GeoTwin viewer now?** Clicking **Open GeoTwin viewer** embeds the viewer in the notebook; it serves and loads the GLB and matching SRT from the run folder directly. There is no separate server command or browser file upload. Clicking **Skip viewer** ends without opening it.

Colab currently documents `serve_kernel_port_as_iframe` as the supported in-notebook preview route; opening a new browser tab from a proxied kernel port is deprecated in Colab's helper library. The notebook therefore opens the fully interactive viewer in an embedded notebook frame.

The viewer automatically turns on an amber-labeled, adjacent-color mean fill after the model loads. It is a screen-space inference overlay inside the projected main scene footprint. It does not edit the OBJ/GLB, and the overlay is not evidence that the missing surface was observed.

## Rerun geometry without repeating training

If training completed but the postprocess failed, rerun only the geometry stage:

```bash
python layer3_gpu/python/run_vggt_trim001_clean500.py \
  --video "/content/DJI0004.MP4" \
  --log "/content/DJI0004.srt" \
  --out_dir outputs/srt_visual_odometry_run \
  --target_fps 1.5 \
  --max_candidate_frames 120 \
  --max_vggt_frames 28 \
  --camera_hfov_deg 84 \
  --assumed_gimbal_pitch_deg -45
```
