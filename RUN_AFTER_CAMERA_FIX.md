# Colab validation after the camera-bridge correction

Do not compare this run numerically to a checkpoint made with the old camera
matrix.  Its image registration was invalid, so all prior model weights should
be treated as diagnostic artifacts rather than a resume point.

## 1. Short camera/geometry validation

Run this before allowing Gaussian growth.  It tests whether the corrected
camera bridge and telemetry-derived poses can align the captured frames at all.

```bash
cd /content/sih2026-3d-recon
python3 layer3_gpu/python/train_gpu.py \
  --source real --video /content/flight.mov --log /content/flight.csv \
  --out_dir outputs/camera_bridge_check \
  --max_iterations 1500 --densify_until_iter 0 \
  --camera_hfov_deg 84 --target_fps 2.0 --n_eval_views 999
```

`84` is a fallback only. Replace it with the calibrated horizontal FOV of the
actual drone/camera, after accounting for the resize to 640x480. Inspect the
three saved `render_iter*_frame*.png` files at iterations 500, 1000, and 1500.
If training views and held-out views are both globally displaced or blurred,
do not tune Gaussian count: validate log/video synchronization, gimbal-axis
conventions, and intrinsics first.

## 2. Capacity run only after alignment is credible

```bash
python3 layer3_gpu/python/train_gpu.py \
  --source real --video /content/flight.mov --log /content/flight.csv \
  --out_dir outputs/real_single_pass_v2 \
  --max_iterations 8000 --densify_until_iter 6500 \
  --camera_hfov_deg <CALIBRATED_HFOV> --target_fps 2.0 \
  --max_gaussians 150000 --min_densify_observations 2
```

Use the checkpoint with the best held-out SSIM, rather than the last one. A
static scene should improve visibly during the short validation. Water glint,
waves, refraction, and moving shadows should remain low-confidence regions;
sharp geometry there is not a valid quality target for a single static model.

The automatic `mesh_iter8000.obj` this run produces at the end is built from
the LAST iteration's Gaussians, not the best-held-out-SSIM one train_gpu.py
prints at the end (see above) — those are frequently different iterations
(see STATUS.md's overfitting finding). Re-export from whichever
`full_state_iter*.pt` best_checkpoint.json actually names instead:
```bash
python3 layer3_gpu/python/export_obj.py \
  --checkpoint outputs/real_single_pass_v2/full_state_iter<BEST_ITER>.pt \
  --out outputs/real_single_pass_v2/mesh_best_ssim.obj
```

The hard Gaussian budget is intentional.  With sparse single-pass coverage,
an ever-growing splat count can fit adjacent training images while degrading
held-out views.  Increase it only after held-out metrics improve consistently,
not because training loss falls.

For a ground/reef model, the default `--max_forward_z -0.5` excludes
horizon-facing frames such as a take-off or transit shot.  The program prints
their original frame ids. Use `--max_forward_z 1.0` only when deliberately
building a mixed-orientation model.
