# Non-COLMAP run for the supplied DJI clip

This project now includes a fast visual-odometry pose path for FrameCnt-style
DJI SRT files.  It is not COLMAP and does not build a workspace, extract a
sparse model, or run bundle adjustment.  It detects ORB features only between
the already selected consecutive frames, estimates their essential matrices,
and uses the SRT/GPS displacement solely to set translation scale.

Why this is needed: the supplied `DJI0004.srt` carries latitude, longitude,
relative altitude, and focal length, but it contains no gimbal yaw, pitch, or
roll.  The prior pipeline therefore used a fixed guessed pitch and derived
yaw from GPS track; that is not enough to construct correct plane-sweep
homographies.

## Command

Run this in the project root in Colab.  Replace the two paths with your upload
locations.  `--assumed_gimbal_pitch_deg` is still required by the SRT reader
to place the very first camera in the world frame; visual odometry recovers
all following relative rotations from the images, so it is no longer used as
the orientation of every camera.

```bash
python3 layer3_gpu/python/train_gpu.py \
  --source real \
  --video /content/DJI_20250516085606_0035_D.LRF.mp4 \
  --log /content/DJI0004.srt \
  --assumed_gimbal_pitch_deg -45 \
  --camera_hfov_deg 84 \
  --target_fps 1.5 --max_frames 120 \
  --max_focal_drift_frac 0.10 \
  --pose_source visual_odometry --vo_min_inliers 20 --vo_max_frame_gap 3 \
  --keep_fraction 1.0 \
  --depth_min 30 --depth_max 250 --n_depths 128 \
  --depth_boundary_reject \
  --seed_min_observations 2 --seed_min_consistency 0.25 \
  --seed_min_confidence 0.12 --min_seed_points 500 \
  --max_iterations 8000 --densify_until_iter 6500 \
  --max_gaussians 150000 \
  --out_dir outputs/srt_visual_odometry_run
```

The supplied clip was checked locally at the first, constant-focal-length
segment.  The new `--vo_max_frame_gap 3` behaviour handles an occasional weak
adjacent pair by finding the next valid image keyframe and dropping only the
untrackable intermediate frame; it never accepts a weak pose just to keep a
frame.  On the supplied LRF proxy at 2 fps, it retained 12/13 frames with a
median of 454 geometric inliers per retained pair.  The full MP4 can have
different feature/compression behaviour, which is why the command keeps
`--vo_min_inliers 20` explicit.  Do not lower it further.

## Read these lines before trusting the mesh

The run must print a `[visual odometry]` line with a healthy inlier total.
It must *not* print `FRUSTUM-SHAPED` in the geometry sanity report.  A frustum
verdict means the depth estimates are still not physical surfaces, so stop
there instead of accepting an OBJ based only on training-view render quality.

The trainer now also prints an `evidence-backed Gaussian seeds` count.  It
seeds only voxels observed by at least two frames with a non-ambiguous depth
cost curve; the old code incorrectly seeded every single-view depth guess,
which let a camera-shaped point volume turn into a pyramid mesh.  If the count
is below 500, the run stops before wasting GPU time.  Paste that one line into
the next message rather than lowering a threshold blindly.

The changing 24–70 mm focal length in the SRT is real.  The 0.10 focal-drift
filter deliberately keeps the initial constant-focal segment and drops the
zoomed views, because the renderer currently has one shared intrinsics matrix.
Do not bypass that filter with `--max_focal_drift_frac 1.0` unless per-frame
intrinsics support is added first.
