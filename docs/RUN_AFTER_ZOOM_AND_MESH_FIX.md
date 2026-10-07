# Run after: per-frame intrinsics + automatic mesh export fix (2026-09-21)

## 2026-09-21 CORRECTION -- the first version of this fix was incomplete

The run made with the original version of this fix (20 cameras, `--holdout_fraction 0.2`)
came back WORSE, not better: `[dynamic object filter]` flagged a mean of 79%/max 96.6% of
pixels as "dynamic" (on a scene that is not 80% moving traffic), and the NEW post-training
frustum check added below caught the exported Gaussians outright: opening angle matched
the camera's real FOV to within 4.6 deg -- "the cloud IS the frustum", the same failure
`RUN_WITH_SRT_POSE_FIX.md` describes, back again.

Root cause: `real_dataset_builder.py`/`build_cameras_from_real`/`visual_odometry.py`/
`dynamic_object_filter.py` all got per-frame K that day, but TWO more foundational call
sites did not -- `depth_estimation.py`'s `plane_sweep_depth`/`estimate_depth_for_sequence`
(the actual plane-sweep MVS that PRODUCES depth) and `prototype_point_repr.py`'s
`fuse_point_cloud` (which unprojects that depth into 3-D points) were both still building
ONE ray-direction grid from a single shared K and applying it to every frame. With 20
cameras now spanning the full 24-70mm zoom range instead of 9 stuck at 24mm, most
plane-sweep windows and fusion frames actually crossed a focal-length boundary -- so the
fix that was supposed to recover more frames instead fed most of them through the wrong
projection at the two most upstream steps. Confirmed numerically, not just by re-reading
the code: `tests/test_per_frame_intrinsics.py` builds a synthetic wide+tele camera pair
with a known 3-D point/depth and checks recovery against ground truth; with the single-
shared-K call style it reproduces both symptoms (depth off by ~3x on a 3x focal-length
mismatch; two observations of the same real point fusing to two different, wrong
positions instead of one correct one).

Both are now fixed the same way as the original per-frame-K fixes: `plane_sweep_depth`
takes a `neighbor_Ks` list (each neighbor's own K; the reference frame's own K was
already the lone `K` arg), `estimate_depth_for_sequence` takes `K` as either one shared
Intrinsics or a per-frame list, and `fuse_point_cloud`'s `K` argument does the same.
`real_pipeline.py` now builds one `kept_Ks` list up front and passes it to every consumer
(depth estimation, point fusion, the dynamic filter) instead of each call site deciding
separately -- that inconsistency (some call sites fixed, others not) is exactly how this
happened. Also fixed while in this code: the seed-cloud frustum check's expected-FOV
comparison was silently dead (`Intrinsics` has no `hfov_deg` attribute, so it always
compared against `None`) -- it never actually used its strongest signal until now.

`tests/test_per_frame_intrinsics.py` is a new permanent regression test for this exact
bug class across all three functions -- run it (`pytest tests/test_per_frame_intrinsics.py`)
after touching any camera-intrinsics plumbing in either module. Everything below this
point is the ORIGINAL fix description and is still accurate; only the "recovers more
frames" half of the story was incomplete until today.

### Not fully verified

`fuse_tsdf.py`'s equivalent change (per-view intrinsics list instead of one shared
fx/fy/cx/cy) could NOT be re-verified end-to-end here: `tests/test_fuse_tsdf.py` fails in
this sandbox on a minimal repro that uses zero lines of this project's code (a bare
`ScalableTSDFVolume.integrate()` + `extract_triangle_mesh()` on one flat synthetic depth
plane also returns 0 vertices), which points at an open3d 0.20.0 / this-container build
issue rather than the code change itself -- but I can't rule out a real interaction
without an open3d build that actually works here. Re-run `tests/test_fuse_tsdf.py` in
your own environment (where MESH_QUALITY_AUDIT.md's own work confirms open3d TSDF
integration does work) before trusting `reconstruct_via_tsdf.py`'s output.


Context: the iter8000 run (9 training cameras, `mesh_iter8000.obj` = 1042 disconnected
components / a "shard", see screen recording) was diagnosed and the pipeline patched.
This doc is the same kind of note as `RUN_WITH_SRT_POSE_FIX.md` -- what changed, why, and
the exact command to re-run with it.

## What was actually wrong (two separate problems, not one)

**1. View starvation.** `DJI0004.srt`/`.MP4` is a 14.25s clip (427 frames @ 29.97fps). Its
`focal_len` field shows a smooth 24mm -> 70mm zoom starting at candidate frame 9 (t=6.0s,
~42% into the clip) and continuing to the end. `--max_focal_drift_frac 0.10` correctly
excluded every zoomed frame (this repo assumed one shared camera intrinsics matrix, so
that exclusion was the right call at the time) -- but that leaves only **9 frames spanning
6 seconds / 33m of flight** to reconstruct the whole scene from. That's why training-view
PSNR plateaus around 25 even by iter 8000 (200k Gaussians memorizing 9 views should do much
better than that), why 96.7% of the 681k raw fused points get rejected by the
`obs>=2` confidence gate (most of the scene is only ever seen from 1-2 of 9 viewpoints),
and why the raw checkpoint has points with median distance-from-centroid ~103m despite a
33m flight path (large poorly-constrained far-field tail).

**Fix**: per-frame camera intrinsics now exist (`real_dataset_builder.py`,
`build_cameras_from_real`, `visual_odometry.py`, `dynamic_object_filter.py`,
`fuse_tsdf.py` were all touched -- see diffs). `--max_focal_drift_frac` now defaults to
`1.0`: every candidate frame is kept, each rendered/matched with its OWN focal length
converted to its own fx/fy (anchored to your trusted `--camera_hfov_deg 84` at frame 0,
scaled by the focal-length ratio -- see `real_dataset_builder.py`'s comment for the exact
math). This does **not** model the Air 3's wide/tele cameras being two physically separate
lenses (unknown small baseline offset) -- flagged clearly in the code, and something to
watch for specifically in frames deep into the tele end (candidates ~18-21, 60mm+). If
results look worse than 9-frame, not better, pass `--max_focal_drift_frac 0.1` to fall back
to the old behavior and let VO tell you (via the new `n_cross_zoom_pairs` stat it prints)
how many pairs were actually affected.

**This is a partial fix, not a full fix** -- it recovers up to ~22 candidate frames instead
of 9 from THIS clip, but 22 frames over 14s is still thin for a scene this size. The actual
fix is capturing more: lock focal length (disable zoom) for the whole pass, and fly a
longer/slower single pass, or a pattern with real overlap (see "Capture guidance" below).

**2. The automatic end-of-training mesh export never used the September mesh-quality
fixes.** `docs/MESH_QUALITY_AUDIT.md`'s fixes (auto opacity/confidence thresholds,
covariance-derived normals, floater filtering, debris/component cleanup) were all real,
implemented, and used by the standalone `export_obj.py` -- but `train_gpu.py`'s own
`_export_mesh()` (what actually runs at the end of every training run) called the
underlying function with none of them. Measured directly on the shipped
`mesh_iter8000.obj`: **1042 connected components, 97.7% of vertices in the largest one** --
essentially the audit's own "before" numbers on a different checkpoint, not its "after"
ones. That's the small iridescent shard in the screen recording, not a second instance of
the original frustum/pyramid bug (the frustum check on the seed cloud legitimately passed:
R²=0.81 vs a 0.95 threshold for "cone-shaped").

**Fix**: `_export_mesh()` now does what `export_obj.py` does, inline -- auto thresholds,
covariance normals, floater filtering, debris cleanup (new default
`--mesh_min_component_vertices 20`), and a second frustum-shape check on the trained,
filtered Gaussians (not just the pre-training seed cloud). New/changed flags, all with
working defaults: `--mesh_min_opacity`/`--mesh_min_confidence` now default to `"auto"`
(pass a float to override), `--mesh_use_covariance_normals`, `--mesh_floater_filter`,
`--mesh_floater_anisotropy_below`, `--mesh_floater_scale_percentile`,
`--mesh_min_component_vertices`, `--mesh_keep_top_k_components`, `--mesh_voxel_size`.

## Re-run command

```bash
!python3 layer3_gpu/python/train_gpu.py \
  --source real \
  --video /content/DJI0004.MP4 \
  --log /content/DJI0004.srt \
  --assumed_gimbal_pitch_deg -45 \
  --camera_hfov_deg 84 \
  --target_fps 1.5 --max_frames 120 \
  --max_focal_drift_frac 1.0 \
  --pose_source visual_odometry --vo_min_inliers 20 --vo_max_frame_gap 3 \
  --keep_fraction 1.0 \
  --depth_min 30 --depth_max 250 --n_depths 128 \
  --depth_boundary_reject \
  --holdout_fraction 0.2 \
  --max_iterations 8000 --densify_until_iter 6500 \
  --max_gaussians 200000 \
  --out_dir outputs/srt_zoom_and_mesh_fix_v1
```

Only 3 changes from the command that produced iter8000: `--max_focal_drift_frac 1.0`
(was implicit 0.1 default), `--holdout_fraction 0.2` (was 0, see below), everything mesh-
related is now "auto" by default so no new flags are strictly required there. Watch the
new `[cameras] N/M cameras use their own per-frame intrinsics` line to confirm more than 9
made it through, and the `[mesh export ...]` block's component/debris counts at the end.

**Also add `--holdout_fraction 0.2`** (not part of this fix, but you're re-running anyway):
right now every PSNR/SSIM number in the log is measured on training views with
`holdout_fraction<=0`, i.e. explicitly flagged "memorization risk, not a genuine
generalization check" -- every number in this doc's diagnosis inherits that caveat too.
With ~13-22 cameras instead of 9 there's enough to hold a few out and know whether quality
is real. `--densify_until_iter 6500` was tuned on a 77-camera synthetic-ish run's held-out
curve (see that flag's own comment in `TrainConfig`) -- it is not verified for a 13-22
camera real scene; a real holdout is how you'd find out whether 6500 is still right,
too early, or too late for this scene, rather than assuming either the old default or the
newer 20000-iteration default is correct here.

## Still open (not fixed today -- next in priority order)

1. **Dynamic objects (moving vehicles) on the road.** `use_dynamic_filter=True` already
   runs by default but printed NOTHING -- added instrumentation
   (`[dynamic object filter] ... fraction flagged`) so you can actually see whether it's
   doing anything on this scene instead of guessing. A direct Farneback-flow check on two
   real frames at training resolution (640x480) shows the single highest-motion region in
   the whole frame is a tight band tracing the road -- consistent with real vehicle motion,
   though a nearer/shallower-angle road surface also legitimately gets more parallax flow
   than distant buildings, so this isn't proof by itself. Check the new instrumentation
   output first; if `max` flagged-fraction is near 0 on a run with visible traffic, the
   classical filter likely isn't catching it at this resolution (STOP_CONDITIONS.md's own
   "genuinely open, untested" item 3) and is worth a closer look (spot-check a `.mask`
   against its frame) before concluding the road-area error in the render comparisons is
   purely a coverage/geometry issue.
2. **TSDF multi-view-consistent meshing** (`reconstruct_via_tsdf.py`) -- `_build_cameras`
   was a `NotImplementedError` stub; it's now wired to rebuild the same camera list
   `train_gpu.py` would from the same flags (see that script's new argparse block, same
   flag names as `train_gpu.py`). `fuse_tsdf.py` also only accepted ONE shared fx/fy/cx/cy
   for every view, which would have silently mis-fused any frame using its own per-frame K
   from fix #1 above -- it now accepts either a scalar or a per-view list, and
   `reconstruct_via_tsdf.py` passes the real per-view lists. This is genuinely a different
   algorithm from Poisson-on-points (leaves unobserved regions as holes instead of
   interpolating through them), specifically suited to a sparse single-pass capture per
   `docs/MESH_QUALITY_AUDIT.md` -- run it against the new checkpoint and compare to the
   Poisson mesh once you have GPU time:
   ```bash
   python3 layer3_gpu/python/reconstruct_via_tsdf.py \
     --checkpoint outputs/srt_zoom_and_mesh_fix_v1/full_state_iter8000.pt \
     --out outputs/srt_zoom_and_mesh_fix_v1/mesh_tsdf.obj \
     --video /content/DJI0004.MP4 --log /content/DJI0004.srt \
     --assumed_gimbal_pitch_deg -45 --camera_hfov_deg 84 \
     --target_fps 1.5 --max_frames 120 --max_focal_drift_frac 1.0 \
     --pose_source visual_odometry --vo_min_inliers 20 --vo_max_frame_gap 3 \
     --depth_min 30 --depth_max 250 --n_depths 128 --depth_boundary_reject \
     --voxel_size 0.2
   ```
   (untested against real data -- needs Environment B / a real CUDA device; sanity-check
   a couple of its depth maps, e.g. via `--save_depth_dir`, before trusting the mesh.)
3. **Bundle-adjust the VO poses.** Recommended in `docs/MESH_QUALITY_AUDIT.md`, not built.
   VO currently chains consecutive pairwise poses (each pair independent); a small
   reprojection-error refinement across all kept frames + seed points jointly (e.g.
   `scipy.optimize.least_squares` over the existing VO poses as initialization) is the
   next concrete step if quality with more frames + a real holdout still isn't where you
   need it, but wasn't built today -- it touches pose accuracy which this session
   couldn't verify without a held-out view (see `--holdout_fraction` above), and a
   half-verified bundle adjustment is worse than none.
4. **Depth-consistency loss term.** Also recommended, not built -- pull Gaussian depth
   toward `depth_estimation.py`'s own high-confidence plane-sweep depth during
   photometric training. Same reasoning as #3 for why it's scoped as "next", not done now.

## Capture guidance for your next flight (the actual ceiling on all of the above)

14 seconds of usable footage, most of it lost to a mid-clip zoom, is the hard limit under
every fix above. For the next single-pass capture: **lock focal length / disable zoom for
the whole pass** (removes problem #1 above at the source), and fly long enough / slow
enough that a 1.5-2fps sample gives you dozens of frames with real overlap, not ~10-20 --
this project's own `densify_until_iter` comment in `train_gpu.py` references a prior
77-camera run for scale. A steeper gimbal angle (closer to nadir) than -45° was also
`docs/MESH_QUALITY_AUDIT.md`'s capture recommendation, if the mission profile allows it,
to shrink the depth range the scene actually spans.
