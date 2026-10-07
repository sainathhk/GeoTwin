"""
reconstruct_via_tsdf.py
LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, NOT executed (see
render_multiview_depth.py's docstring -- needs Environment B). This is the CLI
for docs/MESH_QUALITY_AUDIT.md's recommended "gold standard" path: render
depth+alpha from the TRAINED Gaussians across many views, fuse via TSDF
(respects visibility -- leaves genuinely unobserved regions as holes instead
of Poisson's closed-surface interpolation), extract a mesh.

This is a SEPARATE, complementary output to export_obj.py's Poisson/ball-
pivoting path, not a replacement for it -- Poisson-on-Gaussian-centers is
cheap (CPU, seconds) and a reasonable first look; this is more expensive
(needs the GPU rasterizer for every rendered view) and, for a single-pass
aerial scene specifically, more likely to be geometrically correct BECAUSE it
respects what was and wasn't actually observed. Run both; compare.

Usage:
    python3 layer3_gpu/python/reconstruct_via_tsdf.py \
        --checkpoint outputs/real_single_pass_v2/full_state_iter8000.pt \
        --out outputs/real_single_pass_v2/mesh_tsdf.obj \
        --every_nth_camera 2 --voxel_size 0.2

Needs the SAME camera list training used (poses + intrinsics + which frames)
-- this script does not re-derive poses, it expects whatever `train_gpu.py`
built cameras from (synthetic or real) to be reconstructible the same way;
see --cameras_from below.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train_gpu import (
    load_full_state, build_cameras_from_real, build_cameras_from_synthetic,
    filter_stable_focal_length, filter_downward_mapping_frames,
)
from render_multiview_depth import render_multiview_depth
from fuse_tsdf import fuse_depth_maps_tsdf
from layer1_cpu_sandbox.reconstruction.mesh_export import save_obj, save_glb


def _build_cameras(args):
    """Rebuilds the same TrainingCamera list train_gpu.py's main() would have used for
    this run, so the checkpoint's Gaussians can be re-rendered from the SAME poses/
    intrinsics they were trained with. 2026-09: this used to raise NotImplementedError --
    it now mirrors train_gpu.py's main() call-for-call (dataset build -> focal/orientation
    filtering -> optional visual odometry -> RealPipelineConfig/run_real_pipeline ->
    build_cameras_from_real), driven by the SAME flag names as train_gpu.py so you can
    reuse the exact command line you trained with, just swapped onto this script. If you
    trained with --use_agents (Frame Agent picking keep_fraction/min_overall), pass the
    same values here explicitly via --keep_fraction/--min_overall -- the agent decision
    itself isn't re-run, since frame_agent_decision.json next to the checkpoint already
    records what it chose.
    """
    if args.source == "synthetic":
        from layer1_cpu_sandbox.synthetic.dataset_builder import build_dataset
        from layer1_cpu_sandbox.pipeline import run_pipeline, PipelineConfig
        dataset = build_dataset(seed=args.seed)
        result = run_pipeline(dataset, PipelineConfig(voxel_size=0.6, min_consistency=0.10,
                                                        keep_fraction=args.keep_fraction,
                                                        min_overall=args.min_overall))
        return build_cameras_from_synthetic(dataset, result)

    assert args.video and args.log, "--video and --log are required for --source real"
    from layer1_cpu_sandbox.real_data.real_dataset_builder import build_real_dataset
    from layer1_cpu_sandbox.real_data.real_pipeline import run_real_pipeline, RealPipelineConfig

    dataset = build_real_dataset(args.video, args.log, camera_hfov_deg=args.camera_hfov_deg,
                                  target_fps=args.target_fps, max_frames=args.max_frames,
                                  resize_to=(args.width, args.height),
                                  position_smoothing_window=args.position_smoothing_window,
                                  assumed_gimbal_pitch_deg=args.assumed_gimbal_pitch_deg,
                                  assumed_gimbal_roll_deg=args.assumed_gimbal_roll_deg)
    dataset, rejected_zoom = filter_stable_focal_length(dataset, args.max_focal_drift_frac)
    if rejected_zoom:
        print(f"Excluded {len(rejected_zoom)} frames outside the longest stable-focal-length run: "
              f"{rejected_zoom}")
    if args.pose_source == "visual_odometry":
        from layer1_cpu_sandbox.real_data.visual_odometry import VisualOdometryError, replace_with_visual_odometry
        try:
            dataset, vo_stats = replace_with_visual_odometry(
                dataset, min_matches=args.vo_min_matches, min_inliers=args.vo_min_inliers,
                max_frame_gap=args.vo_max_frame_gap)
        except VisualOdometryError as e:
            raise RuntimeError(
                "Visual odometry refused to create poses from this clip -- pass the same "
                "--pose_source/--vo_* flags you trained with.") from e
        print(f"[visual odometry] {vo_stats.n_pairs} pairs, {vo_stats.total_inliers} total inliers, "
              f"retained {vo_stats.n_frames}/{vo_stats.n_frames_input} frames")
    dataset, rejected_orient = filter_downward_mapping_frames(dataset, args.max_forward_z)
    if rejected_orient:
        print(f"Excluded {len(rejected_orient)} non-downward frames: {rejected_orient}")

    result = run_real_pipeline(dataset, RealPipelineConfig(
        voxel_size=0.6, min_consistency=0.10, keep_fraction=args.keep_fraction,
        min_overall=args.min_overall, depth_window=args.depth_window, n_depths=args.n_depths,
        depth_min=args.depth_min, depth_max=args.depth_max, depth_spacing=args.depth_spacing,
        depth_boundary_reject=args.depth_boundary_reject))
    return build_cameras_from_real(dataset, result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", help="needs a real CUDA device -- unlike export_obj.py, "
                                                       "this script calls the Gaussian rasterizer.")
    # --- dataset/camera-building args: SAME names/defaults as train_gpu.py's main(), so you can
    # reuse the exact --video/--log/--assumed_gimbal_pitch_deg/etc. you trained this checkpoint
    # with. See train_gpu.py's own argparse block for the full help text on each of these --
    # kept in sync deliberately, not duplicated by accident.
    ap.add_argument("--source", choices=["synthetic", "real"], default="real")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--target_fps", type=float, default=2.0)
    ap.add_argument("--camera_hfov_deg", type=float, default=84.0)
    ap.add_argument("--position_smoothing_window", type=int, default=5)
    ap.add_argument("--max_forward_z", type=float, default=-0.5)
    ap.add_argument("--assumed_gimbal_pitch_deg", type=float, default=None,
                    help="REQUIRED when --log is a .srt file -- see train_gpu.py.")
    ap.add_argument("--assumed_gimbal_roll_deg", type=float, default=0.0)
    ap.add_argument("--max_focal_drift_frac", type=float, default=1.0,
                    help="Default matches train_gpu.py's new default: 1.0 (keep every frame, each with "
                         "its own per-frame K) now that per-frame intrinsics exist. Pass the SAME value "
                         "you trained with if you overrode it.")
    ap.add_argument("--pose_source", choices=["telemetry", "visual_odometry"], default="telemetry")
    ap.add_argument("--vo_min_matches", type=int, default=80)
    ap.add_argument("--vo_min_inliers", type=int, default=40)
    ap.add_argument("--vo_max_frame_gap", type=int, default=3)
    ap.add_argument("--keep_fraction", type=float, default=1.0)
    ap.add_argument("--min_overall", type=float, default=0.15)
    ap.add_argument("--depth_window", type=int, default=2)
    ap.add_argument("--n_depths", type=int, default=128)
    ap.add_argument("--depth_min", type=float, default=3.0)
    ap.add_argument("--depth_max", type=float, default=100.0)
    ap.add_argument("--depth_spacing", choices=["log", "linear"], default="log")
    ap.add_argument("--depth_boundary_reject", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--every_nth_camera", type=int, default=1,
                     help="use every Nth training camera instead of all of them -- cheaper, still plenty "
                          "of overlap for TSDF fusion on a single-pass flight's dense frame spacing.")
    ap.add_argument("--voxel_size", type=float, default=0.2,
                     help="TSDF voxel size, world units -- see fuse_tsdf.py docstring for how to pick this "
                          "(start near this checkpoint's median Gaussian scale).")
    ap.add_argument("--depth_trunc", type=float, default=200.0)
    ap.add_argument("--alpha_valid_thresh", type=float, default=0.05)
    ap.add_argument("--min_alpha_for_fusion", type=float, default=0.5)
    ap.add_argument("--save_depth_dir", default=None,
                     help="optional -- dump per-view depth/alpha/color .npy + manifest.json here for debugging")
    ap.add_argument("--glb", action="store_true")
    args = ap.parse_args()

    model = load_full_state(args.checkpoint, device=args.device)
    cameras = _build_cameras(args)
    cameras = cameras[::args.every_nth_camera]
    print(f"rendering depth+alpha+color from {len(cameras)} cameras (every {args.every_nth_camera})...")

    views = render_multiview_depth(model, cameras, args.device, save_dir=args.save_depth_dir,
                                    alpha_valid_thresh=args.alpha_valid_thresh)
    coverage = [float((v.alpha > args.alpha_valid_thresh).mean()) for v in views]
    print(f"  per-view valid-pixel fraction: min={min(coverage):.2f} mean={sum(coverage)/len(coverage):.2f} "
          f"max={max(coverage):.2f}")

    mesh, stats = fuse_depth_maps_tsdf(
        [v.depth for v in views], [v.color for v in views], [v.R_wc for v in views],
        [v.position for v in views], [v.fx for v in views], [v.fy for v in views],
        [v.cx for v in views], [v.cy for v in views],
        alpha_maps=[v.alpha for v in views], min_alpha=args.min_alpha_for_fusion,
        voxel_size=args.voxel_size, depth_trunc=args.depth_trunc)

    print(f"TSDF fusion: {stats.n_views_integrated} views integrated, {stats.n_views_skipped_empty} skipped "
          f"(no valid depth) -> {stats.n_vertices} vertices, {stats.n_triangles} triangles, "
          f"bbox {stats.bbox_min} .. {stats.bbox_max}")

    save_obj(mesh, args.out)
    print(f"wrote {args.out}")
    if args.glb:
        glb_path = args.out.rsplit(".", 1)[0] + ".glb"
        save_glb(mesh, glb_path)
        print(f"wrote {glb_path}")


if __name__ == "__main__":
    main()
