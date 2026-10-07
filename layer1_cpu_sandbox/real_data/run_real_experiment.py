"""
run_real_experiment.py
REAL-DATA INGESTION -- STATUS: implemented, CPU-tested against synthetic mock
video+log pairs. NOT yet run against real downloaded footage in this
environment -- run this yourself once you have the files (e.g. in Colab).

Usage (matches the exact files from Zenodo record 3604005 the way you'd
have them after your two `wget` commands):

    python3 -m layer1_cpu_sandbox.real_data.run_real_experiment \\
        --video flight.mov --log flight.csv --out_dir outputs/real_run1

See docs/GETTING_STARTED.md "Real data" section for the full walkthrough,
including WHY DJI_0013.MOV specifically (stationary hover, gimbal-pitch-only
motion) is a poor choice for testing reconstruction QUALITY even though it's
fine for testing the ingestion PLUMBING -- use DJI_0004.MOV (the orbit) from
the same record for an actual multi-view test.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .real_dataset_builder import build_real_dataset
from .real_pipeline import run_real_pipeline, evaluate_real_result, RealPipelineConfig
from ..reconstruction.point_renderer import render_point_cloud
from ..reconstruction.mesh_export import export_point_cloud_to_obj, MeshExportError
from ..utils import save_ply, ensure_dir


def plot_real_qualitative(dataset, result, frame_idx: int, out_path: str):
    frame = dataset.frames[frame_idx]
    actual = frame.rgb
    pose = frame.pose
    # That frame's own K when it has one (zoom lens), else dataset.K -- same fix as
    # real_pipeline.py's evaluate_real_result, 2026-09-21.
    frame_K = getattr(frame, "K", None) or dataset.K
    rendered, mask = render_point_cloud(result.final_positions, result.final_colors, frame_K, pose)
    diff = np.abs(rendered.astype(np.float32) - actual.astype(np.float32)).mean(axis=-1)
    diff = np.where(mask, diff, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(actual); axes[0].set_title(f"Actual captured frame #{frame_idx}"); axes[0].axis("off")
    axes[1].imshow(rendered); axes[1].set_title(f"Reconstruction re-rendered from same pose\n({mask.mean()*100:.1f}% pixels covered)")
    axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="inferno", vmin=0, vmax=80)
    axes[2].set_title("Abs. error (self-consistency, not GT)"); axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("REAL DATA -- no ground truth. This checks self-consistency only.")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="layer1_cpu_sandbox/outputs/real_run1")
    ap.add_argument("--hfov", type=float, default=84.0, help="assumed horizontal FOV, deg (DJI Phantom 4 Pro spec)")
    ap.add_argument("--target_fps", type=float, default=2.0)
    ap.add_argument("--max_frames", type=int, default=80)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    t_start = time.perf_counter()

    print(f"[1/4] Loading video + flight log...")
    dataset = build_real_dataset(args.video, args.log, camera_hfov_deg=args.hfov, target_fps=args.target_fps,
                                  max_frames=args.max_frames, resize_to=(args.width, args.height))
    print(f"      video: {dataset.video_info.width}x{dataset.video_info.height} @ {dataset.video_info.fps:.1f}fps, "
          f"{dataset.video_info.duration_s:.1f}s -- log format: {dataset.log_format_detected}")
    print(f"      extracted {len(dataset.frames)} frames, matched to log offset {dataset.matched_log_offset_s:.1f}s")
    print(f"      camera intrinsics: {'ASSUMED ' + str(args.hfov) + ' deg HFOV (not calibrated)' if dataset.K_is_assumed else 'calibrated'}")
    xs = [f.pose.position[0] for f in dataset.frames]
    ys = [f.pose.position[1] for f in dataset.frames]
    span = max(np.ptp(xs), np.ptp(ys))
    print(f"      flight footprint span: ~{span:.1f} m -- "
          f"{'looks like real translation (good for MVS)' if span > 5 else 'WARNING: very little translation detected -- likely a hover/rotation-only clip, multi-view triangulation will be weak or fail (see run_real_experiment.py docstring)'}")

    print("[2/4] Running pipeline (frame quality -> depth -> dynamic filter -> fusion -> confidence)...")
    result = run_real_pipeline(dataset, RealPipelineConfig())
    print(f"      kept {len(result.kept_frame_indices)}/{len(dataset.frames)} frames, "
          f"{result.final_positions.shape[0]} final points after confidence gating")

    print("[3/4] Self-consistency evaluation (NO ground truth exists for real footage)...")
    ev = evaluate_real_result(dataset, result)
    for v in ev.self_consistency_views:
        vf = v["visual"]
        lpips_str = "n/a (needs `lpips` pkg)" if np.isnan(vf.lpips) else f"{vf.lpips:.3f}"
        print(f"      frame {v['frame_idx']}: coverage={v['coverage_frac']*100:.1f}%  PSNR={vf.psnr:.2f}  "
              f"SSIM={vf.ssim:.3f}  LPIPS={lpips_str}")
    print(f"      confidence: mean={ev.confidence_mean:.3f}  bands={ev.confidence_band_counts}")

    print("[4/4] Saving outputs...")
    save_ply(os.path.join(args.out_dir, "reconstruction_real.ply"), result.final_positions, result.final_colors)
    mid_frame = result.kept_frame_indices[len(result.kept_frame_indices) // 2]
    plot_real_qualitative(dataset, result, mid_frame, os.path.join(args.out_dir, "real_qualitative.png"))

    # See layer1_cpu_sandbox/reconstruction/mesh_export.py. Real single-pass drone
    # footage is a much rougher point cloud than the synthetic dataset (fewer views,
    # real depth noise) -- more likely than the synthetic path to genuinely not have
    # enough confident points for a mesh yet, hence the explicit try/except+message
    # here rather than assuming it will always succeed.
    mesh_stats = None
    try:
        mesh_stats = export_point_cloud_to_obj(
            os.path.join(args.out_dir, "reconstruction_real.obj"), result.final_positions, result.final_colors)
        print(f"      mesh: {mesh_stats.n_vertices} vertices, {mesh_stats.n_triangles} triangles "
              f"from {mesh_stats.n_points_after_filter} points -> reconstruction_real.obj")
    except MeshExportError as e:
        print(f"      WARNING: mesh export skipped ({e}); reconstruction_real.ply is still available.")

    def _nan_safe(x):
        return None if isinstance(x, float) and np.isnan(x) else x

    summary = {
        "video_path": args.video, "log_path": args.log,
        "log_format_detected": dataset.log_format_detected,
        "video_duration_s": dataset.video_info.duration_s,
        "flight_footprint_span_m": float(span),
        "camera_intrinsics_assumed": dataset.K_is_assumed,
        "assumed_hfov_deg": args.hfov,
        "n_frames_extracted": len(dataset.frames),
        "n_frames_kept": len(result.kept_frame_indices),
        "n_final_points": ev.n_final_points,
        "mesh_n_vertices": mesh_stats.n_vertices if mesh_stats else None,
        "mesh_n_triangles": mesh_stats.n_triangles if mesh_stats else None,
        "confidence_mean": ev.confidence_mean,
        "confidence_bands": ev.confidence_band_counts,
        "self_consistency_views": [
            {"frame_idx": v["frame_idx"], "coverage_frac": v["coverage_frac"], "psnr": _nan_safe(v["visual"].psnr),
             "ssim": _nan_safe(v["visual"].ssim), "lpips": _nan_safe(v["visual"].lpips)}
            for v in ev.self_consistency_views],
        "total_wall_time_s": round(time.perf_counter() - t_start, 2),
        "NOTE": ev.NOTE,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s. Artifacts written to: {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
