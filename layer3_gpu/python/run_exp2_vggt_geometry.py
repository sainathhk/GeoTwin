#!/usr/bin/env python3
"""Experiment 2: VGGT geometry-first reconstruction from a DJI video + SRT.

This deliberately stops before Gaussian training. It exports a single shared
VGGT point map, metric-scales it using the SRT GPS path length, and writes a
diagnostic preview. Inspect that preview before running build_exp2_tsdf_mesh.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--log", required=True, help="Matching DJI SRT telemetry")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--target_fps", type=float, default=4.0,
                   help="Candidate sampling before choosing evenly spaced frames from one focal block.")
    p.add_argument("--max_frames", type=int, default=18,
                   help="Keep one global VGGT pass small enough for a Colab T4 (default 18).")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera_hfov_deg", type=float, default=84.0)
    p.add_argument("--assumed_gimbal_pitch_deg", type=float, default=-45.0,
                   help="Only used to build telemetry; VGGT estimates visual camera orientation.")
    p.add_argument("--max_focal_drift_frac", type=float, default=0.10,
                   help="Use one contiguous focal-length block to avoid mixing DJI camera modules.")
    p.add_argument("--confidence_threshold", type=float, default=1.5)
    p.add_argument("--top_mask_fraction", type=float, default=0.08,
                   help="Trim this fraction of the active image at the top to suppress sky/haze.")
    p.add_argument("--voxel_size_m", type=float, default=0.40)
    p.add_argument("--max_depth_m", type=float, default=250.0)
    p.add_argument("--model_name", default="facebook/VGGT-1B")
    return p.parse_args()


def _camera_centers(w2c: np.ndarray) -> np.ndarray:
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)


def _input_valid_mask(width: int, height: int, size: int = 518,
                      top_mask_fraction: float = 0.08) -> np.ndarray:
    # Mirrors VGGT's official load_and_preprocess_images(..., mode="pad"):
    # resize longest side to 518, round short side to a multiple of 14, center-pad.
    if width >= height:
        new_w = size
        new_h = round((height * (new_w / width)) / 14) * 14
    else:
        new_h = size
        new_w = round((width * (new_h / height)) / 14) * 14
    new_h, new_w = min(size, new_h), min(size, new_w)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    mask = np.zeros((size, size), dtype=bool)
    crop_top = int(round(new_h * float(top_mask_fraction)))
    mask[pad_top + crop_top:pad_top + new_h, pad_left:pad_left + new_w] = True
    return mask


def _save_preview(path: Path, points: np.ndarray, colors: np.ndarray,
                  cameras: np.ndarray, voxel_size: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    if len(points) > 180_000:
        keep = rng.choice(len(points), 180_000, replace=False)
        plot_points, plot_colors = points[keep], colors[keep]
    else:
        plot_points, plot_colors = points, colors
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [(0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z")]
    for ax, (a, b, label) in zip(axes, pairs):
        ax.scatter(plot_points[:, a], plot_points[:, b], c=plot_colors,
                   s=0.12, linewidths=0, rasterized=True)
        ax.plot(cameras[:, a], cameras[:, b], color="red", marker="o", markersize=3,
                linewidth=1.2, label="VGGT camera path")
        ax.set_xlabel("world " + "XYZ"[a] + " (m)")
        ax.set_ylabel("world " + "XYZ"[b] + " (m)")
        ax.set_title(label + " projection")
        ax.axis("equal")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best")
    fig.suptitle("Experiment 2 VGGT shared point map — inspect before meshing")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if not (0 <= args.top_mask_fraction < 0.35):
        raise SystemExit("--top_mask_fraction must be in [0, 0.35).")
    if args.max_frames < 4:
        raise SystemExit("Use at least 4 frames for multi-view geometry.")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    out = Path(args.out_dir).resolve()
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    from layer1_cpu_sandbox.real_data.real_dataset_builder import build_real_dataset
    from layer1_cpu_sandbox.real_data.dji_log_parser import filter_stable_focal_length_frames

    print("[1/5] Extracting synchronized video frames and matching SRT telemetry...")
    dataset = build_real_dataset(
        video_path=args.video, log_path=args.log,
        camera_hfov_deg=args.camera_hfov_deg,
        target_fps=args.target_fps, max_frames=240,
        resize_to=(args.width, args.height), position_smoothing_window=1,
        assumed_gimbal_pitch_deg=args.assumed_gimbal_pitch_deg,
    )
    kept, rejected = filter_stable_focal_length_frames(dataset, args.max_focal_drift_frac)
    stable_count = len(kept)
    if stable_count > args.max_frames:
        selected_ids = np.linspace(0, stable_count - 1, args.max_frames).round().astype(int)
        kept = [kept[int(i)] for i in selected_ids]
    if len(kept) < 6:
        raise SystemExit(
            f"Only {len(kept)} frames remain in the stable focal block (from {len(dataset.frames)}). "
            "Need at least 6 for a useful shared map. Increase --max_frames / change sampling, "
            "but do not merge wide and tele camera modules."
        )
    if len(kept) > 28:
        raise SystemExit(f"{len(kept)} selected frames may exceed T4 memory. Lower --max_frames to 18-24.")
    print(f"Frames: {len(dataset.frames)} candidate frames; {stable_count} in one stable focal block; "
          f"{len(rejected)} excluded for focal drift; {len(kept)} evenly spaced frames sent to VGGT.")
    if kept[0].focal_len_mm is not None:
        focals = [float(f.focal_len_mm) for f in kept]
        print(f"Focal block: {min(focals):.2f}–{max(focals):.2f} mm (35 mm equivalent).")

    image_paths = []
    gps = []
    timestamps = []
    frame_indices = []
    for i, frame in enumerate(kept):
        path = frames_dir / f"frame_{i:04d}.jpg"
        Image.fromarray(frame.rgb.astype(np.uint8)).save(path, quality=95)
        image_paths.append(str(path))
        gps.append(np.asarray(frame.pose.position, dtype=np.float64))
        timestamps.append(float(frame.t))
        frame_indices.append(int(frame.idx))

    print("[2/5] Loading VGGT and estimating one joint camera/depth solution...")
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required. In Colab choose Runtime > Change runtime type > T4 GPU.")
    device = "cuda"
    major, _ = torch.cuda.get_device_capability()
    amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}; mixed precision: {amp_dtype}.")
    images = load_and_preprocess_images(image_paths, mode="pad")
    if images.ndim != 4:
        raise RuntimeError(f"VGGT loader returned unexpected image tensor shape {tuple(images.shape)}")
    valid_pixel_mask = _input_valid_mask(args.width, args.height,
                                         size=int(images.shape[-1]),
                                         top_mask_fraction=args.top_mask_fraction)
    model = VGGT.from_pretrained(args.model_name).to(device).eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            pred = model(images.to(device))
            pose_enc = pred["pose_enc"]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            w2c = extrinsics[0].float().cpu().numpy().astype(np.float32)
            K = intrinsics[0].float().cpu().numpy().astype(np.float32)
            depth = pred["depth"][0].float().cpu().numpy()
            depth_conf = pred["depth_conf"][0].float().cpu().numpy()
            if depth.ndim == 4 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
                depth_conf = depth_conf[..., 0]
            image_rgb = (images.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            if "world_points" in pred:
                world_points = pred["world_points"][0].float().cpu().numpy()
                world_conf = pred.get("world_points_conf", pred["depth_conf"])[0].float().cpu().numpy()
                if world_conf.ndim == 4 and world_conf.shape[-1] == 1:
                    world_conf = world_conf[..., 0]
            else:
                from vggt.utils.geometry import unproject_depth_map_to_point_map
                world_points = unproject_depth_map_to_point_map(depth, w2c, K)
                world_conf = depth_conf
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise SystemExit(
            "VGGT ran out of GPU memory. Rerun with --max_frames 10 or 12; keep all frames in "
            "one pass so their world coordinates remain shared."
        ) from exc
    finally:
        del model
        if "pred" in locals():
            del pred
        torch.cuda.empty_cache()

    n, h, w = depth.shape
    if world_points.shape[:3] != (n, h, w):
        raise RuntimeError(f"Predicted point map shape {world_points.shape} does not match depth {(n,h,w)}")
    if K.shape[0] != n or w2c.shape[0] != n:
        raise RuntimeError("VGGT camera count differs from sampled frame count.")
    mask2d = valid_pixel_mask[:h, :w]
    if mask2d.shape != (h, w):
        raise RuntimeError(f"Image mask shape {mask2d.shape} differs from prediction {(h,w)}")

    print("[3/5] Aligning only the global scale to the SRT GPS path length...")
    camera_centers = _camera_centers(w2c)
    vggt_path = float(np.linalg.norm(np.diff(camera_centers, axis=0), axis=1).sum())
    gps_positions = np.asarray(gps, dtype=np.float64)
    gps_path = float(np.linalg.norm(np.diff(gps_positions, axis=0), axis=1).sum())
    if not np.isfinite(vggt_path) or vggt_path < 1e-5:
        raise SystemExit("VGGT camera trajectory has near-zero length; geometry is not usable.")
    if not np.isfinite(gps_path) or gps_path < 1e-3:
        raise SystemExit("SRT GPS path length is near zero; cannot set metric scale.")
    scale = gps_path / vggt_path
    if scale < 0.01 or scale > 100.0:
        warnings.warn(f"GPS/VGGT path scale factor {scale:.4g} is extreme; metric scale is suspect.")
    origin = camera_centers[0].copy()
    world_points_m = (world_points.astype(np.float32) - origin[None, None, None, :]) * np.float32(scale)
    cameras_m = (camera_centers - origin[None, :]) * scale
    w2c_m = w2c.copy()
    w2c_m[:, :3, 3] = (w2c[:, :3, 3] + np.einsum("nij,j->ni", w2c[:, :3, :3], origin)) * scale
    depth_m = depth.astype(np.float32) * np.float32(scale)

    print("[4/5] Exporting confidence-filtered point map, camera path, cache, and preview...")
    rgb_flat = image_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    points_flat = world_points_m.reshape(-1, 3)
    conf_flat = np.asarray(world_conf).reshape(-1)
    depth_conf_flat = depth_conf.reshape(-1)
    valid_flat = np.tile(mask2d.reshape(-1), n)
    usable = (valid_flat & np.isfinite(points_flat).all(axis=1) & np.isfinite(conf_flat) &
              (conf_flat >= args.confidence_threshold) & np.isfinite(depth_conf_flat) &
              (depth_conf_flat >= args.confidence_threshold) &
              np.isfinite(depth_m.reshape(-1)) & (depth_m.reshape(-1) > 0) &
              (depth_m.reshape(-1) <= args.max_depth_m))
    points = points_flat[usable]
    colors = rgb_flat[usable]
    if len(points) < 5000:
        raise SystemExit(f"Only {len(points)} points survived confidence/range filters; no mesh should be built.")
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    cloud = cloud.voxel_down_sample(voxel_size=float(args.voxel_size_m))
    cloud_path = out / "vggt_geometry_metric.ply"
    o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False, compressed=True)

    np.savez_compressed(
        out / "vggt_depth_cache.npz",
        depth_m=depth_m.astype(np.float32), depth_conf=depth_conf.astype(np.float16),
        rgb=image_rgb, valid_mask=np.broadcast_to(mask2d, (n, h, w)).astype(np.uint8),
        w2c_m=w2c_m.astype(np.float32), K=K.astype(np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64), frame_indices=np.asarray(frame_indices),
        gps_positions=gps_positions.astype(np.float64), gps_path_m=np.float64(gps_path),
        vggt_path_before_scale=np.float64(vggt_path), metric_scale=np.float64(scale),
        confidence_threshold=np.float32(args.confidence_threshold),
        max_depth_m=np.float32(args.max_depth_m),
    )
    preview_path = out / "exp2_geometry_preview.png"
    cloud_np = np.asarray(cloud.points)
    colors_np = np.asarray(cloud.colors)
    _save_preview(preview_path, cloud_np, colors_np, cameras_m, args.voxel_size_m)

    mins, maxs = points.min(axis=0), points.max(axis=0)
    summary = {
        "experiment": "exp2_vggt_geometry_first",
        "frames_sampled": len(dataset.frames), "frames_used": n,
        "frames_rejected_focal_drift": len(rejected),
        "focal_mm_range": ([float(min(f.focal_len_mm for f in kept)), float(max(f.focal_len_mm for f in kept))]
                           if kept[0].focal_len_mm is not None else None),
        "srt_gps_path_m": gps_path, "vggt_path_before_scale": vggt_path,
        "metric_scale_factor": scale, "point_count_confidence_filtered": int(len(points)),
        "point_count_voxelized": int(len(cloud.points)), "bbox_min_m": mins.tolist(),
        "bbox_max_m": maxs.tolist(), "point_cloud": str(cloud_path),
        "depth_cache": str(out / "vggt_depth_cache.npz"), "preview": str(preview_path),
        "warnings": ["GPS path sets approximate scale only; no geographic north/origin alignment.",
                     "Do not interpret this as a validated twin until the shared point map is visually coherent."],
    }
    (out / "exp2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Metric scale factor: {scale:.6g} m per VGGT unit")
    print(f"GPS path: {gps_path:.2f} m; predicted path before scaling: {vggt_path:.4f}")
    print(f"Points: {len(points):,} filtered -> {len(cloud.points):,} voxelized")
    print(f"Point cloud: {cloud_path}\nPreview: {preview_path}")
    print("STOP HERE and inspect the preview. Do not build a mesh if roads/building masses do not form a coherent shared scene.")


if __name__ == "__main__":
    main()
