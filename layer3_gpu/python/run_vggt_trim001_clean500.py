#!/usr/bin/env python3
"""Run the validated VGGT -> confidence fusion -> Trim001/Clean500 mesh pass.

This is called automatically after a successful real-data Gaussian training run.
It uses one joint VGGT inference over at most 28 evenly spaced frames so all
predicted views share a single coordinate frame, aligns the camera path scale
to the DJI SRT/CSV telemetry, fuses and confidence-gates the depth points, then
exports Poisson depth 12 with density trim 0.01 and a 500-vertex component floor.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--log", required=True, help="Matching DJI SRT or CSV telemetry")
    p.add_argument("--out_dir", required=True, help="The Gaussian training run's output folder")
    p.add_argument("--target_fps", type=float, default=1.5)
    p.add_argument("--max_candidate_frames", type=int, default=120)
    p.add_argument("--max_vggt_frames", type=int, default=28,
                   help="VGGT views in one shared-coordinate inference; keep <=28 for Colab T4")
    p.add_argument("--width", type=int, default=518)
    p.add_argument("--height", type=int, default=392)
    p.add_argument("--camera_hfov_deg", type=float, default=84.0)
    p.add_argument("--assumed_gimbal_pitch_deg", type=float, required=True)
    p.add_argument("--voxel_size", type=float, default=0.4)
    p.add_argument("--poisson_depth", type=int, default=12)
    p.add_argument("--density_trim_quantile", type=float, default=0.01)
    p.add_argument("--min_component_vertices", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    if args.max_vggt_frames < 6 or args.max_vggt_frames > 40:
        raise SystemExit("--max_vggt_frames must be between 6 and 40; 28 is the Colab T4 default.")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    out = Path(args.out_dir).resolve() / "vggt_trim001_clean500"
    out.mkdir(parents=True, exist_ok=True)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("VGGT postprocess needs the Colab GPU. Select Runtime > Change runtime type > T4 GPU.")

    from layer1_cpu_sandbox.real_data.real_dataset_builder import build_real_dataset
    from layer1_cpu_sandbox.real_data.vggt_pose_depth_source import (
        estimate_poses_vggt_with_conf, build_pose_depth_source)
    from layer1_cpu_sandbox.reconstruction.prototype_point_repr import fuse_point_cloud
    from layer1_cpu_sandbox.reconstruction.confidence import compute_confidence, gate_by_confidence

    print("\n=== Automatic VGGT geometry + Trim001/Clean500 postprocess ===", flush=True)
    print("[1/4] Extracting the same real clip and matching telemetry...", flush=True)
    dataset = build_real_dataset(
        args.video, args.log, camera_hfov_deg=args.camera_hfov_deg,
        target_fps=args.target_fps, max_frames=args.max_candidate_frames,
        resize_to=(args.width, args.height), assumed_gimbal_pitch_deg=args.assumed_gimbal_pitch_deg)
    candidates = dataset.frames
    if len(candidates) < 6:
        raise SystemExit(f"Only {len(candidates)} video frames were available; at least 6 are required for VGGT fusion.")
    if len(candidates) > args.max_vggt_frames:
        chosen = np.linspace(0, len(candidates) - 1, args.max_vggt_frames).round().astype(int)
        frames = [candidates[int(i)] for i in chosen]
    else:
        frames = candidates
    print(f"Frames: {len(candidates)} candidates; {len(frames)} evenly spaced views sent together to VGGT.", flush=True)

    print("[2/4] Estimating shared VGGT poses and depth...", flush=True)
    with tempfile.TemporaryDirectory(prefix="vggt_frames_") as frame_dir:
        image_paths = []
        for i, frame in enumerate(frames):
            path = Path(frame_dir) / f"frame_{i:04d}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not write temporary VGGT input frame: {path}")
            image_paths.append(str(path))
        vggt_result = estimate_poses_vggt_with_conf(
            image_paths, device="cuda", max_batch_frames=args.max_vggt_frames)

    print("[3/4] Aligning to the telemetry path, fusing depth, and applying confidence gating...", flush=True)
    poses, depth_estimates, Ks, frame_quality, pose_conf = build_pose_depth_source(frames, vggt_result)
    fused = fuse_point_cloud(
        depth_estimates, dynamic_masks=None, poses=poses,
        rgb_frames=[frame.rgb for frame in frames], frame_quality_scores=frame_quality,
        pose_confidences=pose_conf, K=Ks, voxel_size=args.voxel_size)
    confidence = compute_confidence(fused)
    decision = gate_by_confidence(confidence)
    kept_count = int(decision.keep_mask.sum())
    if kept_count < 50_000:
        raise SystemExit(f"Only {kept_count:,} confidence-gated points remain; refusing to generate a misleading mesh.")
    print(f"Fused {len(fused.positions):,} points; kept {kept_count:,} ({decision.keep_mask.mean()*100:.1f}%).", flush=True)

    cloud_path = out / "exp2_fused_cloud.npz"
    np.savez_compressed(
        cloud_path,
        positions=fused.positions.astype(np.float32),
        colors=fused.colors.astype(np.float32),
        keep_mask=decision.keep_mask.astype(bool),
        confidence=confidence.confidence.astype(np.float32))
    summary = {
        "method": "VGGT poses/depth + DJI telemetry scale alignment + project confidence fusion",
        "frames_available": len(candidates), "frames_used_joint_vggt_pass": len(frames),
        "target_fps": args.target_fps, "input_size": [args.width, args.height],
        "camera_hfov_deg_assumed": args.camera_hfov_deg,
        "fused_points": int(len(fused.positions)), "confidence_gated_points": kept_count,
        "confidence_kept_fraction": float(decision.keep_mask.mean()),
        "voxel_size_m": args.voxel_size,
        "poisson_depth": args.poisson_depth,
        "density_trim_quantile": args.density_trim_quantile,
        "min_component_vertices": args.min_component_vertices,
        "fused_cloud_npz": str(cloud_path),
        "coordinate_scale": "SRT/CSV-aligned metric scale; geographic north/origin is not established by this step",
    }
    (out / "vggt_fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[4/4] Building Poisson depth 12, Trim001, Clean500 OBJ + GLB...", flush=True)
    del fused, confidence, decision, poses, depth_estimates, Ks, frame_quality, pose_conf, vggt_result, frames, candidates, dataset
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    mesh_script = root / "layer3_gpu" / "python" / "mesh_exp2_poisson_sweep.py"
    cmd = [
        sys.executable, str(mesh_script),
        "--input_npz", str(cloud_path),
        "--out_dir", str(out),
        "--depths", str(args.poisson_depth),
        "--density_trim_quantile", str(args.density_trim_quantile),
        "--min_component_vertices", str(args.min_component_vertices),
        "--also_save_glb",
    ]
    subprocess.run(cmd, cwd=root, check=True)
    print(f"\nVGGT Clean500 outputs saved under: {out}", flush=True)


if __name__ == "__main__":
    main()
