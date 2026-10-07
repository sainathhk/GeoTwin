#!/usr/bin/env python3
"""Build an RGB TSDF surface from an inspected Experiment 2 VGGT depth cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="vggt_depth_cache.npz from run_exp2_vggt_geometry.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--voxel_size_m", type=float, default=0.40)
    p.add_argument("--sdf_trunc_m", type=float, default=1.20)
    p.add_argument("--max_depth_m", type=float, default=250.0)
    p.add_argument("--confidence_threshold", type=float, default=None,
                   help="Defaults to the threshold recorded in the cache.")
    args = p.parse_args()
    if args.voxel_size_m <= 0 or args.sdf_trunc_m <= 0:
        raise SystemExit("voxel and SDF truncation sizes must be positive.")
    data = np.load(args.cache)
    depth = data["depth_m"].astype(np.float32)
    confidence = data["depth_conf"].astype(np.float32)
    rgb = data["rgb"].astype(np.uint8)
    valid = data["valid_mask"].astype(bool)
    w2c = data["w2c_m"].astype(np.float64)
    K = data["K"].astype(np.float64)
    threshold = float(data["confidence_threshold"]) if args.confidence_threshold is None else args.confidence_threshold

    import open3d as o3d
    n, h, w = depth.shape
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.voxel_size_m),
        sdf_trunc=float(args.sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    integrated = 0
    for i in range(n):
        d = depth[i].copy()
        mask = (valid[i] & np.isfinite(d) & (d > 0) & (d <= args.max_depth_m) &
                np.isfinite(confidence[i]) & (confidence[i] >= threshold))
        d[~mask] = 0.0
        color = o3d.geometry.Image(np.ascontiguousarray(rgb[i]))
        depth_image = o3d.geometry.Image(np.ascontiguousarray(d, dtype=np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_image, depth_scale=1.0, depth_trunc=float(args.max_depth_m),
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(w), int(h), float(K[i, 0, 0]), float(K[i, 1, 1]),
            float(K[i, 0, 2]), float(K[i, 1, 2]),
        )
        # VGGT stores OpenCV world-to-camera extrinsics as 3x4 [R|t].
        # Open3D's ScalableTSDFVolume.integrate requires a homogeneous 4x4 matrix.
        extrinsic_4x4 = np.eye(4, dtype=np.float64)
        extrinsic_4x4[:3, :4] = w2c[i]
        volume.integrate(rgbd, intrinsic, extrinsic_4x4)
        integrated += 1
        print(f"Integrated frame {i + 1}/{n}: {int(mask.sum()):,} depth pixels")

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise SystemExit("TSDF produced an empty surface. Review depth confidence/range and the point map before retrying.")
    mesh.compute_vertex_normals()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mesh_path = out / "exp2_vggt_tsdf_mesh.ply"
    obj_path = out / "exp2_vggt_tsdf_mesh.obj"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False, compressed=True)
    o3d.io.write_triangle_mesh(str(obj_path), mesh, write_ascii=False)

    labels, counts, _ = mesh.cluster_connected_triangles()
    components = int(len(counts))
    largest_fraction = float(max(counts) / sum(counts)) if counts and sum(counts) else 0.0
    vertices = np.asarray(mesh.vertices)
    summary = {
        "experiment": "exp2_vggt_tsdf_mesh",
        "frames_integrated": integrated,
        "confidence_threshold": threshold,
        "voxel_size_m": float(args.voxel_size_m), "sdf_trunc_m": float(args.sdf_trunc_m),
        "depth_trunc_m": float(args.max_depth_m),
        "vertices": int(len(mesh.vertices)), "triangles": int(len(mesh.triangles)),
        "triangle_components": components, "largest_component_triangle_fraction": largest_fraction,
        "bbox_min_m": vertices.min(axis=0).tolist(), "bbox_max_m": vertices.max(axis=0).tolist(),
        "mesh_ply": str(mesh_path), "mesh_obj": str(obj_path),
        "metric_scale_factor_from_srt": float(data["metric_scale"]),
        "gps_path_m_used_for_scale": float(data["gps_path_m"]),
    }
    (out / "exp2_mesh_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
