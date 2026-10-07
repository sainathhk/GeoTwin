#!/usr/bin/env python3
"""Make detail variants from an Experiment 2 fused VGGT point cloud (.npz).

Expected arrays: positions (N,3), colors (N,3), keep_mask (N,). These are
exported from the Colab run before Poisson meshing. This script intentionally
does not resample or invent input points; it compares Poisson octree depths on
the same confidence-gated cloud.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np


# Internal engineering screening targets only. These are tied to this run's
# 0.4 m fusion voxel and are not survey/industry accuracy standards.
INTERNAL_TARGETS = {
    "point_to_surface_p50_m_max": 0.40,
    "point_to_surface_p90_m_max": 1.00,
    "confidence_gated_points_within_1m_min_fraction": 0.90,
    "largest_component_min_fraction": 0.90,
    "nonmanifold_edges_per_approx_edge_max_fraction": 0.001,
}


def mesh_consistency_report(mesh, positions, keep_mask, stats, sample_limit=50000, seed=20261002):
    """Measure in-sample fused-cloud to mesh distance plus basic topology.

    This is a mesh fit/consistency diagnostic, not absolute scene accuracy: the
    same VGGT cloud is used to form and assess the surface. Absolute accuracy
    requires independent surveyed geometry or a reference scan.
    """
    import open3d as o3d

    valid = keep_mask & np.isfinite(positions).all(axis=1)
    points = positions[valid]
    if len(points) == 0:
        raise ValueError("No valid confidence-gated points available for mesh consistency metrics.")
    rng = np.random.default_rng(seed)
    if len(points) > sample_limit:
        take = rng.choice(len(points), size=sample_limit, replace=False)
        sample = points[take]
    else:
        sample = points

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    distances = []
    # Chunk the CPU distance queries to keep temporary memory predictable.
    for start in range(0, len(sample), 10000):
        batch = o3d.core.Tensor(sample[start:start + 10000].astype(np.float32),
                                dtype=o3d.core.Dtype.Float32)
        distances.append(scene.compute_distance(batch).numpy().astype(np.float64))
    d = np.concatenate(distances)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        raise RuntimeError("Open3D returned no finite point-to-surface distances.")

    p50, p90, p95 = (float(np.quantile(d, q)) for q in (0.50, 0.90, 0.95))
    within = {str(m): float(np.mean(d <= m)) for m in (0.4, 1.0, 2.0)}
    approx_edges = max(1.5 * int(stats.n_triangles), 1)
    nonmanifold_fraction = float(stats.n_nonmanifold_edges / approx_edges)
    degenerate_fraction = float(stats.n_degenerate_triangles / max(int(stats.n_triangles), 1))
    metrics = {
        "assessment": "in_sample_mesh_consistency_not_ground_truth_accuracy",
        "reference_points": "random deterministic sample from the confidence-gated VGGT fused cloud used to generate this mesh",
        "units": "approximate metres; VGGT scale aligned to DJI telemetry, not independently surveyed",
        "sample_count": int(len(d)),
        "reference_point_count_total": int(len(points)),
        "point_to_surface_distance_m": {
            "mean": float(np.mean(d)), "rmse": float(np.sqrt(np.mean(np.square(d)))),
            "p50": p50, "p90": p90, "p95": p95, "max": float(np.max(d)),
            "fraction_within_0_4m": within["0.4"],
            "fraction_within_1m": within["1.0"],
            "fraction_within_2m": within["2.0"],
        },
        "mesh_topology": {
            "vertices": int(stats.n_vertices), "triangles": int(stats.n_triangles),
            "connected_components": int(stats.n_connected_components),
            "largest_component_fraction": float(stats.largest_component_fraction),
            "boundary_edges": int(stats.n_boundary_edges),
            "nonmanifold_edges": int(stats.n_nonmanifold_edges),
            "nonmanifold_edges_fraction_approx": nonmanifold_fraction,
            "degenerate_triangles": int(stats.n_degenerate_triangles),
            "degenerate_triangles_fraction": degenerate_fraction,
            "surface_area_m2_approx": float(stats.surface_area),
            "bounding_box_min_m_approx": [float(x) for x in stats.bbox_min],
            "bounding_box_max_m_approx": [float(x) for x in stats.bbox_max],
        },
        "internal_screening_targets_only": {
            "p50_distance_m_at_most": INTERNAL_TARGETS["point_to_surface_p50_m_max"],
            "p90_distance_m_at_most": INTERNAL_TARGETS["point_to_surface_p90_m_max"],
            "fraction_within_1m_at_least": INTERNAL_TARGETS["confidence_gated_points_within_1m_min_fraction"],
            "largest_component_fraction_at_least": INTERNAL_TARGETS["largest_component_min_fraction"],
            "nonmanifold_edges_fraction_approx_at_most": INTERNAL_TARGETS["nonmanifold_edges_per_approx_edge_max_fraction"],
            "basis": "The point-distance goals are approximately 1x and 2.5x the 0.4 m fusion voxel. They are project screening goals, not universal or externally validated acceptance criteria.",
        },
        "target_checks": {
            "p50_distance": p50 <= INTERNAL_TARGETS["point_to_surface_p50_m_max"],
            "p90_distance": p90 <= INTERNAL_TARGETS["point_to_surface_p90_m_max"],
            "within_1m_coverage": within["1.0"] >= INTERNAL_TARGETS["confidence_gated_points_within_1m_min_fraction"],
            "largest_component": float(stats.largest_component_fraction) >= INTERNAL_TARGETS["largest_component_min_fraction"],
            "nonmanifold_edge_fraction": nonmanifold_fraction <= INTERNAL_TARGETS["nonmanifold_edges_per_approx_edge_max_fraction"],
        },
        "limitations": [
            "These in-sample distances describe how closely the extracted mesh follows the same fused VGGT points that generated it; they are not independent accuracy validation.",
            "No absolute geographic, vertical, or building-dimension accuracy is claimed without a surveyed control/reference dataset.",
            "The threshold checks are internal screening targets only and must not be presented as SIH, survey, or industry accuracy standards.",
        ],
    }
    metrics["internal_targets_met_count"] = int(sum(metrics["target_checks"].values()))
    metrics["internal_targets_total"] = len(metrics["target_checks"])
    return metrics


def format_consistency_report(metrics):
    d = metrics["point_to_surface_distance_m"]
    t = metrics["mesh_topology"]
    checks = metrics["target_checks"]
    targets = metrics["internal_screening_targets_only"]
    top = metrics["mesh_topology"]
    nm_pct = top["nonmanifold_edges_fraction_approx"] * 100.0
    lines = [
        "Mesh consistency report (in-sample; NOT ground-truth accuracy)",
        f"  Reference sample: {metrics['sample_count']:,} confidence-gated VGGT points",
        f"  Point-to-surface: mean {d['mean']:.3f} m | RMSE {d['rmse']:.3f} m | "
        f"P50 {d['p50']:.3f} m | P90 {d['p90']:.3f} m | P95 {d['p95']:.3f} m",
        f"  Within 0.4 / 1 / 2 m: {d['fraction_within_0_4m']*100:.1f}% / "
        f"{d['fraction_within_1m']*100:.1f}% / {d['fraction_within_2m']*100:.1f}%",
        f"  Topology: {t['connected_components']} components; largest {t['largest_component_fraction']*100:.1f}%; "
        f"non-manifold {t['nonmanifold_edges']}; boundary {t['boundary_edges']}; "
        f"degenerate triangles {t['degenerate_triangles']}",
        "  Internal targets (actual / target; PASS means screening target met):",
        f"    P50: {d['p50']:.3f} / <= {targets['p50_distance_m_at_most']:.2f} m [{ 'PASS' if checks['p50_distance'] else 'REVIEW' }]",
        f"    P90: {d['p90']:.3f} / <= {targets['p90_distance_m_at_most']:.2f} m [{ 'PASS' if checks['p90_distance'] else 'REVIEW' }]",
        f"    Within 1 m: {d['fraction_within_1m']*100:.1f}% / >= {targets['fraction_within_1m_at_least']*100:.1f}% [{ 'PASS' if checks['within_1m_coverage'] else 'REVIEW' }]",
        f"    Largest component: {top['largest_component_fraction']*100:.1f}% / >= {targets['largest_component_fraction_at_least']*100:.1f}% [{ 'PASS' if checks['largest_component'] else 'REVIEW' }]",
        f"    Non-manifold edge ratio: {nm_pct:.3f}% / <= {targets['nonmanifold_edges_fraction_approx_at_most']*100:.3f}% [{ 'PASS' if checks['nonmanifold_edge_fraction'] else 'REVIEW' }]",
        f"  Internal screening targets met: {metrics['internal_targets_met_count']}/{metrics['internal_targets_total']}.",
        "  Important: these are in-sample mesh-vs-input-cloud checks, not true-world accuracy or completeness. Unseen gaps require independent coverage/reference data to measure.",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_npz", required=True,
                   help="NPZ with positions, colors, keep_mask; optional confidence array.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--depths", type=int, nargs="+", default=[10, 11],
                   help="Poisson octree depths to compare. Start with 10; 11 costs more RAM.")
    p.add_argument("--density_trim_quantile", type=float, default=0.03)
    p.add_argument("--normal_knn", type=int, default=30)
    p.add_argument("--outlier_neighbors", type=int, default=20)
    p.add_argument("--outlier_std_ratio", type=float, default=2.0)
    p.add_argument("--min_component_vertices", type=int, default=0,
                   help="Optional debris cleanup; start with 200 or 500 and compare against an unfiltered export.")
    p.add_argument("--keep_largest_n_components", type=int, default=None,
                   help="Optional explicit component cap. Leave unset unless the scene ROI is well understood.")
    p.add_argument("--also_save_glb", action="store_true")
    args = p.parse_args()
    if any(d < 8 or d > 12 for d in args.depths):
        raise SystemExit("Use Poisson depths 8–12; higher values can consume excessive RAM and amplify noise.")
    if not (0 <= args.density_trim_quantile < 0.5):
        raise SystemExit("density_trim_quantile must be in [0, 0.5).")

    source = np.load(args.input_npz)
    required = {"positions", "colors", "keep_mask"}
    missing = required.difference(source.files)
    if missing:
        raise SystemExit(f"Input archive is missing arrays: {sorted(missing)}")
    positions = np.asarray(source["positions"], dtype=np.float32)
    colors = np.asarray(source["colors"], dtype=np.float32)
    keep = np.asarray(source["keep_mask"], dtype=bool)
    if positions.ndim != 2 or positions.shape[1] != 3 or colors.shape != positions.shape or keep.shape != (len(positions),):
        raise SystemExit("positions/colors/keep_mask must have shapes (N,3), (N,3), and (N,).")
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(colors).all(axis=1)
    keep &= finite
    if int(keep.sum()) < 50_000:
        raise SystemExit(f"Only {int(keep.sum()):,} valid gated points remain; inspect the fusion first.")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from layer1_cpu_sandbox.reconstruction.mesh_export import points_to_mesh, save_obj, save_glb

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for depth in args.depths:
        start = time.time()
        print(f"\nPoisson depth {depth}: meshing {int(keep.sum()):,} gated points...")
        mesh, stats = points_to_mesh(
            positions, colors=colors, keep_mask=keep,
            method="poisson", poisson_depth=int(depth),
            density_trim_quantile=args.density_trim_quantile,
            normal_knn=args.normal_knn,
            statistical_outlier_neighbors=args.outlier_neighbors,
            statistical_outlier_std_ratio=args.outlier_std_ratio,
            min_component_vertices=args.min_component_vertices,
            keep_largest_n_components=args.keep_largest_n_components,
        )
        base = out / f"vggt_poisson_depth{depth}"
        obj_path = str(base.with_suffix(".obj"))
        save_obj(mesh, obj_path)
        glb_path = None
        if args.also_save_glb:
            glb_path = str(base.with_suffix(".glb"))
            save_glb(mesh, glb_path)
        report = dataclasses.asdict(stats)
        accuracy = mesh_consistency_report(mesh, positions, keep, stats)
        accuracy_json = out / f"vggt_poisson_depth{depth}_mesh_accuracy.json"
        accuracy_text = out / f"vggt_poisson_depth{depth}_mesh_accuracy.txt"
        accuracy_json.write_text(json.dumps(accuracy, indent=2), encoding="utf-8")
        accuracy_text.write_text(format_consistency_report(accuracy) + "\n", encoding="utf-8")
        report.update({
            "poisson_depth": int(depth), "input_gated_points": int(keep.sum()),
            "density_trim_quantile": float(args.density_trim_quantile),
            "elapsed_seconds": round(time.time() - start, 2),
            "obj": obj_path, "glb": glb_path,
            "mesh_consistency_report_json": str(accuracy_json),
            "mesh_consistency_report_text": str(accuracy_text),
            "mesh_consistency": accuracy,
        })
        reports.append(report)
        print(stats.format_report())
        print(format_consistency_report(accuracy))
        print(f"Saved: {obj_path}" + (f" and {glb_path}" if glb_path else ""))
        print(f"Mesh consistency report: {accuracy_json}")

    report_path = out / "poisson_depth_sweep.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nComparison report: {report_path}")
    print("The reported screening thresholds are internal project goals, not external accuracy standards. Absolute OBJ accuracy cannot be determined without independent surveyed/reference geometry.")


if __name__ == "__main__":
    main()
