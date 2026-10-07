"""
export_obj.py
LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, CPU-executed against
a real checkpoint (docs/MESH_QUALITY_AUDIT.md, 2026-09) -- unlike the rest of
train_gpu.py, this script needs no CUDA/rasterizer (loading a checkpoint +
confidence/opacity/geometry math is plain tensor/numpy ops, see --device below),
so it was possible to actually run it against a real full_state_iter8000.pt
rather than only exercising it by inspection.

Stand-alone .obj (re-)export from a saved `full_state_iter*.pt` checkpoint
(written by train_gpu.py's `_save_full_state`), so a mesh can be regenerated
from ANY checkpoint -- in particular the best-held-out-SSIM/PSNR iteration
`best_checkpoint.json` records, which train()'s own final-iteration printout
says explicitly is not always the LAST checkpoint (see STATUS.md's overfitting
finding: best held-out SSIM at iter 6500/246K Gaussians on one real run,
declining monotonically through iter 20000/829K) -- without re-running
training, and with different --mesh_* settings than whatever the original
`train_gpu.py` invocation used.

Usage:
    python3 layer3_gpu/python/export_obj.py \
        --checkpoint outputs/real_single_pass_v2/full_state_iter6500.pt \
        --out outputs/real_single_pass_v2/mesh_iter6500_best_ssim.obj

--min_opacity/--min_confidence default to "auto": a data-driven threshold found
from the ACTUAL distribution in this checkpoint (see
gaussian_geometry.suggest_threshold_from_valley), not a fixed number carried
over from a different checkpoint's calibration. See docs/MESH_QUALITY_AUDIT.md
for why this matters: the previous fixed default (--min_confidence 0.15) turned
out to be a complete no-op on the checkpoint that prompted this audit -- 100% of
its Gaussians cleared it, because 0.15 had never been checked against a real
confidence distribution, only chosen as a plausible-sounding number. Pass an
explicit float to override auto-detection.

Checkpoints saved before the confidence_stats fix in _save_full_state (see
that function's docstring) will still load, but --min_confidence then filters
on each Gaussian's creation-time prior rather than its true end-of-training
confidence -- a printed warning below makes this visible rather than silent.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train_gpu import load_full_state, _gaussian_positions_and_colors
from layer1_cpu_sandbox.reconstruction.mesh_export import export_point_cloud_to_obj, MeshExportError
from layer1_cpu_sandbox.reconstruction.gaussian_geometry import (
    compute_gaussian_geometry, suggest_threshold_from_valley, floater_mask,
)
from layer1_cpu_sandbox.reconstruction.frustum_check import check_frustum_shape, vfov_from_hfov


def _float_or_auto(s: str):
    if s == "auto":
        return "auto"
    return float(s)


def _print_percentiles(name: str, values: np.ndarray, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    pct = np.percentile(values, ps)
    body = "  ".join(f"p{p:02d}={v:.4g}" for p, v in zip(ps, pct))
    print(f"    {name}: {body}")


def _resolve_threshold(name: str, arg_value, values: np.ndarray) -> float:
    if arg_value == "auto":
        t = suggest_threshold_from_valley(values)
        print(f"  --{name} auto -> {t:.4g} "
              f"(histogram-valley threshold found in this checkpoint's actual distribution)")
        return t
    print(f"  --{name} {arg_value} (explicit)")
    return float(arg_value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="full_state_iter*.pt written by train_gpu.py")
    ap.add_argument("--out", required=True, help="output .obj path")
    ap.add_argument("--device", default="cpu",
                    help="'cpu' is fine and is the default -- this is a read-only export (no training, "
                         "no rasterizer call), so no CUDA device is required here even though train_gpu.py "
                         "itself needs one.")
    ap.add_argument("--min_opacity", type=_float_or_auto, default="auto",
                     help="float, or 'auto' (default) for a data-driven threshold -- see module docstring")
    ap.add_argument("--min_confidence", type=_float_or_auto, default="auto",
                     help="float, or 'auto' (default) for a data-driven threshold -- see module docstring")
    ap.add_argument("--floater_filter", action=argparse.BooleanOptionalAction, default=True,
                     help="exclude roughly-isotropic, unusually-large Gaussians that look like haze/sky/"
                          "background rather than surface samples (gaussian_geometry.floater_mask). "
                          "On (2026-09-audited data): ~1%% of Gaussians. --no-floater_filter to disable.")
    ap.add_argument("--floater_anisotropy_below", type=float, default=3.0)
    ap.add_argument("--floater_scale_percentile", type=float, default=90.0)
    ap.add_argument("--use_covariance_normals", action=argparse.BooleanOptionalAction, default=True,
                     help="derive surface normals from each Gaussian's own scale+rotation "
                          "(gaussian_geometry.compute_gaussian_geometry) instead of generic point-cloud "
                          "PCA. See docs/MESH_QUALITY_AUDIT.md for what this does and does not fix by "
                          "itself. --no-use_covariance_normals for the old behavior.")
    ap.add_argument("--method", choices=["poisson", "ball_pivoting"], default="poisson")
    ap.add_argument("--poisson_depth", type=int, default=9)
    ap.add_argument("--density_trim_quantile", type=float, default=0.03)
    ap.add_argument("--voxel_size", type=float, default=None,
                     help="optional pre-reconstruction downsample to homogenize point density "
                          "(Gaussian centers are NOT uniformly spaced -- see mesh_export.py docstring). "
                          "Try ~1-2x the p50 nearest-neighbor spacing printed below if the mesh looks "
                          "noisy/spiky at the default (None = no downsampling).")
    ap.add_argument("--sor_neighbors", type=int, default=20,
                     help="statistical outlier removal neighbor count; 0 disables")
    ap.add_argument("--sor_std_ratio", type=float, default=2.0)
    ap.add_argument("--min_component_vertices", type=int, default=0,
                     help="drop connected components smaller than this many vertices (debris cleanup); "
                          "0 disables")
    ap.add_argument("--keep_top_k_components", type=int, default=None,
                     help="keep only the N largest connected components; None disables")
    ap.add_argument("--glb", action="store_true", help="also write a .glb next to --out")
    ap.add_argument("--camera_hfov_deg", type=float, default=84.0,
                     help="camera horizontal FOV, used by the frustum-shape sanity check below. "
                          "Must match what training used (train_gpu.py's own --camera_hfov_deg).")
    ap.add_argument("--frame_width", type=int, default=1280)
    ap.add_argument("--frame_height", type=int, default=720)
    ap.add_argument("--skip_frustum_check", action="store_true",
                     help="skip the geometry sanity check (not recommended -- see frustum_check.py)")
    ap.add_argument("--min_points", type=int, default=50)
    args = ap.parse_args()

    model = load_full_state(args.checkpoint, device=args.device)
    if float(model.confidence_stats.n_accum.sum()) == 0.0:
        print("NOTE: this checkpoint's confidence_stats are all-zero -- either it predates the "
              "_save_full_state confidence fix (see train_gpu.py), or training never actually observed "
              "any of these Gaussians (unlikely for a real run). --min_confidence is filtering on "
              "creation-time prior_confidence, not accumulated training evidence. --min_opacity is "
              "unaffected either way (opacity is a direct model parameter, not an accumulator).")

    positions, colors = _gaussian_positions_and_colors(model)
    opacity = model.get_opacity().detach().cpu().numpy().reshape(-1)
    confidence = model.confidence().detach().cpu().numpy().reshape(-1)
    scale = model.get_scaling().detach().cpu().numpy()
    rotation = model.get_rotation().detach().cpu().numpy()
    geometry = compute_gaussian_geometry(scale, rotation)
    n = model.n_gaussians  # @property on GaussianModel, NOT a method -- do not call it

    print(f"=== Gaussian stage: {n} Gaussians loaded from {args.checkpoint} ===")
    bbox_min, bbox_max = positions.min(0), positions.max(0)
    centroid = positions.mean(0)
    dist = np.linalg.norm(positions - centroid, axis=1)
    print(f"  spatial extent: bbox {np.round(bbox_min,3).tolist()} .. {np.round(bbox_max,3).tolist()} "
          f"(diag={np.linalg.norm(bbox_max-bbox_min):.3g})")
    _print_percentiles("distance from centroid", dist)
    _print_percentiles("opacity", opacity)
    _print_percentiles("confidence", confidence)
    _print_percentiles("scale (mean of 3 axes)", geometry.scale_mean)
    _print_percentiles("anisotropy (max/min axis)", geometry.anisotropy)
    print(f"    fraction with anisotropy>5 (disk-like/surface-hugging): {(geometry.anisotropy>5).mean()*100:.1f}%")

    print("\n=== Filtering thresholds ===")
    min_opacity = _resolve_threshold("min_opacity", args.min_opacity, opacity)
    min_confidence = _resolve_threshold("min_confidence", args.min_confidence, confidence)

    mask_opacity = opacity >= min_opacity
    mask_confidence = confidence >= min_confidence
    keep_mask = mask_opacity & mask_confidence
    if args.floater_filter:
        mask_floater = floater_mask(geometry, anisotropy_below=args.floater_anisotropy_below,
                                     scale_above_percentile=args.floater_scale_percentile)
        keep_mask &= ~mask_floater
        print(f"  floater_filter: {int(mask_floater.sum())}/{n} flagged "
              f"(anisotropy<{args.floater_anisotropy_below} & scale>p{args.floater_scale_percentile})")

    print(f"\n  removed by opacity threshold alone:    {int((~mask_opacity).sum())}/{n}")
    print(f"  removed by confidence threshold alone: {int((~mask_confidence).sum())}/{n}")
    print(f"  SURVIVING (all filters combined):      {int(keep_mask.sum())}/{n} "
          f"({100*keep_mask.mean():.1f}%)")

    if not args.skip_frustum_check:
        vfov = vfov_from_hfov(args.camera_hfov_deg, args.frame_width, args.frame_height)
        fres = check_frustum_shape(positions[keep_mask], expected_hfov_deg=args.camera_hfov_deg,
                                    expected_vfov_deg=vfov)
        print(f"\n=== Geometry sanity check (frustum_check.py) ===")
        print(fres.format_report())
        if fres.is_frustum_shaped:
            print("\n  !!! STOP AND READ THIS BEFORE USING THE MESH !!!")
            print("  This point cloud is shaped like the camera's viewing volume, not like a scene.")
            print("  That means the DEPTHS are not real: every pixel has been placed at some")
            print("  arbitrary distance inside the tested depth range. 3DGS can still render such a")
            print("  model beautifully (photometric loss only constrains where a Gaussian PROJECTS,")
            print("  not where it SITS), so good PSNR/SSIM does NOT clear this.")
            print("  No meshing algorithm can fix it -- the geometry was never recovered.")
            print("  See docs/MESH_QUALITY_AUDIT.md for what to change upstream.")

    normals = geometry.normals if args.use_covariance_normals else None
    print(f"\n  normals: {'Gaussian covariance (scale+rotation)' if normals is not None else 'generic point-cloud PCA (open3d estimate_normals)'}")

    try:
        stats = export_point_cloud_to_obj(
            args.out, positions, colors, keep_mask=keep_mask, normals=normals,
            method=args.method, poisson_depth=args.poisson_depth,
            density_trim_quantile=args.density_trim_quantile, voxel_size=args.voxel_size,
            statistical_outlier_neighbors=args.sor_neighbors, statistical_outlier_std_ratio=args.sor_std_ratio,
            min_component_vertices=args.min_component_vertices,
            keep_largest_n_components=args.keep_top_k_components,
            min_points=args.min_points, also_save_glb=args.glb)
    except MeshExportError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)

    print(f"\n=== Mesh stage: wrote {stats.path} ===")
    print(stats.format_report())
    if args.glb:
        print(f"  also wrote {stats.path.rsplit('.', 1)[0]}.glb")


if __name__ == "__main__":
    main()
