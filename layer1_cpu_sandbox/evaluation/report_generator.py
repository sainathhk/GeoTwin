"""
report_generator.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested

Turns real computed results (ablation DataFrames, a PipelineResult, a
SyntheticDroneDataset) into the artifacts a reader/judge actually looks
at: a markdown ablation table, a confidence-vs-error validation figure
(the single most important plot for the core novelty claim), a
trajectory/coverage figure, and a qualitative GT-vs-reconstruction figure
with an error heatmap. Every number plotted here was computed by the
modules above -- nothing in this file invents data.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from ..synthetic.dataset_builder import SyntheticDroneDataset
from ..pipeline import PipelineResult
from ..reconstruction.point_renderer import render_point_cloud


def save_ablation_table(df: pd.DataFrame, out_dir: str, cols=None, filename="ablation_table"):
    if cols is None:
        cols = ["run_name", "n_points_final", "psnr_novel_view", "ssim_novel_view",
                "chamfer_distance_m", "point_to_point_rmse_m", "mean_abs_height_error_m",
                "control_point_rmse_m", "surface_completeness", "total_time_s"]
    sub = df[cols].copy()
    for c in sub.columns:
        if sub[c].dtype == float:
            sub[c] = sub[c].round(3)
    csv_path = os.path.join(out_dir, f"{filename}.csv")
    sub.to_csv(csv_path, index=False)
    md_path = os.path.join(out_dir, f"{filename}.md")
    with open(md_path, "w") as f:
        f.write(sub.to_markdown(index=False))
    return csv_path, md_path


def plot_confidence_vs_error(dataset: SyntheticDroneDataset, result: PipelineResult, out_path: str,
                              n_bins: int = 10):
    """THE core-novelty validation figure: does confidence actually predict error?
    Uses the PRE-gating point set so the full confidence spectrum is visible."""
    tree = cKDTree(dataset.gt_points.points)
    d, _ = tree.query(result.points.positions, k=1)
    conf = result.confidence.confidence

    df = pd.DataFrame({"confidence": conf, "error_m": d})
    df["bin"] = pd.qcut(df["confidence"], n_bins, labels=False, duplicates="drop")
    grouped = df.groupby("bin").agg(mean_conf=("confidence", "mean"), mean_err=("error_m", "mean"),
                                     median_err=("error_m", "median"), n=("error_m", "size")).reset_index()

    corr = float(np.corrcoef(conf, d)[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(conf, d, s=4, alpha=0.15, color="#2b6cb0")
    axes[0].set_xlabel("Reconstruction confidence")
    axes[0].set_ylabel("Distance to nearest GT surface point (m)")
    axes[0].set_title(f"Per-point confidence vs error (n={len(d)}, Pearson r={corr:.3f})")

    axes[1].bar(grouped["mean_conf"], grouped["mean_err"], width=0.06, color="#2f855a")
    axes[1].set_xlabel("Mean confidence (per decile bin)")
    axes[1].set_ylabel("Mean error (m)")
    axes[1].set_title("Confidence-decile vs mean error (lower is better)")
    fig.suptitle("Core novelty check: does the confidence score actually predict error?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return corr, grouped


def plot_trajectory_and_coverage(dataset: SyntheticDroneDataset, coverage_status: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    tri_centroids = np.stack([t.centroid[:2] for t in dataset.scene.triangles])
    color_map = {"never_observable": "#c53030", "observable_but_dropped": "#dd6b20", "observed": "#2f855a"}
    for status, col in color_map.items():
        m = coverage_status == status
        if m.any():
            ax.scatter(tri_centroids[m, 0], tri_centroids[m, 1], s=3, color=col, label=status, alpha=0.6)

    true_xy = np.stack([s.position[:2] for s in dataset.trajectory.samples])
    ax.plot(true_xy[:, 0], true_xy[:, 1], color="black", linewidth=1.5, label="true flight path")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Single-pass coverage: fundamental blind spot vs pipeline-dropped vs observed")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_qualitative_comparison(dataset: SyntheticDroneDataset, result: PipelineResult, frame_idx: int, out_path: str):
    pose = dataset.frames[frame_idx].true_pose
    gt_rgb = dataset.frames[frame_idx].rgb_clean
    rendered, mask = render_point_cloud(result.final_positions, result.final_colors, dataset.K, pose)

    diff = np.abs(rendered.astype(np.float32) - gt_rgb.astype(np.float32)).mean(axis=-1)
    diff = np.where(mask, diff, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt_rgb); axes[0].set_title("Ground truth render"); axes[0].axis("off")
    axes[1].imshow(rendered); axes[1].set_title(f"Reconstruction render\n({mask.mean()*100:.1f}% pixels covered)"); axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="inferno", vmin=0, vmax=80)
    axes[2].set_title("Abs. error map (covered px only)"); axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
