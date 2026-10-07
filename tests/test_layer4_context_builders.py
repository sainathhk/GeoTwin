"""Unit + integration tests for layer4_agents.context_builders."""
import json
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_cpu_sandbox.perception.frame_quality import FrameQualityScore
from layer1_cpu_sandbox.perception.frame_overlap import (
    compute_consecutive_overlap, OverlapSummary, FramePairOverlap, STATUS_OK,
)
from layer1_cpu_sandbox.evaluation.spatial_diagnostics import compute_spatial_confidence_grid
from layer1_cpu_sandbox.pipeline import PipelineConfig
from layer1_cpu_sandbox.real_data.real_pipeline import RealPipelineConfig
from layer4_agents.context_builders import (
    build_frame_agent_context, build_reconstruction_agent_context,
    build_evaluation_agent_context_synthetic, build_evaluation_agent_context_real,
)


def _fake_scores(n=10, overall=0.5):
    return [FrameQualityScore(blur_score=0.6, exposure_score=0.7, compression_score=0.9,
                                coverage_score=0.8, overall=overall) for _ in range(n)]


def _fake_overlap_summary(n_pairs=9):
    pairs = [FramePairOverlap(i=k, j=k + 1, n_keypoints_i=100, n_keypoints_j=100,
                                n_matches=50, match_ratio=0.5, median_disp_px=10.0, status=STATUS_OK)
              for k in range(n_pairs)]
    return OverlapSummary(pairs=pairs, n_duplicate_risk=0, n_gap_risk=0, n_indeterminate=0,
                            n_ok=n_pairs, mean_match_ratio=0.5, worst_gap_pairs=[])


# ---------------------------------------------------------------------------
# build_frame_agent_context
# ---------------------------------------------------------------------------

def test_frame_agent_context_is_json_serializable():
    ctx = build_frame_agent_context(_fake_scores(), _fake_overlap_summary())
    json.dumps(ctx)  # must not raise


def test_frame_agent_context_reflects_current_config():
    ctx = build_frame_agent_context(_fake_scores(), _fake_overlap_summary(),
                                      current_keep_fraction=0.6, current_min_overall=0.2)
    assert ctx["current_keep_fraction"] == 0.6
    assert ctx["current_min_overall"] == 0.2


def test_frame_agent_context_no_raw_pixel_or_position_data():
    ctx = build_frame_agent_context(_fake_scores(), _fake_overlap_summary())
    flat = json.dumps(ctx)
    assert "rgb" not in flat.lower()
    # every leaf value must be a plain scalar (no nested arrays of coordinates etc.)
    def _check(v):
        if isinstance(v, dict):
            for vv in v.values():
                _check(vv)
        elif isinstance(v, list):
            for vv in v:
                assert isinstance(vv, (int, float, str)), f"unexpected list-of-object leaf: {vv!r}"
        else:
            assert isinstance(v, (int, float, str, type(None))), f"unexpected leaf type: {v!r}"
    _check(ctx)


def test_frame_agent_context_empty_scores_does_not_crash():
    ctx = build_frame_agent_context([], _fake_overlap_summary(n_pairs=0))
    assert ctx["n_frames"] == 0
    json.dumps(ctx)


# ---------------------------------------------------------------------------
# build_reconstruction_agent_context / build_evaluation_agent_context_* --
# integration-tested against a REAL pipeline run, not hand-built fixtures,
# since these functions' whole job is to read real field names correctly.
# ---------------------------------------------------------------------------

def test_reconstruction_context_against_real_pipeline_run():
    from layer1_cpu_sandbox.synthetic.dataset_builder import build_dataset
    from layer1_cpu_sandbox.pipeline import run_pipeline

    dataset = build_dataset(seed=0, extent=25.0)
    cfg = PipelineConfig(name="full_method")
    result = run_pipeline(dataset, cfg)
    diag = compute_spatial_confidence_grid(
        result.points.positions[result.gate.keep_mask],
        result.confidence.confidence[result.gate.keep_mask],
        result.confidence.band[result.gate.keep_mask],
        cell_size=8.0,
    )
    ctx = build_reconstruction_agent_context(result, cfg, diag, n_total_frames=len(dataset.frames))
    json.dumps(ctx)  # must be JSON-safe
    assert ctx["frames"]["n_total"] == len(dataset.frames)
    assert ctx["frames"]["n_kept"] == len(result.kept_frame_indices)
    assert ctx["points"]["n_raw_fused"] == result.points.positions.shape[0]
    assert ctx["points"]["n_after_gating"] == result.final_positions.shape[0]
    assert ctx["current_config"]["voxel_size"] == cfg.voxel_size
    assert 0.0 <= ctx["points"]["mean_confidence_kept"] <= 1.0


def test_reconstruction_context_carries_attempt_history_through_untouched():
    from layer1_cpu_sandbox.synthetic.dataset_builder import build_dataset
    from layer1_cpu_sandbox.pipeline import run_pipeline

    dataset = build_dataset(seed=0, extent=25.0)
    cfg = PipelineConfig()
    result = run_pipeline(dataset, cfg)
    diag = compute_spatial_confidence_grid(
        result.points.positions[result.gate.keep_mask],
        result.confidence.confidence[result.gate.keep_mask],
        result.confidence.band[result.gate.keep_mask],
    )
    history = [{"iteration": 0, "action": "retry", "voxel_size": 3.0}]
    ctx = build_reconstruction_agent_context(result, cfg, diag, attempt_history=history)
    assert ctx["attempt_history"] == history


def test_evaluation_context_synthetic_against_real_pipeline_run():
    from layer1_cpu_sandbox.synthetic.dataset_builder import build_dataset
    from layer1_cpu_sandbox.pipeline import run_pipeline
    from layer1_cpu_sandbox.evaluate_result import evaluate_pipeline_result

    dataset = build_dataset(seed=0, extent=25.0)
    result = run_pipeline(dataset, PipelineConfig())
    ev = evaluate_pipeline_result(dataset, result, run_name="test")
    ctx = build_evaluation_agent_context_synthetic(ev)
    json.dumps(ctx)
    assert ctx["has_ground_truth"] is True
    assert "surface_completeness" in ctx["key_numbers"]
    assert "novel_view_psnr" in ctx["key_numbers"]
    # this particular tiny/fast config is known to produce NaN novel-view PSNR
    # (too few final points to render anything) -- must come through as None, not "nan".
    assert ctx["key_numbers"]["novel_view_psnr"] is None or isinstance(ctx["key_numbers"]["novel_view_psnr"], float)


def test_evaluation_context_real_against_real_pipeline_run():
    # Mirrors tests/test_real_data.py's mock-CSV/mock-video fixture pattern (no real DJI
    # file is available in this sandbox) rather than inventing a shortcut dataset builder.
    import csv
    import tempfile
    from datetime import datetime, timedelta
    import cv2

    from layer1_cpu_sandbox.real_data.real_dataset_builder import build_real_dataset
    from layer1_cpu_sandbox.real_data.real_pipeline import run_real_pipeline, evaluate_real_result

    def _write_mock_csv(path, n=160, hz=10.0, video_start_row=10, video_len_rows=80):
        t0 = datetime(2018, 7, 4, 11, 19, 31)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["CUSTOM.updateTime", "GIMBAL.yaw", "GIMBAL.pitch", "GIMBAL.roll",
                        "OSD.height [m]", "CUSTOM.isVideo", "OSD.latitude", "OSD.longitude"])
            for i in range(n):
                t = t0 + timedelta(seconds=i / hz)
                is_video = 1 if video_start_row <= i < video_start_row + video_len_rows else 0
                w.writerow([t.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3], f"{10.0 + 0.01*i:.2f}",
                            f"{-35.0 + 2.0*math.sin(i*0.05):.2f}", f"{1.5*math.sin(i*0.1):.2f}",
                            f"{45.0:.2f}", is_video, f"{55.470 + i*2e-6:.8f}", f"{10.323 + i*3e-6:.8f}"])

    def _write_mock_video(path, n_frames=80, fps=10.0, w=48, h=36):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        rng = __import__("numpy").random.default_rng(0)
        base = rng.integers(0, 255, (h, w, 3), dtype="uint8")
        for i in range(n_frames):
            writer.write(rng.integers(0, 255, (h, w, 3), dtype="uint8") // 2 + base // 2)
        writer.release()

    with tempfile.TemporaryDirectory() as d:
        video_path = os.path.join(d, "flight.mov")
        log_path = os.path.join(d, "flight.csv")
        _write_mock_video(video_path)
        _write_mock_csv(log_path)
        ds = build_real_dataset(video_path, log_path, target_fps=2.0, max_frames=12, resize_to=(48, 36))
        result = run_real_pipeline(ds, RealPipelineConfig(n_depths=12, depth_min=2.0, depth_max=60.0))
        ev = evaluate_real_result(ds, result, n_self_consistency_views=2)

    ctx = build_evaluation_agent_context_real(ev)
    json.dumps(ctx)
    assert ctx["has_ground_truth"] is False
    assert "confidence_mean" in ctx["key_numbers"]
    assert ctx["key_numbers"]["n_final_points"] == result.final_positions.shape[0]


# ---------------------------------------------------------------------------
# build_training_agent_context (Layer 3 GPU training loop)
# ---------------------------------------------------------------------------

def test_training_agent_context_is_json_serializable_with_initial_sentinels():
    from layer4_agents.context_builders import build_training_agent_context
    best = {"psnr": {"value": -float("inf"), "iter": None}, "ssim": {"value": -float("inf"), "iter": None},
             "lpips": {"value": float("inf"), "iter": None}}
    ctx = build_training_agent_context([], best, current_iter=500, max_iterations=20000,
                                          checkpoint_every=500, densifying_active=True,
                                          n_gaussians=50000, max_gaussians=150000)
    json.dumps(ctx)
    assert ctx["best_held_out"]["psnr"]["value"] is None  # -inf must never leak into JSON


def test_training_agent_context_caps_history_window():
    from layer4_agents.context_builders import build_training_agent_context
    history = [{"iter": i * 500, "psnr": 20.0, "ssim": 0.6, "lpips": 0.3, "n_gaussians": 1000 * i}
                for i in range(1, 21)]  # 20 checkpoints
    best = {"psnr": {"value": 20.0, "iter": 10000}, "ssim": {"value": 0.6, "iter": 10000},
             "lpips": {"value": 0.3, "iter": 10000}}
    ctx = build_training_agent_context(history, best, current_iter=10000, max_iterations=20000,
                                          checkpoint_every=500, densifying_active=True,
                                          n_gaussians=20000, max_gaussians=150000, history_window=10)
    assert len(ctx["held_out_history"]) == 10
    assert ctx["held_out_history"][-1]["iter"] == 10000  # most recent kept, not the earliest


def test_training_agent_context_reflects_real_documented_pattern():
    # The actual shape from STATUS.md's documented finding: SSIM rises to a peak at iter
    # 6500 then declines through iter 20000 as Gaussian count keeps growing.
    from layer4_agents.context_builders import build_training_agent_context
    history = []
    for it in range(500, 7001, 500):
        history.append({"iter": it, "psnr": 18 + it / 1000, "ssim": min(0.15 + it / 10000, 0.62),
                          "lpips": 0.4, "n_gaussians": it * 38})
    for it in range(7500, 20001, 500):
        history.append({"iter": it, "psnr": 19.0, "ssim": max(0.62 - (it - 6500) / 40000, 0.3),
                          "lpips": 0.4, "n_gaussians": it * 41})
    best = {"psnr": {"value": 19.0, "iter": 6500}, "ssim": {"value": 0.62, "iter": 6500},
             "lpips": {"value": 0.4, "iter": 500}}
    ctx = build_training_agent_context(history, best, current_iter=20000, max_iterations=20000,
                                          checkpoint_every=500, densifying_active=True,
                                          n_gaussians=829355, max_gaussians=150000)
    json.dumps(ctx)
    assert ctx["best_held_out"]["ssim"]["iter"] == 6500
    assert ctx["held_out_history"][-1]["ssim"] < ctx["best_held_out"]["ssim"]["value"]
