"""Unit tests for layer4_agents.mock_agents -- each branch of the rule tables exercised
directly against hand-built context dicts (these agents' whole job is being predictable
and inspectable, so the tests check the rules, not just "did it return an object")."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer4_agents.mock_agents import MockFrameAgent, MockReconstructionAgent, MockEvaluationAgent
from layer4_agents.schemas import KEEP_FRACTION_BOUNDS


# ---------------------------------------------------------------------------
# MockFrameAgent
# ---------------------------------------------------------------------------

def _frame_ctx(n_pairs=20, n_gap=0, n_dup=0, frac_unusable=0.0,
                keep_fraction=0.75, min_overall=0.15):
    return {
        "current_keep_fraction": keep_fraction,
        "current_min_overall": min_overall,
        "quality": {"fraction_below_0.15": frac_unusable},
        "raw_capture_overlap": {"n_pairs": n_pairs, "n_gap_risk": n_gap, "n_duplicate_risk": n_dup},
    }


def test_frame_agent_healthy_case_keeps_defaults():
    d = MockFrameAgent().decide(_frame_ctx())
    assert d.keep_fraction == 0.75
    assert d.min_overall == 0.15
    assert d.flags == []


def test_frame_agent_raises_keep_fraction_on_gap_risk():
    d = MockFrameAgent().decide(_frame_ctx(n_pairs=20, n_gap=5))  # 25% gap_risk
    assert d.keep_fraction > 0.75
    assert d.keep_fraction <= KEEP_FRACTION_BOUNDS[1]
    assert any("gap" in f.lower() for f in d.flags)


def test_frame_agent_lowers_keep_fraction_on_duplicate_risk():
    d = MockFrameAgent().decide(_frame_ctx(n_pairs=20, n_dup=8))  # 40% duplicate_risk, no gap
    assert d.keep_fraction < 0.75
    assert d.keep_fraction >= KEEP_FRACTION_BOUNDS[0]
    assert any("duplicate" in f.lower() for f in d.flags)


def test_frame_agent_gap_risk_takes_priority_over_duplicate_risk():
    # Both present -> protecting coverage (gap) matters more than thinning redundancy (dup).
    d = MockFrameAgent().decide(_frame_ctx(n_pairs=20, n_gap=5, n_dup=8))
    assert d.keep_fraction > 0.75


def test_frame_agent_lowers_min_overall_when_most_frames_unusable():
    d = MockFrameAgent().decide(_frame_ctx(frac_unusable=0.6))
    assert d.min_overall < 0.15
    assert any("floor" in f.lower() for f in d.flags)


def test_frame_agent_decision_never_exceeds_documented_bounds_even_from_extreme_input():
    d = MockFrameAgent().decide(_frame_ctx(n_pairs=1, n_gap=1, frac_unusable=1.0,
                                             keep_fraction=0.95, min_overall=0.02))
    assert KEEP_FRACTION_BOUNDS[0] <= d.keep_fraction <= KEEP_FRACTION_BOUNDS[1]


# ---------------------------------------------------------------------------
# MockReconstructionAgent
# ---------------------------------------------------------------------------

def _recon_ctx(pts_per_frame=50.0, n_kept=20, weak_regions=None, low_frac=0.1,
                history=None, depth_min=8.0, depth_max=140.0, n_depths=32, depth_window=2):
    return {
        "current_config": {"depth_min": depth_min, "depth_max": depth_max,
                             "n_depths": n_depths, "depth_window": depth_window},
        "frames": {"n_kept": n_kept, "n_total": n_kept + 5, "n_rejected": 5},
        "points": {"n_raw_fused": int(pts_per_frame * n_kept),
                    "low_confidence_fraction_kept": low_frac},
        "spatial_diagnostic": {"weak_regions": weak_regions or []},
        "attempt_history": history or [],
    }


def test_reconstruction_agent_healthy_case_accepts():
    d = MockReconstructionAgent().decide(_recon_ctx())
    assert d.action == "accept"


def test_reconstruction_agent_retries_on_low_point_density():
    d = MockReconstructionAgent().decide(_recon_ctx(pts_per_frame=2.0))
    assert d.action == "retry"
    assert d.depth_min is not None and d.depth_max is not None
    assert d.depth_min < 8.0   # widened outward, not narrowed
    assert d.depth_max > 140.0
    assert d.n_depths is not None and d.n_depths > 32


def test_reconstruction_agent_retries_depth_window_on_weak_region_first_attempt():
    weak = [{"direction": "northwest", "n_points": 9, "low_confidence_fraction": 0.6}]
    d = MockReconstructionAgent().decide(_recon_ctx(pts_per_frame=50.0, weak_regions=weak))
    assert d.action == "retry"
    assert d.depth_window == 3  # 2 -> 3
    assert "northwest" in d.reasoning


def test_reconstruction_agent_flags_and_accepts_high_low_confidence_with_no_fixable_cause():
    d = MockReconstructionAgent().decide(_recon_ctx(pts_per_frame=50.0, weak_regions=[], low_frac=0.6))
    assert d.action == "flag_and_accept"


def test_reconstruction_agent_never_touches_prune_or_densify_thresholds():
    # Across every branch, retry must never propose changing the confidence gate itself.
    scenarios = [
        _recon_ctx(pts_per_frame=2.0),
        _recon_ctx(weak_regions=[{"direction": "east", "n_points": 9, "low_confidence_fraction": 0.5}]),
        _recon_ctx(low_frac=0.6),
    ]
    for ctx in scenarios:
        d = MockReconstructionAgent().decide(ctx)
        assert d.prune_below is None
        assert d.densify_above is None


def test_reconstruction_agent_stops_retrying_after_two_attempts():
    history = [{"action": "retry"}, {"action": "retry"}]
    d = MockReconstructionAgent().decide(_recon_ctx(pts_per_frame=2.0, low_frac=0.6, history=history))
    assert d.action == "flag_and_accept"


def test_reconstruction_agent_accepts_after_retries_if_metrics_recovered():
    history = [{"action": "retry"}, {"action": "retry"}]
    d = MockReconstructionAgent().decide(_recon_ctx(pts_per_frame=50.0, low_frac=0.1, history=history))
    assert d.action == "accept"


# ---------------------------------------------------------------------------
# MockEvaluationAgent
# ---------------------------------------------------------------------------

def test_evaluation_agent_synthetic_good_rmse_accepts():
    r = MockEvaluationAgent().decide({"has_ground_truth": True,
                                        "key_numbers": {"control_point_rmse_m": 8.0, "surface_completeness": 0.3}})
    assert r.verdict == "accept"


def test_evaluation_agent_synthetic_bad_rmse_and_low_completeness_needs_more_data():
    r = MockEvaluationAgent().decide({"has_ground_truth": True,
                                        "key_numbers": {"control_point_rmse_m": 40.0, "surface_completeness": 0.01}})
    assert r.verdict == "needs_more_data"


def test_evaluation_agent_key_numbers_always_echoed_back_unchanged():
    nums = {"control_point_rmse_m": 8.0, "surface_completeness": 0.3, "extra_field": 123}
    r = MockEvaluationAgent().decide({"has_ground_truth": True, "key_numbers": nums})
    assert r.key_numbers == nums


def test_evaluation_agent_real_good_psnr_accepts():
    r = MockEvaluationAgent().decide({"has_ground_truth": False,
                                        "key_numbers": {"mean_self_consistency_psnr": 22.0, "confidence_mean": 0.6}})
    assert r.verdict == "accept"


def test_evaluation_agent_real_no_psnr_needs_more_data():
    r = MockEvaluationAgent().decide({"has_ground_truth": False,
                                        "key_numbers": {"mean_self_consistency_psnr": None, "confidence_mean": 0.6}})
    assert r.verdict == "needs_more_data"


def test_evaluation_agent_handles_completely_empty_key_numbers():
    r = MockEvaluationAgent().decide({"has_ground_truth": True, "key_numbers": {}})
    assert r.verdict in ("needs_more_data", "reject")
    assert r.diagnosis


# ---------------------------------------------------------------------------
# MockTrainingControlAgent
# ---------------------------------------------------------------------------

def _training_ctx(history, best_ssim_iter, best_ssim_value, densifying_active=True, checkpoint_every=500):
    return {
        "held_out_history": history,
        "best_held_out": {"ssim": {"iter": best_ssim_iter, "value": best_ssim_value}},
        "checkpoint_every": checkpoint_every,
        "densifying_active": densifying_active,
    }


def test_training_control_too_early_to_judge():
    from layer4_agents.mock_agents import MockTrainingControlAgent
    history = [{"iter": 500, "ssim": 0.4, "n_gaussians": 1000}]
    d = MockTrainingControlAgent().decide(_training_ctx(history, 500, 0.4))
    assert d.action == "continue"


def test_training_control_continues_while_ssim_still_improving():
    from layer4_agents.mock_agents import MockTrainingControlAgent
    history = [{"iter": i, "ssim": 0.3 + i / 10000, "n_gaussians": i * 30} for i in (500, 1000, 1500)]
    d = MockTrainingControlAgent().decide(_training_ctx(history, 1500, 0.45))  # best IS the latest
    assert d.action == "continue"


def test_training_control_stops_densifying_after_stale_checkpoints():
    from layer4_agents.mock_agents import MockTrainingControlAgent
    # Best was at 6500; now at 8000 = 3 stale checkpoints (checkpoint_every=500).
    history = [{"iter": it, "ssim": 0.55, "n_gaussians": it * 38} for it in (7000, 7500, 8000)]
    d = MockTrainingControlAgent().decide(_training_ctx(history, 6500, 0.62, densifying_active=True))
    assert d.action == "stop_densifying"


def test_training_control_does_not_stop_densifying_twice():
    # Once densification is already frozen, the SAME staleness should not re-trigger
    # "stop_densifying" -- densifying_active=False signals it already happened.
    from layer4_agents.mock_agents import MockTrainingControlAgent
    history = [{"iter": it, "ssim": 0.60, "n_gaussians": 300000} for it in (7000, 7500, 8000)]
    d = MockTrainingControlAgent().decide(_training_ctx(history, 6500, 0.62, densifying_active=False))
    assert d.action != "stop_densifying"


def test_training_control_early_stops_after_prolonged_decline_post_freeze():
    from layer4_agents.mock_agents import MockTrainingControlAgent
    # 6 stale checkpoints since the best, densification already frozen, and current SSIM
    # meaningfully below the best (0.55 < 0.62 * 0.97). Needs >= MIN_CHECKPOINTS(3) entries
    # in the history itself, separately from the staleness count.
    history = [{"iter": it, "ssim": 0.55, "n_gaussians": 400000} for it in range(8000, 9501, 500)]
    d = MockTrainingControlAgent().decide(_training_ctx(history, 6500, 0.62, densifying_active=False))
    assert d.action == "early_stop"


def test_training_control_reproduces_documented_pattern_much_earlier_than_manual_retune():
    # Replays STATUS.md's real finding: SSIM peaks at iter 6500 (0.62), then declines
    # monotonically. In the ACTUAL manual process, a human didn't catch this until the
    # full 20000-iteration run finished (27 stale checkpoints). This agent, run at each
    # checkpoint as training progresses, must flag "stop_densifying" well before that.
    from layer4_agents.mock_agents import MockTrainingControlAgent
    agent = MockTrainingControlAgent()
    history = []
    first_stop_densify_iter = None
    for it in range(500, 20001, 500):
        if it <= 6500:
            ssim = 0.15 + (it / 6500) * 0.47  # rising to 0.62 at the peak
        else:
            ssim = max(0.62 - (it - 6500) / 40000, 0.30)  # declining afterward
        history.append({"iter": it, "ssim": ssim, "n_gaussians": it * 40})
        best_iter = max((h["iter"] for h in history if h["ssim"] >= max(x["ssim"] for x in history)), default=None)
        best_value = max(h["ssim"] for h in history)
        d = agent.decide(_training_ctx(history, best_iter, best_value, densifying_active=True))
        if d.action == "stop_densifying" and first_stop_densify_iter is None:
            first_stop_densify_iter = it

    assert first_stop_densify_iter is not None
    # 6500 (the true peak) + 3 stale checkpoints * 500 = 8000 at the latest.
    assert first_stop_densify_iter <= 8000
    # And unambiguously earlier than the 20000-iteration point a human actually caught it at.
    assert first_stop_densify_iter < 20000
