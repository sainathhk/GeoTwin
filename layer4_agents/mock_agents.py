"""
mock_agents.py
LAYER 4 (AGENTS) -- STATUS: implemented, CPU-tested

Plays exactly the role a synthetic dataset plays for Layer 1, or a CPU
prototype plays for a Layer 3 CUDA kernel: a deterministic stand-in that is
cheap to run everywhere and lets the surrounding machinery (here,
`orchestrator.py`'s retry loop) be fully tested in THIS sandbox, with no
network and no API key. `llm_agents.py` implements the exact same
`IFrameAgent`/`IReconstructionAgent`/`IEvaluationAgent` contracts and is a
drop-in replacement -- swapping one for the other changes nothing else.

These rule tables are not trying to be as good as an LLM at reading a
situation; they exist to be predictable and inspectable so orchestration
bugs show up here, not three layers away inside a model's hidden reasoning.
Thresholds are commented with where they came from -- either read directly
off a real run in this repo, or a plainly-labeled placeholder -- so future
recalibration has somewhere to start, exactly like frame_overlap.py's own
"tuned against Environment A, re-check on real footage" notes.

ON THE RECONSTRUCTION AGENT'S RETRY POLICY SPECIFICALLY: it deliberately
does NOT retry by loosening `prune_below`/`densify_above` to keep more
low-confidence points. This repo's whole novelty claim
(docs/NOVELTY_CANDIDATES.md; the confidence-gating Spearman result in
README.md) is that pruning low-confidence points on purpose, even at a
completeness cost, is the honest thing to do. An agent that quietly relaxed
the gate to make completeness numbers look better would be undermining the
exact thing it's supposed to be helping demonstrate. So: a `retry` here only
ever adjusts things that address a genuine CONFIGURATION mismatch (depth
search range/resolution, neighbor window) -- never the confidence bar
itself. When the diagnosis looks like real scene difficulty rather than a
fixable mismatch, the correct action is `flag_and_accept`, not `retry`.
"""
from __future__ import annotations

from typing import Any, Dict

from .interfaces import IFrameAgent, IReconstructionAgent, IEvaluationAgent, ITrainingControlAgent
from .schemas import (
    FrameAgentDecision, ReconstructionAgentDecision, EvaluationAgentReport, TrainingControlDecision,
    KEEP_FRACTION_BOUNDS, MIN_OVERALL_BOUNDS, RECON_BOUNDS, RECON_INT_BOUNDS,
)


class MockFrameAgent(IFrameAgent):
    def decide(self, context: Dict[str, Any]) -> FrameAgentDecision:
        q = context.get("quality", {})
        overlap = context.get("raw_capture_overlap", {})
        keep_fraction = context.get("current_keep_fraction", 0.75)
        min_overall = context.get("current_min_overall", 0.15)
        flags, reasons = [], []

        n_pairs = max(1, overlap.get("n_pairs", 1))
        n_gap = overlap.get("n_gap_risk", 0)
        n_dup = overlap.get("n_duplicate_risk", 0)
        frac_unusable = q.get("fraction_below_0.15") or 0.0

        # Threshold read off this repo's own small-scene test runs (frame_overlap on a
        # 30-frame synthetic clip): >10% gap-risk pairs is already a lot for a single
        # continuous pass and is worth protecting against by keeping more frames.
        if n_gap / n_pairs > 0.10:
            new_kf = min(KEEP_FRACTION_BOUNDS[1], keep_fraction + 0.15)
            flags.append(f"{n_gap}/{n_pairs} consecutive raw-frame pairs are gap_risk "
                          f"(low overlap) -- raised keep_fraction {keep_fraction:.2f} -> {new_kf:.2f} "
                          f"to reduce the chance quality-filtering opens a coverage gap")
            reasons.append("coverage-gap risk in raw capture")
            keep_fraction = new_kf
        elif n_dup / n_pairs > 0.30:
            new_kf = max(KEEP_FRACTION_BOUNDS[0], keep_fraction - 0.10)
            flags.append(f"{n_dup}/{n_pairs} pairs look like near-duplicates (hover/stall) -- "
                          f"lowered keep_fraction {keep_fraction:.2f} -> {new_kf:.2f}, "
                          f"redundant frames are safe to thin")
            reasons.append("high near-duplicate rate")
            keep_fraction = new_kf

        # Placeholder threshold (not yet calibrated against a real footage run): flag
        # rather than silently accept when a large fraction of frames fail the floor.
        if frac_unusable > 0.40:
            new_mo = max(MIN_OVERALL_BOUNDS[0], min_overall - 0.05)
            flags.append(f"{frac_unusable:.0%} of frames are below the quality floor -- "
                          f"lowered min_overall {min_overall:.2f} -> {new_mo:.2f} to avoid "
                          f"starving the reconstruction of frames")
            reasons.append("large fraction of frames below quality floor")
            min_overall = new_mo

        if not reasons:
            reasons.append("quality and overlap both within normal range; keeping current settings")

        return FrameAgentDecision(keep_fraction=keep_fraction, min_overall=min_overall,
                                    reasoning="; ".join(reasons), flags=flags)


class MockReconstructionAgent(IReconstructionAgent):
    # Read off this repo's own synthetic runs (extent=25-40 test scenes): ~30 points per
    # kept frame is a reasonable "depth estimation is basically working" floor at this
    # resolution; well below that usually means the depth search range or resolution
    # doesn't cover the scene, not that the scene is inherently unreconstructable.
    MIN_POINTS_PER_KEPT_FRAME = 10.0

    def decide(self, context: Dict[str, Any]) -> ReconstructionAgentDecision:
        cfg = context.get("current_config", {})
        pts = context.get("points", {})
        frames = context.get("frames", {})
        spatial = context.get("spatial_diagnostic", {})
        history = context.get("attempt_history", []) or []

        n_kept = max(1, frames.get("n_kept", 1))
        pts_per_frame = pts.get("n_raw_fused", 0) / n_kept
        weak_regions = spatial.get("weak_regions", [])
        low_frac = pts.get("low_confidence_fraction_kept") or 0.0

        # Give up on retrying once we've already tried twice -- diminishing returns, and
        # an unbounded loop is exactly what the "Agent -> Decision -> Tool -> Result ->
        # Agent" principle in docs/AGENTIC_ARCHITECTURE.md means to prevent.
        retries_so_far = sum(1 for h in history if h.get("action") == "retry")
        if retries_so_far >= 2:
            if low_frac > 0.4 or weak_regions:
                return ReconstructionAgentDecision(
                    action="flag_and_accept", agent_confidence=0.6,
                    reasoning=f"already retried {retries_so_far} times with limited change; "
                               f"remaining low confidence ({low_frac:.0%}) looks like real scene "
                               f"difficulty rather than a fixable setting -- accepting and flagging "
                               f"the weak region(s) instead of retrying further",
                )
            return ReconstructionAgentDecision(action="accept", agent_confidence=0.7,
                                                 reasoning="metrics acceptable after prior retries")

        if pts_per_frame < self.MIN_POINTS_PER_KEPT_FRAME:
            depth_min = cfg.get("depth_min")
            depth_max = cfg.get("depth_max")
            new_depth_min = max(RECON_BOUNDS["depth_min"][0], depth_min * 0.7) if depth_min else None
            new_depth_max = min(RECON_BOUNDS["depth_max"][1], depth_max * 1.3) if depth_max else None
            n_depths = cfg.get("n_depths")
            new_n_depths = min(RECON_INT_BOUNDS["n_depths"][1], int(n_depths * 1.5)) if n_depths else None
            return ReconstructionAgentDecision(
                action="retry", depth_min=new_depth_min, depth_max=new_depth_max,
                n_depths=new_n_depths, agent_confidence=0.55,
                reasoning=f"only {pts_per_frame:.1f} raw fused points per kept frame "
                           f"(floor: {self.MIN_POINTS_PER_KEPT_FRAME}) -- widening the depth search "
                           f"range and increasing depth samples rather than assuming the scene "
                           f"itself is unreconstructable",
            )

        if weak_regions and retries_so_far == 0:
            depth_window = cfg.get("depth_window")
            new_window = min(RECON_INT_BOUNDS["depth_window"][1], depth_window + 1) if depth_window else None
            region = weak_regions[0]
            return ReconstructionAgentDecision(
                action="retry", depth_window=new_window, agent_confidence=0.5,
                reasoning=f"a {region['direction']} region has {region['low_confidence_fraction']:.0%} "
                           f"low-confidence points ({region['n_points']} points there) -- increasing "
                           f"depth_window {depth_window} -> {new_window} in case the kept-frame "
                           f"spacing there is wider than plane-sweep's neighbor assumption expects",
            )

        if low_frac > 0.5:
            return ReconstructionAgentDecision(
                action="flag_and_accept", agent_confidence=0.6,
                reasoning=f"{low_frac:.0%} of kept points are Low-confidence with no specific "
                           f"fixable cause identified (point density and regional spread both look "
                           f"normal) -- accepting as a genuine coverage limitation rather than "
                           f"retrying blindly",
            )

        return ReconstructionAgentDecision(action="accept", agent_confidence=0.8,
                                             reasoning="point density, confidence distribution, and "
                                                        "spatial coverage all within normal range")


class MockEvaluationAgent(IEvaluationAgent):
    # Thresholds below are read directly off this repo's own committed reference run
    # (layer1_cpu_sandbox/outputs/run1/ablation_table.md): confidence gating there moved
    # point-to-point RMSE from ~43.9m to ~11.1m at completeness ~0.03-0.4 depending on
    # variant. These are NOT universal cutoffs -- they're this repo's own observed range,
    # written down here so the next real run can sanity-check against them and this
    # comment can be updated with better numbers as they accumulate.
    GOOD_RMSE_M = 15.0
    LOW_COMPLETENESS = 0.02
    GOOD_SELF_CONSISTENCY_PSNR = 18.0

    def decide(self, context: Dict[str, Any]) -> EvaluationAgentReport:
        nums = context.get("key_numbers", {})
        if context.get("has_ground_truth"):
            return self._decide_synthetic(nums)
        return self._decide_real(nums)

    def _decide_synthetic(self, n: dict) -> EvaluationAgentReport:
        rmse = n.get("control_point_rmse_m")
        completeness = n.get("surface_completeness")
        parts = []
        verdict = "needs_more_data"

        if rmse is not None:
            parts.append(f"control-point RMSE is {rmse:.1f}m "
                          f"({'within' if rmse <= self.GOOD_RMSE_M else 'above'} "
                          f"this repo's own {self.GOOD_RMSE_M}m reference band)")
        if completeness is not None:
            parts.append(f"surface completeness is {completeness:.1%}")

        if rmse is not None and rmse <= self.GOOD_RMSE_M:
            verdict = "accept"
        elif completeness is not None and completeness < self.LOW_COMPLETENESS and \
                (rmse is None or rmse > self.GOOD_RMSE_M):
            verdict = "needs_more_data"
            parts.append("both metric accuracy and completeness are weak -- more/better "
                          "coverage of the scene would likely help more than another retry")
        elif rmse is not None:
            verdict = "accept" if rmse <= self.GOOD_RMSE_M * 1.5 else "needs_more_data"

        diagnosis = "; ".join(parts) if parts else "insufficient metrics computed to form a verdict"
        next_step = {
            "accept": "no further action needed for this pass",
            "needs_more_data": "consider a denser or repeated flight pass over the weak area(s), "
                                 "or accept the current honest completeness figure if a re-flight "
                                 "isn't possible",
            "reject": "",
        }[verdict]
        return EvaluationAgentReport(verdict=verdict, diagnosis=diagnosis,
                                       key_numbers=n, suggested_next_step=next_step)

    def _decide_real(self, n: dict) -> EvaluationAgentReport:
        psnr = n.get("mean_self_consistency_psnr")
        conf_mean = n.get("confidence_mean")
        parts = []
        if psnr is not None:
            parts.append(f"mean self-consistency PSNR is {psnr:.1f}dB")
        if conf_mean is not None:
            parts.append(f"mean confidence over kept points is {conf_mean:.2f}")

        if psnr is not None and psnr >= self.GOOD_SELF_CONSISTENCY_PSNR:
            verdict = "accept"
        elif psnr is None:
            verdict = "needs_more_data"
            parts.append("no self-consistency views were available to check against")
        else:
            verdict = "needs_more_data"
            parts.append(f"below this repo's {self.GOOD_SELF_CONSISTENCY_PSNR}dB reference for "
                          f"held-out view agreement")

        diagnosis = "; ".join(parts) if parts else "insufficient metrics computed to form a verdict"
        next_step = ("no further action needed" if verdict == "accept" else
                     "consider re-flying the weak section, or reviewing the flagged low-confidence "
                     "regions before treating this reconstruction as final -- there is no ground "
                     "truth for real footage, so this verdict is based on self-consistency only")
        return EvaluationAgentReport(verdict=verdict, diagnosis=diagnosis,
                                       key_numbers=n, suggested_next_step=next_step)


class MockTrainingControlAgent(ITrainingControlAgent):
    # Calibrated against this repo's own documented real finding (STATUS.md): a
    # 20,000-iteration run's held-out SSIM peaked at iter 6500 (246K Gaussians) and
    # declined monotonically through iter 20000 (829K Gaussians) before a human noticed
    # (by reading the console log after the full run finished) and retuned
    # densify_until_iter for the NEXT run. These thresholds are deliberately much more
    # impatient than that took: "3 stale checkpoints" is 1500 iterations at the default
    # checkpoint_every=500, not the 13500 iterations (27 checkpoints) it actually took.
    MIN_CHECKPOINTS = 3
    STOP_DENSIFY_STALE_CHECKPOINTS = 3
    EARLY_STOP_STALE_CHECKPOINTS = 6
    DECLINE_MARGIN = 0.97  # current SSIM below best * this counts as "declining", not "flat"

    def decide(self, context: Dict[str, Any]) -> TrainingControlDecision:
        history = context.get("held_out_history", [])
        if len(history) < self.MIN_CHECKPOINTS:
            return TrainingControlDecision(
                action="continue", agent_confidence=0.3,
                reasoning=f"only {len(history)} held-out checkpoint(s) so far -- too early to judge a trend",
            )

        best_ssim = context.get("best_held_out", {}).get("ssim", {})
        best_iter, best_value = best_ssim.get("iter"), best_ssim.get("value")
        current = history[-1]
        checkpoint_every = max(1, context.get("checkpoint_every", 500))

        if best_iter is None or best_value is None or current.get("ssim") is None:
            return TrainingControlDecision(action="continue", agent_confidence=0.3,
                                             reasoning="held-out SSIM not available yet; continuing")

        stale_checkpoints = (current["iter"] - best_iter) / checkpoint_every
        densifying_active = context.get("densifying_active", True)

        if (stale_checkpoints >= self.EARLY_STOP_STALE_CHECKPOINTS and not densifying_active
                and current["ssim"] < best_value * self.DECLINE_MARGIN):
            return TrainingControlDecision(
                action="early_stop", agent_confidence=0.6,
                reasoning=f"held-out SSIM has not improved in {int(stale_checkpoints)} checkpoints since "
                           f"iter {best_iter} (best={best_value:.3f}), densification is already frozen, and "
                           f"the current value ({current['ssim']:.3f}) is meaningfully below that best -- "
                           f"further iterations are unlikely to help and continue to risk overfitting "
                           f"(compare this repo's own iter-6500-peak-then-decline finding in STATUS.md)",
            )

        if stale_checkpoints >= self.STOP_DENSIFY_STALE_CHECKPOINTS and densifying_active:
            return TrainingControlDecision(
                action="stop_densifying", agent_confidence=0.55,
                reasoning=f"held-out SSIM has not improved in {int(stale_checkpoints)} checkpoints since "
                           f"iter {best_iter} (best={best_value:.3f}); now at {current.get('n_gaussians')} "
                           f"Gaussians -- freezing further densification rather than continuing to grow "
                           f"the model past what these training views can constrain",
            )

        return TrainingControlDecision(
            action="continue", agent_confidence=0.7,
            reasoning="held-out SSIM is still improving or too recently at its best to call a trend yet",
        )
