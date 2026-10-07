"""
confidence_propagation.py
LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, CPU-tested,
CPU-executed (see __main__ and tests/test_layer3_confidence_propagation.py).
NOT GPU-executed (no CUDA device in Environment A).

Two additions to the core novelty (confidence_gaussian_model.py), both
DESIGN-INSPIRED by mechanisms in the author's own prior work, FastGS-QADS
(specifically its STCP and OSAD modules), reimplemented independently from
scratch for a different signal and a different role in this project -- see
the module-level comparison in each class/function docstring below. No code
from FastGS-QADS is copied here; only the underlying idea and, in one case
(OSAD's Adam-moment ratio), a piece of open optimizer mathematics that
isn't anyone's IP to begin with.

------------------------------------------------------------------------
1. ObservationConfidencePropagator (STCP-INSPIRED, DIFFERENT SIGNAL)
------------------------------------------------------------------------
FastGS-QADS's STCP stabilizes a noisy PHOTOMETRIC-FIT score (sampled from
K=10 random cameras each densification round) via temporal EMA + spatial
voxel diffusion. This project's confidence.py signal is not noisy in that
sense -- it's an OBSERVATION-COVERAGE score accumulated once from the
evidence available at initialization. But at GPU scale, observation
confidence is no longer static either: as training proceeds, MORE evidence
can arrive (a Gaussian split from a well-observed parent should inherit
strong confidence; a relocated Gaussian, see MVCR-inspired notes in
confidence_gaussian_model.py's densify_and_prune docstring, starts with NO
evidence and must earn confidence over subsequent rounds). So the same
temporal-EMA + spatial-voxel-diffusion PATTERN genuinely applies here, for
a different reason: not to denoise a single-round photometric sample, but
to let observation evidence accumulate smoothly and propagate to
geometrically nearby Gaussians (a facade partially seen by one frame can
be corroborated by a barely-overlapping neighboring Gaussian that a
DIFFERENT frame saw well).

------------------------------------------------------------------------
2. optimizer_state_struggle_signal() (OSAD-INSPIRED)
------------------------------------------------------------------------
Same core mathematical relationship as FastGS-QADS's OSAD (Adam's
||m||/||sqrt(v)|| directional-bias ratio for the position parameter) --
this ratio is a property of the Adam optimizer itself, not proprietary to
any one implementation. What's original here is ITS ROLE in this project's
architecture: OSAD in FastGS-QADS ORs directly with the gradient-based
densify trigger (nothing in that codebase gates it further). Here, this
signal is explicitly SUBORDINATE to observation confidence -- see
`densify_and_prune`'s updated logic in confidence_gaussian_model.py: an
optimizer-struggle signal can only ADD densify candidates WITHIN
observation-confidence-passing Gaussians, never bypass that gate. The
reasoning: an oscillating/non-converging optimizer in a POORLY observed
region is just as (or more) likely to mean "not enough real evidence to
converge at all" as "needs more Gaussian capacity" -- densifying there
would spend budget compounding an already-unreliable region. This
project's whole thesis is that single-pass coverage gaps must be respected,
not optimized around; FastGS-QADS had no such constraint to design for
(its target scenes have normal multi-view SfM coverage everywhere).
"""
from __future__ import annotations

import dataclasses
import torch


# --------------------------------------------------------------- STCP-inspired --

@dataclasses.dataclass
class PropagationConfig:
    gamma: float = 0.85          # temporal EMA decay; higher = smoother/slower to react
    voxel_size: float = 1.0      # spatial diffusion voxel edge length, scene units
    spatial_beta: float = 0.3    # blend weight toward neighborhood mean, 0=off
    new_gaussian_confidence: float = 0.0  # confidence assigned to a freshly split/relocated
                                           # Gaussian with no observation history of its own yet


class ObservationConfidencePropagator:
    """
    Maintains a persistent, propagated observation-confidence buffer across
    training rounds (densification intervals). Call `update()` once per
    round with the latest raw per-Gaussian confidence (e.g. from
    `ConfidenceAwareGaussianModel.confidence()`); use `.propagated` for
    densify/prune decisions instead of the raw value.
    """

    def __init__(self, n: int, cfg: PropagationConfig = None, device="cpu"):
        self.cfg = cfg or PropagationConfig()
        self.propagated = torch.zeros(n, device=device)
        self._initialized = torch.zeros(n, dtype=torch.bool, device=device)

    def _voxel_key(self, xyz: torch.Tensor) -> torch.Tensor:
        return torch.floor(xyz / self.cfg.voxel_size).long()

    def _spatial_diffuse(self, xyz: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Blend each Gaussian's value with the mean of every Gaussian sharing its
        coarse voxel cell -- a single vectorized scatter/gather, O(N log N), no
        explicit neighbor search. Same asymptotic approach as voxel-merging in
        prototype_point_repr.py, applied here as a continuous smoothing step
        instead of a one-shot fusion."""
        if xyz.shape[0] == 0 or self.cfg.spatial_beta <= 0.0:
            return values
        keys = self._voxel_key(xyz)
        uniq, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
        voxel_sum = torch.zeros(uniq.shape[0], device=values.device, dtype=values.dtype)
        voxel_sum.scatter_add_(0, inverse, values)
        voxel_mean = voxel_sum / counts.to(values.dtype)
        neighborhood_value = voxel_mean[inverse]
        beta = self.cfg.spatial_beta
        return (1.0 - beta) * values + beta * neighborhood_value

    def update(self, xyz: torch.Tensor, raw_confidence: torch.Tensor) -> torch.Tensor:
        """One round: temporal EMA blended with the previous propagated value,
        then diffused spatially. Returns the updated propagated confidence
        (also stored in `self.propagated`)."""
        raw_confidence = raw_confidence.to(self.propagated.device)
        if raw_confidence.shape[0] != self.propagated.shape[0]:
            raise ValueError(
                f"Size mismatch ({raw_confidence.shape[0]} vs {self.propagated.shape[0]}) -- "
                f"call `resize()` immediately after any densify/prune/relocate that changes "
                f"the Gaussian count, before the next `update()`."
            )
        ema = torch.where(self._initialized,
                           self.cfg.gamma * self.propagated + (1 - self.cfg.gamma) * raw_confidence,
                           raw_confidence)  # first observation for a Gaussian: no history to blend with
        self._initialized |= True
        self.propagated = self._spatial_diffuse(xyz, ema)
        return self.propagated

    def resize(self, keep_mask: torch.Tensor = None, n_new: int = 0, new_confidence: torch.Tensor = None):
        """Call immediately after prune (keep_mask) and/or densify/relocate (n_new new
        Gaussians appended) so the buffer stays index-aligned with the Gaussian model.
        New Gaussians default to `new_gaussian_confidence` (deliberately low/zero --
        a relocated or newly split Gaussian has not yet earned confidence of its own;
        see module docstring) UNLESS `new_confidence` is supplied, which lets a caller
        explicitly INHERIT a value instead (e.g. gaussian_model.py's clone/split: a
        clone is the same evidence covering a smaller region, not new evidence, so it
        should inherit its parent's propagated confidence rather than restart at zero)."""
        if keep_mask is not None:
            self.propagated = self.propagated[keep_mask]
            self._initialized = self._initialized[keep_mask]
        if n_new > 0:
            if new_confidence is not None:
                pad_val = new_confidence.to(self.propagated.device)
                if pad_val.shape[0] != n_new:
                    raise ValueError(f"new_confidence has {pad_val.shape[0]} entries, expected {n_new}")
            else:
                pad_val = torch.full((n_new,), self.cfg.new_gaussian_confidence, device=self.propagated.device)
            pad_init = torch.zeros(n_new, dtype=torch.bool, device=self.propagated.device)
            self.propagated = torch.cat([self.propagated, pad_val])
            self._initialized = torch.cat([self._initialized, pad_init])


# --------------------------------------------------------------- OSAD-inspired --

def optimizer_state_struggle_signal(optimizer: torch.optim.Adam, param: torch.nn.Parameter,
                                     eps: float = 1e-8) -> torch.Tensor:
    """
    Per-row ||m|| / ||sqrt(v)|| for `param`'s Adam state -- see module docstring
    for the derivation and, importantly, how this project uses it differently
    from its inspiration. Returns a zero tensor (not None) before Adam has
    warmed up, so callers can use it unconditionally without a None-check.
    """
    if optimizer is None or param not in optimizer.state:
        return torch.zeros(param.shape[0], device=param.device)
    state = optimizer.state[param]
    if "exp_avg" not in state or "exp_avg_sq" not in state:
        return torch.zeros(param.shape[0], device=param.device)
    m = state["exp_avg"].detach()
    v = state["exp_avg_sq"].detach()
    flat_dims = tuple(range(1, m.dim())) if m.dim() > 1 else ()
    m_norm = torch.linalg.vector_norm(m, dim=flat_dims) if flat_dims else m.abs()
    v_norm = torch.linalg.vector_norm(v.sqrt(), dim=flat_dims) if flat_dims else v.sqrt().abs()
    return m_norm / (v_norm + eps)


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Testing ObservationConfidencePropagator (STCP-inspired, CPU, real execution)...")

    n = 8
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0],   # voxel A: 2 close Gaussians
                         [10.0, 0.0, 0.0], [10.05, 0.0, 0.0],  # voxel B: 2 close Gaussians
                         [20.0, 0.0, 0.0], [40.0, 0.0, 0.0],
                         [60.0, 0.0, 0.0], [80.0, 0.0, 0.0]])
    prop = ObservationConfidencePropagator(n, PropagationConfig(gamma=0.5, voxel_size=1.0, spatial_beta=0.5))

    # Round 1: one Gaussian in voxel A has strong evidence (0.9), its close neighbor has none (0.0) yet.
    raw1 = torch.tensor([0.9, 0.0, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3])
    out1 = prop.update(xyz, raw1)
    print(f"  round 1: voxel-A pair confidences = {out1[0]:.3f}, {out1[1]:.3f} "
          f"(neighbor pulled up from 0.0 by spatial diffusion: {'PASS' if out1[1] > 0.0 else 'FAIL'})")
    assert out1[1] > raw1[1], "spatial propagation should pull the under-evidenced neighbor's confidence up"

    # Round 2: same evidence again -- EMA should stabilize, not oscillate back to raw.
    out2 = prop.update(xyz, raw1)
    print(f"  round 2: voxel-A pair confidences = {out2[0]:.3f}, {out2[1]:.3f}")

    print("\nTesting resize() after a simulated prune+split...")
    keep_mask = torch.tensor([True, True, True, True, True, True, False, False])  # prune last 2
    prop.resize(keep_mask=keep_mask, n_new=3)  # 3 new Gaussians from a split
    print(f"  buffer size after prune(6 kept)+split(+3): {prop.propagated.shape[0]} (expected 9)")
    assert prop.propagated.shape[0] == 9
    assert prop.propagated[-1].item() == 0.0, "new Gaussians should start at new_gaussian_confidence"
    print("PASS")

    print("\nTesting optimizer_state_struggle_signal (OSAD-inspired, CPU, real Adam execution)...")
    # A parameter that oscillates (high m/v ratio expected) vs one that's converged (low ratio).
    positions = torch.nn.Parameter(torch.zeros(2, 3))
    opt = torch.optim.Adam([positions], lr=0.1)
    for step in range(20):
        opt.zero_grad()
        # Gaussian 0: gradient keeps pushing the same direction (+1) every step -> should NOT trigger
        #             (consistent gradient = m and v both grow together, ratio stays moderate)
        # Gaussian 1: gradient sign flips each step -> oscillation -> should show HIGH m/v is wrong intuition;
        #             actually oscillation means m (mean) stays SMALL while v (mean of squares) stays large,
        #             so the ratio is LOW for oscillation and HIGH for sustained one-directional push.
        grad = torch.zeros(2, 3)
        grad[0] = 1.0                                  # sustained push, same direction every step
        grad[1] = 1.0 if step % 2 == 0 else -1.0        # oscillating sign
        positions.grad = grad
        opt.step()
    signal = optimizer_state_struggle_signal(opt, positions)
    print(f"  sustained-push Gaussian signal: {signal[0]:.3f}")
    print(f"  oscillating Gaussian signal:    {signal[1]:.3f}")
    assert signal[0] > signal[1], (
        "a Gaussian being pushed consistently in one direction should show a HIGHER "
        "directional-bias ratio than one whose gradient sign is flipping every step"
    )
    print("PASS: signal correctly distinguishes sustained directional push from oscillation.")
