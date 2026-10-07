"""
Unit tests for the Layer-3 CPU-testable path (layer3_gpu/python/confidence_gaussian_model.py).
Skips cleanly if torch is not installed -- Layer 3 CPU-testing is a bonus validation,
not a Layer-1 CPU-sandbox requirement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layer3_gpu", "python"))

import pytest
torch = pytest.importorskip("torch")

from confidence_gaussian_model import (GaussianConfidenceStats, ConfidenceAwareGaussianModel,
                                        simplified_splat_cpu)
from confidence_propagation import (ObservationConfidencePropagator, PropagationConfig,
                                     optimizer_state_struggle_signal)


def test_confidence_stats_single_observation_capped():
    stats = GaussianConfidenceStats(n=2)
    idx = torch.tensor([0])
    stats.update(idx, torch.tensor([0.2]), torch.tensor([0.95]), 0.95, 0.95)
    conf = stats.confidence(single_view_cap=0.55)
    assert conf[0].item() <= 0.55 + 1e-6


def test_confidence_stats_multi_view_exceeds_cap_when_warranted():
    stats = GaussianConfidenceStats(n=1)
    idx = torch.tensor([0, 0, 0, 0])
    stats.update(idx, torch.tensor([0.05, 0.15, 0.25, 0.35]), torch.tensor([0.85, 0.85, 0.85, 0.85]), 0.9, 0.9)
    conf = stats.confidence(single_view_cap=0.55)
    assert conf[0].item() > 0.55


def test_splatter_gradient_flow_reduces_loss():
    torch.manual_seed(1)
    H, W, N = 16, 16, 20
    target = torch.zeros(H, W, 3)
    target[6:10, 6:10, 0] = 1.0  # a flat red square target

    init_pos = torch.rand(N, 2) * torch.tensor([W, H])
    init_col = torch.rand(N, 3) * 0.3
    model = ConfidenceAwareGaussianModel(init_pos, init_col)
    opt = torch.optim.Adam(model.parameters(), lr=0.2)

    losses = []
    for _ in range(80):
        opt.zero_grad()
        rendered, _ = model.render(H, W)
        loss = torch.mean((rendered - target) ** 2)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_densify_prune_gating_respects_single_view_cap():
    torch.manual_seed(2)
    N = 5
    model = ConfidenceAwareGaussianModel(torch.rand(N, 2) * 16, torch.rand(N, 3))
    with torch.no_grad():
        model.opacity_logits[:] = 3.0  # all high opacity
        model.prior_confidence[:] = 0.8

    idx_multi = torch.tensor([0, 0, 0])
    model.stats.update(idx_multi, torch.tensor([0.0, 0.1, 0.2]), torch.tensor([0.9, 0.9, 0.9]), 0.9, 0.9)
    idx_single = torch.tensor([1])
    model.stats.update(idx_single, torch.tensor([0.1]), torch.tensor([0.9]), 0.9, 0.9)

    grad_accum = torch.full((N,), 0.001)
    prune_mask, densify_mask, conf = model.densify_and_prune(grad_accum)
    assert bool(densify_mask[0]) is True
    assert bool(densify_mask[1]) is False


def test_optimizer_struggle_signal_cannot_bypass_observation_gate():
    """The key architectural guarantee of the 3-signal design (see
    confidence_propagation.py's module docstring): an OSAD-inspired
    optimizer-struggle signal can ADD densify candidates, but only among
    Gaussians that already pass the observation-confidence gate -- it must
    never override that gate, however large the struggle signal is."""
    torch.manual_seed(3)
    N = 3
    model = ConfidenceAwareGaussianModel(torch.rand(N, 2) * 16, torch.rand(N, 3))
    with torch.no_grad():
        model.opacity_logits[:] = 3.0
        model.prior_confidence[0] = 0.85  # well observed
        model.prior_confidence[1] = 0.05  # poorly observed -- single-pass blind spot
        model.prior_confidence[2] = 0.85

    grad_accum = torch.zeros(N)  # no gradient-based trigger for anyone
    huge_struggle_on_blindspot = torch.tensor([0.0, 50.0, 0.0])
    _, densify_mask, _ = model.densify_and_prune(grad_accum, optimizer_struggle_signal=huge_struggle_on_blindspot,
                                                  struggle_thresh=3.0)
    assert bool(densify_mask[1]) is False, "struggle signal must not bypass the observation-confidence gate"

    huge_struggle_on_well_observed = torch.tensor([50.0, 0.0, 0.0])
    _, densify_mask2, _ = model.densify_and_prune(grad_accum, optimizer_struggle_signal=huge_struggle_on_well_observed,
                                                   struggle_thresh=3.0)
    assert bool(densify_mask2[0]) is True, "the same signal on a well-observed Gaussian should correctly trigger"


def test_propagator_spatial_diffusion_pulls_up_neighbor():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])  # same voxel
    prop = ObservationConfidencePropagator(2, PropagationConfig(gamma=0.5, voxel_size=1.0, spatial_beta=0.5))
    out = prop.update(xyz, torch.tensor([0.9, 0.0]))
    assert out[1] > 0.0  # pulled up by its confident neighbor


def test_propagator_resize_after_prune_and_split():
    prop = ObservationConfidencePropagator(4, PropagationConfig())
    prop.update(torch.rand(4, 3), torch.tensor([0.1, 0.2, 0.3, 0.4]))
    keep_mask = torch.tensor([True, False, True, False])
    prop.resize(keep_mask=keep_mask, n_new=2)
    assert prop.propagated.shape[0] == 4  # 2 kept + 2 new
    assert prop.propagated[-1].item() == 0.0


def test_optimizer_struggle_signal_higher_for_sustained_push_than_oscillation():
    positions = torch.nn.Parameter(torch.zeros(2, 3))
    opt = torch.optim.Adam([positions], lr=0.1)
    for step in range(20):
        opt.zero_grad()
        grad = torch.zeros(2, 3)
        grad[0] = 1.0
        grad[1] = 1.0 if step % 2 == 0 else -1.0
        positions.grad = grad
        opt.step()
    signal = optimizer_state_struggle_signal(opt, positions)
    assert signal[0] > signal[1]


def test_optimizer_struggle_signal_zero_before_warmup():
    positions = torch.nn.Parameter(torch.zeros(3, 3))
    opt = torch.optim.Adam([positions], lr=0.1)
    signal = optimizer_state_struggle_signal(opt, positions)
    assert torch.all(signal == 0.0)
