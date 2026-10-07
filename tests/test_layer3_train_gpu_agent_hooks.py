"""Tests for layer3_gpu/python/train_gpu.py's Layer-4 agent-hook additions
(apply_training_agent_at_checkpoint specifically -- the one new piece of integration
logic that couldn't be tested by testing layer4_agents in isolation).

train_gpu.py imports torch + this project's own CUDA-adjacent modules
(gaussian_model, confidence_gaussian_model, confidence_propagation, losses) at module
level. None of that is available in a CPU-only sandbox (see STATUS.md), and
apply_training_agent_at_checkpoint itself uses none of those names -- it's plain
Python + calls into the already-tested layer4_agents package. So: if the real
modules are importable (e.g. running this suite in the actual Colab GPU environment),
this uses them, untouched; otherwise it installs minimal stand-ins ONLY for the
specific names actually missing, purely so `import train_gpu` succeeds far enough to
reach the one function this file tests. This never masks a real bug in
gaussian_model.py etc. (a genuine SyntaxError there is NOT an ImportError and would
still surface).

Import is done through a scoped helper that SNAPSHOTS and RESTORES sys.modules
afterward -- this file must never leave a stub sitting in sys.modules for whichever
test file pytest happens to collect next in the same process. In a real GPU
environment where torch etc. are genuinely installed, a later test file (e.g.
tests/test_layer3_gaussian_model.py) must see the real thing, not a leftover stub
from this one.
"""
import importlib
import sys
import types
import os
import contextlib


_STUB_SPECS = {
    "torch": {"Tensor": object, "cuda": types.SimpleNamespace(is_available=lambda: False)},
    "gaussian_model": {"GaussianModel": object, "GaussianTrainingConfig": object},
    "confidence_gaussian_model": {"GaussianConfidenceStats": object},
    "confidence_propagation": {
        "PropagationConfig": object, "ObservationConfidencePropagator": object,
        "optimizer_state_struggle_signal": lambda *a, **k: None,
    },
    "losses": {"combined_loss": lambda *a, **k: None},
}


@contextlib.contextmanager
def _scoped_train_gpu_import():
    """Yields the train_gpu module, real dependencies preferred, minimal stubs only for
    whatever's actually missing -- and restores sys.modules to exactly how it looked
    before, on the way out, so this test file never affects any other."""
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_THIS_DIR)
    added_paths = []
    for p in (os.path.join(_REPO_ROOT, "layer3_gpu", "python"), _REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
            added_paths.append(p)

    snapshot = {name: sys.modules.get(name) for name in list(_STUB_SPECS) + ["train_gpu"]}
    installed_stub_for = []
    try:
        for name, attrs in _STUB_SPECS.items():
            if name in sys.modules:
                continue
            try:
                importlib.import_module(name)  # real one available -- use it, don't stub
            except ImportError:
                stub = types.ModuleType(name)
                for k, v in attrs.items():
                    setattr(stub, k, v)
                sys.modules[name] = stub
                installed_stub_for.append(name)

        sys.modules.pop("train_gpu", None)  # force a fresh import under current sys.modules state
        import train_gpu
        yield train_gpu
    finally:
        for name, original in snapshot.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        for p in added_paths:
            if p in sys.path:
                sys.path.remove(p)


from layer4_agents.mock_agents import MockTrainingControlAgent
from layer4_agents.schemas import TrainingControlDecision


def _history(peak_iter, peak_value, current_iter, current_value, checkpoint_every=500):
    """A minimal 3-checkpoint held_out_history: rising to a peak, then one later point."""
    return [
        {"iter": max(500, peak_iter - checkpoint_every), "ssim": peak_value - 0.1, "n_gaussians": 1000},
        {"iter": peak_iter, "ssim": peak_value, "n_gaussians": 2000},
        {"iter": current_iter, "ssim": current_value, "n_gaussians": 3000},
    ]


def test_apply_training_agent_skips_call_below_min_checkpoints():
    class ExplodingAgent:
        def decide(self, ctx):
            raise AssertionError("should never be called below min_checkpoints_before_acting")

    with _scoped_train_gpu_import() as train_gpu:
        new_densify, should_stop, record = train_gpu.apply_training_agent_at_checkpoint(
            ExplodingAgent(), held_out_history=[{"iter": 500, "ssim": 0.4, "n_gaussians": 100}],
            best_held_out={"ssim": {"iter": 500, "value": 0.4}}, current_iter=500, max_iterations=20000,
            checkpoint_every=500, effective_densify_until_iter=7000, n_gaussians=100, max_gaussians=150000,
            min_checkpoints_before_acting=2,
        )
    assert record is None
    assert should_stop is False
    assert new_densify == 7000  # unchanged


def test_apply_training_agent_continue_leaves_state_unchanged():
    class AlwaysContinue:
        def decide(self, ctx):
            return TrainingControlDecision(action="continue", reasoning="fine", agent_confidence=0.7)

    history = _history(peak_iter=1000, peak_value=0.5, current_iter=1500, current_value=0.55)
    with _scoped_train_gpu_import() as train_gpu:
        new_densify, should_stop, record = train_gpu.apply_training_agent_at_checkpoint(
            AlwaysContinue(), history, {"ssim": {"iter": 1500, "value": 0.55}}, current_iter=1500,
            max_iterations=20000, checkpoint_every=500, effective_densify_until_iter=7000,
            n_gaussians=3000, max_gaussians=150000,
        )
    assert record["action"] == "continue"
    assert should_stop is False
    assert new_densify == 7000


def test_apply_training_agent_stop_densifying_lowers_effective_value_to_current_iter():
    class AlwaysStopDensifying:
        def decide(self, ctx):
            return TrainingControlDecision(action="stop_densifying", reasoning="stalled", agent_confidence=0.5)

    history = _history(peak_iter=6500, peak_value=0.62, current_iter=8000, current_value=0.60)
    with _scoped_train_gpu_import() as train_gpu:
        new_densify, should_stop, record = train_gpu.apply_training_agent_at_checkpoint(
            AlwaysStopDensifying(), history, {"ssim": {"iter": 6500, "value": 0.62}}, current_iter=8000,
            max_iterations=20000, checkpoint_every=500, effective_densify_until_iter=15000,
            n_gaussians=320000, max_gaussians=150000,
        )
    assert new_densify == 8000  # lowered to current_iter, not left at 15000
    assert should_stop is False
    assert record["action"] == "stop_densifying"


def test_apply_training_agent_stop_densifying_never_raises_the_value():
    # If effective_densify_until_iter is ALREADY lower than current_iter (a prior
    # stop_densifying already happened), a later stop_densifying call must not raise it.
    class AlwaysStopDensifying:
        def decide(self, ctx):
            return TrainingControlDecision(action="stop_densifying", reasoning="still stalled")

    history = _history(peak_iter=6500, peak_value=0.62, current_iter=9000, current_value=0.55)
    with _scoped_train_gpu_import() as train_gpu:
        new_densify, _, _ = train_gpu.apply_training_agent_at_checkpoint(
            AlwaysStopDensifying(), history, {"ssim": {"iter": 6500, "value": 0.62}}, current_iter=9000,
            max_iterations=20000, checkpoint_every=500, effective_densify_until_iter=8000,  # already lower
            n_gaussians=400000, max_gaussians=150000,
        )
    assert new_densify == 8000  # min(8000, 9000) == 8000, not raised to 9000


def test_apply_training_agent_early_stop_sets_flag_and_leaves_densify_value_alone():
    class AlwaysEarlyStop:
        def decide(self, ctx):
            return TrainingControlDecision(action="early_stop", reasoning="declining", agent_confidence=0.6)

    history = _history(peak_iter=6500, peak_value=0.62, current_iter=9500, current_value=0.50)
    with _scoped_train_gpu_import() as train_gpu:
        new_densify, should_stop, record = train_gpu.apply_training_agent_at_checkpoint(
            AlwaysEarlyStop(), history, {"ssim": {"iter": 6500, "value": 0.62}}, current_iter=9500,
            max_iterations=20000, checkpoint_every=500, effective_densify_until_iter=8000,
            n_gaussians=400000, max_gaussians=150000,
        )
    assert should_stop is True
    assert new_densify == 8000  # early_stop doesn't touch densify_until_iter
    assert record["action"] == "early_stop"


def test_apply_training_agent_end_to_end_with_real_mock_training_control_agent():
    # Uses the REAL MockTrainingControlAgent (not a test stub) against the REAL
    # documented pattern, through the REAL apply_training_agent_at_checkpoint in
    # train_gpu.py -- the actual code path a live run takes with --use_agents.
    agent = MockTrainingControlAgent()
    history = []
    effective_densify_until_iter = 15000
    first_stop_iter = None
    with _scoped_train_gpu_import() as train_gpu:
        for it in range(500, 12001, 500):
            if it <= 6500:
                ssim = 0.15 + (it / 6500) * 0.47
            else:
                ssim = max(0.62 - (it - 6500) / 40000, 0.30)
            history.append({"iter": it, "ssim": ssim, "n_gaussians": it * 40})
            best_iter = max((h["iter"] for h in history if h["ssim"] >= max(x["ssim"] for x in history)), default=None)
            best_value = max(h["ssim"] for h in history)
            effective_densify_until_iter, should_stop, record = train_gpu.apply_training_agent_at_checkpoint(
                agent, history, {"ssim": {"iter": best_iter, "value": best_value}}, current_iter=it,
                max_iterations=20000, checkpoint_every=500,
                effective_densify_until_iter=effective_densify_until_iter,
                n_gaussians=it * 40, max_gaussians=150000,
            )
            if record and record["action"] == "stop_densifying" and first_stop_iter is None:
                first_stop_iter = it
            if should_stop:
                break

    assert first_stop_iter is not None
    assert first_stop_iter <= 8000  # matches the standalone mock_agents test's finding
    assert effective_densify_until_iter <= first_stop_iter


def test_scoped_import_restores_sys_modules_afterward():
    # The whole point of the context manager: must never leave stub pollution behind
    # for whichever test file pytest collects next in the same process.
    import sys as _sys
    before = {name: _sys.modules.get(name) for name in
              ("torch", "gaussian_model", "confidence_gaussian_model", "confidence_propagation", "losses")}
    with _scoped_train_gpu_import():
        pass
    after = {name: _sys.modules.get(name) for name in before}
    assert before == after
