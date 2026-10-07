# SIH Implementation Plan — What To Build Once Environment B Is Available

## Day 1 (Colab GPU session opens)
1. `bash colab_bootstrap.sh` — GPU check, deps, build `confidence_gaussian_cuda`,
   run `colab_smoke_test.py`. **Do not proceed past a failing smoke test** —
   it exists specifically to catch a CUDA kernel that "compiles but is
   numerically wrong" before any time is spent training on top of it.
2. `pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git`
   (already in `colab_bootstrap.sh`, listed again here because it's the one
   external dependency the whole training loop hinges on).
3. Detect the actual Colab GPU architecture (`nvidia-smi --query-gpu=name
   --format=csv`) and set `CMAKE_CUDA_ARCHITECTURES` / the `setup.py` build
   accordingly if it's not one of T4/A100/L4/V100.

## Day 1-2: Gaussian initializer + training loop
4. Implement `i_gaussian_init.py`'s concrete class: seed one Gaussian per
   confidence-passing fused point from a GPU-side port of
   `prototype_point_repr.py`'s fusion logic (the fusion LOGIC is already
   validated on CPU — this is a port to `torch`/CUDA tensors, not a redesign).
5. Implement the training loop: standard 3DGS photometric loss (L1 + D-SSIM)
   via `diff-gaussian-rasterization`, PLUS per-view confidence-stat
   accumulation via `confidence_gaussian_cuda.accumulate_confidence_stats`
   at every training iteration.
6. Every `densify_interval` iterations, call
   `confidence_gaussian_cuda.gate_densify_prune` instead of (or alongside,
   for the A/B ablation) standard 3DGS's gradient-only rule.

## Day 2-3: first real dataset run
7. Run on a real or high-fidelity simulated single-pass drone video (see
   `docs/EXPERIMENT_PLAN.md` dataset strategy) — start with the smallest
   real scene available to catch integration bugs cheaply.
8. Port `evaluate_result.py`'s metric suite to consume Gaussian-derived
   point positions (`gaussians.positions`) — the geometry/metric-
   accuracy/completeness code needs NO changes, only the renderer call for
   visual metrics changes (swap `point_renderer.render_point_cloud` for the
   real differentiable rasterizer's inference-mode render).

## Day 3-4: ablation + baselines
9. Run the GPU-side ablation: confidence-gated vs. gradient-only
   densification, holding everything else fixed. This is the single most
   important number in the whole project — see
   `docs/EXPERIMENT_PLAN.md` Experiment 1.
10. Run baseline comparisons: classical COLMAP+MVS, standard 3DGS, FastGS
    (all installable from their public repos), our method.

## Day 4-5: demo polish
11. Wire up the 5-minute demo flow (`docs/` — see the demo flow section of
    the original design conversation) using REAL output from steps above:
    progressive reconstruction, confidence heatmap, GT-vs-rendered
    comparison, metric measurement overlay. Close with the exported
    `mesh_iter<N>.obj` (`train_gpu.py`'s automatic end-of-training export,
    see `docs/GETTING_STARTED.md` Part F4) as the tangible takeaway a judge
    can open in Blender/MeshLab themselves, not just watch on your screen.
12. Freeze numbers, regenerate all tables/figures one final time from a
    clean run (never hand-edit a result table).

## Explicit non-goals for the Colab sprint
- Do not attempt surface/occlusion completion (see `STOP_CONDITIONS.md`).
- Do not attempt real-time processing optimization before correctness and
  the core ablation are both validated — premature optimization here risks
  the one experiment that actually matters.
- Do not swap in a learned depth network or learned dynamic segmenter
  before the confidence-gated densification ablation is complete; those are
  independent upgrades that would confound attribution if changed together.
