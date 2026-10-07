// confidence_gaussian.h
// LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, NOT CUDA-compiled,
// NOT GPU-executed in Environment A (no nvcc / no CUDA device here). Must be built
// and validated in Environment B (Colab GPU) -- see layer3_gpu/setup.py and
// docs/SIH_IMPLEMENTATION_PLAN.md "first thing to run in Colab".
//
// Declares the three CUDA kernel launchers behind confidence_gaussian.cpp's
// torch extension bindings:
//   1. accumulate_confidence_stats -- per-view scatter-accumulate of the 5
//      evidence signals into per-Gaussian running buffers (the CUDA-scale
//      equivalent of GaussianConfidenceStats.update() in
//      layer3_gpu/python/confidence_gaussian_model.py, but O(touched
//      Gaussians) per view via atomics instead of a Python-level scatter,
//      which matters once N is in the hundreds of thousands to millions).
//   2. compute_confidence -- the weighted-geometric-mean formula, identical
//      to layer1_cpu_sandbox/reconstruction/confidence.py and
//      GaussianConfidenceStats.confidence(), evaluated per-Gaussian on
//      device so it can run every densification interval without a
//      device<->host round trip.
//   3. gate_densify_prune -- the core novel control logic: confidence AND
//      view-space-gradient jointly gate densify/prune decisions (see
//      layer2_interfaces/i_gaussian_refine.py for why this differs from
//      standard 3DGS/FastGS gradient-only densification). Extended to
//      optionally accept an OSAD-inspired optimizer-struggle signal as an
//      ADDITIONAL densify trigger, subordinate to the confidence gate --
//      see layer3_gpu/python/confidence_propagation.py's module docstring
//      for the full design rationale.
//
// DELIBERATELY NOT a CUDA kernel here: the STCP-inspired temporal+spatial
// confidence propagation (ObservationConfidencePropagator in
// confidence_propagation.py). torch.unique/scatter_add_ already run
// efficiently on CUDA tensors as plain PyTorch ops -- writing a hand-rolled
// CUDA voxel-hash kernel for this would be premature optimization without a
// demonstrated throughput need (see docs/STOP_CONDITIONS.md). Revisit only
// if profiling in Environment B shows it as an actual bottleneck.
#pragma once
#include <torch/extension.h>
#include <cstdint>

void accumulate_confidence_stats_cuda(
    torch::Tensor touched_idx,       // (K,) int64 -- Gaussian indices touched by this view
    torch::Tensor view_angle,        // (K,) float32
    torch::Tensor consistency,       // (K,) float32
    float quality,                   // scalar for this view
    float pose_confidence,           // scalar for this view
    torch::Tensor obs_count,         // (N,) float32, in/out
    torch::Tensor angle_min,         // (N,) float32, in/out
    torch::Tensor angle_max,         // (N,) float32, in/out
    torch::Tensor consistency_sum,   // (N,) float32, in/out
    torch::Tensor quality_sum,       // (N,) float32, in/out
    torch::Tensor pose_conf_sum,     // (N,) float32, in/out
    torch::Tensor n_accum            // (N,) float32, in/out
);

torch::Tensor compute_confidence_cuda(
    torch::Tensor obs_count, torch::Tensor angle_min, torch::Tensor angle_max,
    torch::Tensor consistency_sum, torch::Tensor quality_sum, torch::Tensor pose_conf_sum,
    torch::Tensor n_accum,
    float w_obs, float w_angle, float w_consistency, float w_quality, float w_pose,
    float obs_tau, float target_angle_spread_rad, float single_view_cap
);

std::vector<torch::Tensor> gate_densify_prune_cuda(
    torch::Tensor confidence, torch::Tensor opacity, torch::Tensor grad_accum,
    float opacity_thresh, float confidence_prune_below,
    float grad_thresh, float confidence_densify_above,
    torch::Tensor optimizer_struggle_signal, bool has_struggle_signal, float struggle_thresh
);
