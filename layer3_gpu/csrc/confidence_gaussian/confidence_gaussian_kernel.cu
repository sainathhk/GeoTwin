// confidence_gaussian_kernel.cu
// LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, NOT CUDA-compiled,
// NOT GPU-executed in Environment A. See confidence_gaussian.h for the module-level
// explanation of what these three kernels do and why they exist as CUDA (rather than
// staying in Python/PyTorch, as in confidence_gaussian_model.py's CPU-testable path).
//
// Numerically, `compute_confidence_kernel` below implements EXACTLY the same formula
// as layer1_cpu_sandbox/reconstruction/confidence.py's `compute_confidence()` and
// layer3_gpu/python/confidence_gaussian_model.py's `GaussianConfidenceStats.confidence()`
// -- a weighted geometric mean of 5 clamped [0,1] evidence scores, with the
// single-observation confidence cap. Keeping three independent implementations of the
// same formula in sync is a real maintenance risk; docs/STOP_CONDITIONS.md flags
// consolidating them (e.g. code-generating this kernel's math from one source of
// truth) as a fast-follow, not a Day-1 requirement.

#include "confidence_gaussian.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

#define CUDA_1D_LAUNCH(N, THREADS) <<<(( (N) + (THREADS) - 1) / (THREADS)), (THREADS), 0, at::cuda::getCurrentCUDAStream()>>>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

// CUDA has no native atomicMin/atomicMax for float; the standard bit-trick CAS loop.
__device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    while (value < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        if (assumed == old) break;
    }
}

__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    while (value > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        if (assumed == old) break;
    }
}

__global__ void accumulate_confidence_stats_kernel(
    const int64_t* __restrict__ touched_idx, const float* __restrict__ view_angle,
    const float* __restrict__ consistency, const float quality, const float pose_confidence,
    const int K, float* __restrict__ obs_count, float* __restrict__ angle_min,
    float* __restrict__ angle_max, float* __restrict__ consistency_sum,
    float* __restrict__ quality_sum, float* __restrict__ pose_conf_sum, float* __restrict__ n_accum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    int64_t g = touched_idx[i];
    atomicAdd(&obs_count[g], 1.0f);
    atomicMinFloat(&angle_min[g], view_angle[i]);
    atomicMaxFloat(&angle_max[g], view_angle[i]);
    atomicAdd(&consistency_sum[g], consistency[i]);
    atomicAdd(&quality_sum[g], quality);
    atomicAdd(&pose_conf_sum[g], pose_confidence);
    atomicAdd(&n_accum[g], 1.0f);
}

__global__ void compute_confidence_kernel(
    const float* __restrict__ obs_count, const float* __restrict__ angle_min,
    const float* __restrict__ angle_max, const float* __restrict__ consistency_sum,
    const float* __restrict__ quality_sum, const float* __restrict__ pose_conf_sum,
    const float* __restrict__ n_accum, const int N,
    const float w_obs, const float w_angle, const float w_consistency, const float w_quality,
    const float w_pose, const float obs_tau, const float target_angle_spread_rad,
    const float single_view_cap, float* __restrict__ confidence_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float eps = 1e-6f;

    float obs = obs_count[i];
    float obs_score = 1.0f - expf(-fmaxf(obs - 1.0f, 0.0f) / obs_tau);

    float spread = angle_max[i] - angle_min[i];
    if (!isfinite(spread) || spread < 0.0f) spread = 0.0f;
    float angle_score = fminf(fmaxf(spread / target_angle_spread_rad, 0.0f), 1.0f);

    float n_safe = fmaxf(n_accum[i], 1.0f);
    float consistency_score = fminf(fmaxf(consistency_sum[i] / n_safe, 0.0f), 1.0f);
    float quality_score = fminf(fmaxf(quality_sum[i] / n_safe, 0.0f), 1.0f);
    float pose_score = fminf(fmaxf(pose_conf_sum[i] / n_safe, 0.0f), 1.0f);

    float log_conf = w_obs * logf(obs_score + eps) + w_angle * logf(angle_score + eps) +
                      w_consistency * logf(consistency_score + eps) +
                      w_quality * logf(quality_score + eps) + w_pose * logf(pose_score + eps);
    float confidence = expf(log_conf);

    if (obs <= 1.0f && confidence > single_view_cap) {
        confidence = single_view_cap;
    }
    confidence_out[i] = confidence;
}

__global__ void gate_densify_prune_kernel(
    const float* __restrict__ confidence, const float* __restrict__ opacity,
    const float* __restrict__ grad_accum, const int N, const float opacity_thresh,
    const float confidence_prune_below, const float grad_thresh, const float confidence_densify_above,
    const float* __restrict__ struggle_signal, const bool has_struggle_signal, const float struggle_thresh,
    bool* __restrict__ prune_mask, bool* __restrict__ densify_mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    bool prune = (opacity[i] < opacity_thresh) || (confidence[i] < confidence_prune_below);
    bool confidence_gate = (confidence[i] >= confidence_densify_above) && !prune;

    bool trigger = grad_accum[i] > grad_thresh;
    if (has_struggle_signal) {
        // OSAD-inspired: ADDS a densify trigger, never bypasses confidence_gate below.
        // See confidence_propagation.py's module docstring for the full rationale.
        trigger = trigger || (struggle_signal[i] > struggle_thresh);
    }

    prune_mask[i] = prune;
    densify_mask[i] = trigger && confidence_gate;
}

}  // namespace

void accumulate_confidence_stats_cuda(
    torch::Tensor touched_idx, torch::Tensor view_angle, torch::Tensor consistency,
    float quality, float pose_confidence, torch::Tensor obs_count, torch::Tensor angle_min,
    torch::Tensor angle_max, torch::Tensor consistency_sum, torch::Tensor quality_sum,
    torch::Tensor pose_conf_sum, torch::Tensor n_accum) {
    const int K = touched_idx.size(0);
    if (K == 0) return;
    accumulate_confidence_stats_kernel CUDA_1D_LAUNCH(K, THREADS_PER_BLOCK) (
        touched_idx.data_ptr<int64_t>(), view_angle.data_ptr<float>(), consistency.data_ptr<float>(),
        quality, pose_confidence, K, obs_count.data_ptr<float>(), angle_min.data_ptr<float>(),
        angle_max.data_ptr<float>(), consistency_sum.data_ptr<float>(), quality_sum.data_ptr<float>(),
        pose_conf_sum.data_ptr<float>(), n_accum.data_ptr<float>());
}

torch::Tensor compute_confidence_cuda(
    torch::Tensor obs_count, torch::Tensor angle_min, torch::Tensor angle_max,
    torch::Tensor consistency_sum, torch::Tensor quality_sum, torch::Tensor pose_conf_sum,
    torch::Tensor n_accum, float w_obs, float w_angle, float w_consistency, float w_quality,
    float w_pose, float obs_tau, float target_angle_spread_rad, float single_view_cap) {
    const int N = obs_count.size(0);
    auto confidence = torch::empty({N}, obs_count.options());
    if (N == 0) return confidence;
    compute_confidence_kernel CUDA_1D_LAUNCH(N, THREADS_PER_BLOCK) (
        obs_count.data_ptr<float>(), angle_min.data_ptr<float>(), angle_max.data_ptr<float>(),
        consistency_sum.data_ptr<float>(), quality_sum.data_ptr<float>(), pose_conf_sum.data_ptr<float>(),
        n_accum.data_ptr<float>(), N, w_obs, w_angle, w_consistency, w_quality, w_pose, obs_tau,
        target_angle_spread_rad, single_view_cap, confidence.data_ptr<float>());
    return confidence;
}

std::vector<torch::Tensor> gate_densify_prune_cuda(
    torch::Tensor confidence, torch::Tensor opacity, torch::Tensor grad_accum,
    float opacity_thresh, float confidence_prune_below, float grad_thresh, float confidence_densify_above,
    torch::Tensor optimizer_struggle_signal, bool has_struggle_signal, float struggle_thresh) {
    const int N = confidence.size(0);
    auto options = torch::TensorOptions().dtype(torch::kBool).device(confidence.device());
    auto prune_mask = torch::empty({N}, options);
    auto densify_mask = torch::empty({N}, options);
    if (N == 0) return {prune_mask, densify_mask};
    const float* struggle_ptr = has_struggle_signal ? optimizer_struggle_signal.data_ptr<float>() : nullptr;
    gate_densify_prune_kernel CUDA_1D_LAUNCH(N, THREADS_PER_BLOCK) (
        confidence.data_ptr<float>(), opacity.data_ptr<float>(), grad_accum.data_ptr<float>(), N,
        opacity_thresh, confidence_prune_below, grad_thresh, confidence_densify_above,
        struggle_ptr, has_struggle_signal, struggle_thresh,
        prune_mask.data_ptr<bool>(), densify_mask.data_ptr<bool>());
    return {prune_mask, densify_mask};
}
