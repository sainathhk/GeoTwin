// confidence_gaussian.cpp
// LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, NOT CUDA-compiled,
// NOT GPU-executed in Environment A. Torch-extension (pybind11) bindings exposing
// the three CUDA kernel launchers declared in confidence_gaussian.h to Python as
// `confidence_gaussian_cuda.accumulate_confidence_stats(...)`, `.compute_confidence(...)`,
// `.gate_densify_prune(...)`. Built by layer3_gpu/setup.py via
// torch.utils.cpp_extension.CUDAExtension; see docs/SIH_IMPLEMENTATION_PLAN.md for
// the exact Colab build command.
//
// Basic shape/dtype/device checks are done here (host side) rather than inside the
// kernels, matching standard torch-extension practice (fail fast with a clear Python
// exception rather than an obscure device-side fault).

#include "confidence_gaussian.h"
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void accumulate_confidence_stats(
    torch::Tensor touched_idx, torch::Tensor view_angle, torch::Tensor consistency,
    double quality, double pose_confidence, torch::Tensor obs_count, torch::Tensor angle_min,
    torch::Tensor angle_max, torch::Tensor consistency_sum, torch::Tensor quality_sum,
    torch::Tensor pose_conf_sum, torch::Tensor n_accum) {
    CHECK_INPUT(touched_idx); CHECK_INPUT(view_angle); CHECK_INPUT(consistency);
    CHECK_INPUT(obs_count); CHECK_INPUT(angle_min); CHECK_INPUT(angle_max);
    CHECK_INPUT(consistency_sum); CHECK_INPUT(quality_sum); CHECK_INPUT(pose_conf_sum); CHECK_INPUT(n_accum);
    TORCH_CHECK(touched_idx.dtype() == torch::kInt64, "touched_idx must be int64");
    TORCH_CHECK(touched_idx.size(0) == view_angle.size(0) && touched_idx.size(0) == consistency.size(0),
                "touched_idx / view_angle / consistency must have matching length K");
    accumulate_confidence_stats_cuda(touched_idx, view_angle, consistency, (float)quality, (float)pose_confidence,
                                      obs_count, angle_min, angle_max, consistency_sum, quality_sum,
                                      pose_conf_sum, n_accum);
}

torch::Tensor compute_confidence(
    torch::Tensor obs_count, torch::Tensor angle_min, torch::Tensor angle_max,
    torch::Tensor consistency_sum, torch::Tensor quality_sum, torch::Tensor pose_conf_sum,
    torch::Tensor n_accum, double w_obs, double w_angle, double w_consistency, double w_quality,
    double w_pose, double obs_tau, double target_angle_spread_rad, double single_view_cap) {
    CHECK_INPUT(obs_count); CHECK_INPUT(angle_min); CHECK_INPUT(angle_max);
    CHECK_INPUT(consistency_sum); CHECK_INPUT(quality_sum); CHECK_INPUT(pose_conf_sum); CHECK_INPUT(n_accum);
    return compute_confidence_cuda(obs_count, angle_min, angle_max, consistency_sum, quality_sum, pose_conf_sum,
                                    n_accum, (float)w_obs, (float)w_angle, (float)w_consistency, (float)w_quality,
                                    (float)w_pose, (float)obs_tau, (float)target_angle_spread_rad,
                                    (float)single_view_cap);
}

std::vector<torch::Tensor> gate_densify_prune(
    torch::Tensor confidence, torch::Tensor opacity, torch::Tensor grad_accum,
    double opacity_thresh, double confidence_prune_below, double grad_thresh, double confidence_densify_above,
    c10::optional<torch::Tensor> optimizer_struggle_signal, double struggle_thresh) {
    CHECK_INPUT(confidence); CHECK_INPUT(opacity); CHECK_INPUT(grad_accum);
    bool has_struggle = optimizer_struggle_signal.has_value();
    torch::Tensor struggle_tensor = has_struggle ? optimizer_struggle_signal.value()
                                                  : torch::zeros_like(confidence);
    if (has_struggle) { CHECK_INPUT(struggle_tensor); }
    return gate_densify_prune_cuda(confidence, opacity, grad_accum, (float)opacity_thresh,
                                    (float)confidence_prune_below, (float)grad_thresh,
                                    (float)confidence_densify_above, struggle_tensor, has_struggle,
                                    (float)struggle_thresh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("accumulate_confidence_stats", &accumulate_confidence_stats,
          "Scatter-accumulate per-view confidence evidence into per-Gaussian buffers (CUDA)");
    m.def("compute_confidence", &compute_confidence,
          "Weighted-geometric-mean confidence from accumulated evidence buffers (CUDA)");
    m.def("gate_densify_prune", &gate_densify_prune,
          "Confidence + gradient (+ optional OSAD-inspired optimizer-struggle signal) joint "
          "densify/prune gating (CUDA)",
          py::arg("confidence"), py::arg("opacity"), py::arg("grad_accum"),
          py::arg("opacity_thresh"), py::arg("confidence_prune_below"),
          py::arg("grad_thresh"), py::arg("confidence_densify_above"),
          py::arg("optimizer_struggle_signal") = py::none(), py::arg("struggle_thresh") = 3.0);
}
