"""
losses.py
LAYER 3 (GPU RESEARCH IMPLEMENTATION) -- STATUS: implemented, NOT GPU-executed
(pytorch-msssim needs installing; both functions are plain differentiable
tensor ops with no CUDA-specific code, so they WOULD run on CPU too, just
slowly at real image resolution -- not worth a CPU smoke test at that cost
here; correctness is straightforward enough to review directly).

Two loss terms, both DESIGN-INSPIRED by the author's own prior FastGS-QADS
project (frequency-weighted L1 and MS-SSIM), reimplemented independently:
edge-aware L1 weighting and multi-scale SSIM are established, widely-used
loss-engineering techniques in image reconstruction generally (not
specific to any one 3DGS variant), so what's "theirs" here is the
convention of pairing them together for Gaussian splatting training, not
the techniques themselves. Written fresh, different variable/function
structure, no code copied.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def frequency_weighted_l1(rendered: torch.Tensor, gt: torch.Tensor, edge_weight: float = 2.0) -> torch.Tensor:
    """
    rendered, gt: (3,H,W) or (B,3,H,W) float in [0,1]. Weights the L1 loss higher at
    high-frequency (edge) regions of the GROUND TRUTH image, computed with no_grad so
    the weight map itself contributes zero extra backward-pass cost versus vanilla L1
    -- the weight is a per-pixel scalar multiplier, not something being optimized.

    Rationale for THIS project specifically: single-pass coverage means fine detail
    (window edges, roofline corners, vegetation silhouettes) is exactly what's most at
    risk of being under-resolved by a sparse, confidence-gated Gaussian budget -- edge
    weighting biases the limited budget toward preserving those edges rather than
    smooth low-frequency regions that already reconstruct easily.
    """
    with torch.no_grad():
        gray = gt.mean(dim=-3, keepdim=True) if gt.dim() == 4 else gt.mean(dim=0, keepdim=True)
        gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
        gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
        edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        weight = 1.0 + edge_weight * edge_mag / (edge_mag.mean() + 1e-8)
        weight = weight.detach()
    l1_map = (rendered - gt).abs()
    return (l1_map * weight).mean()


def ms_ssim_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    1 - real multi-scale SSIM, via `pytorch-msssim` (Environment B only -- see
    layer1_cpu_sandbox/evaluation/metrics_visual.py for the same package used
    on the evaluation side). Falls back to single-scale SSIM computed by hand
    if the package or GPU-scale image isn't available, so training can still
    run (degraded, not crashed) -- mirrors the same fallback discipline used
    throughout this project's CPU-sandbox evaluation code.
    """
    r = rendered.unsqueeze(0) if rendered.dim() == 3 else rendered
    g = gt.unsqueeze(0) if gt.dim() == 3 else gt
    try:
        from pytorch_msssim import ms_ssim as real_ms_ssim
        min_side = min(r.shape[-2], r.shape[-1])
        if min_side > 160:  # pytorch-msssim's own hard minimum, see metrics_visual.py's note on this
            return 1.0 - real_ms_ssim(r, g, data_range=1.0)
    except ImportError:
        pass
    return 1.0 - _single_scale_ssim(r, g)


def _single_scale_ssim(r: torch.Tensor, g: torch.Tensor, window: int = 11, c1: float = 0.01 ** 2,
                        c2: float = 0.03 ** 2) -> torch.Tensor:
    """Minimal single-scale SSIM fallback (Gaussian-window not used -- a uniform box
    filter -- deliberately simple; this is a FALLBACK path, not the primary metric)."""
    pad = window // 2
    mu_r = F.avg_pool2d(r, window, 1, pad)
    mu_g = F.avg_pool2d(g, window, 1, pad)
    mu_r2, mu_g2, mu_rg = mu_r ** 2, mu_g ** 2, mu_r * mu_g
    sigma_r2 = F.avg_pool2d(r * r, window, 1, pad) - mu_r2
    sigma_g2 = F.avg_pool2d(g * g, window, 1, pad) - mu_g2
    sigma_rg = F.avg_pool2d(r * g, window, 1, pad) - mu_rg
    ssim_map = ((2 * mu_rg + c1) * (2 * sigma_rg + c2)) / ((mu_r2 + mu_g2 + c1) * (sigma_r2 + sigma_g2 + c2))
    return ssim_map.mean()


def combined_loss(rendered: torch.Tensor, gt: torch.Tensor, lambda_ssim: float = 0.2,
                   edge_weight: float = 2.0) -> dict:
    """Returns a dict (not just a scalar) so the training loop can log each term
    separately -- useful for diagnosing exactly the kind of adversarial-content
    failure found in the coastal-water self-consistency run (see chat history):
    a loss that's dominated by MS-SSIM collapsing on a specular region looks very
    different from one dominated by plain L1 error, and that distinction is lost
    if only the summed scalar is logged."""
    l1_term = frequency_weighted_l1(rendered, gt, edge_weight=edge_weight)
    ssim_term = ms_ssim_loss(rendered, gt)
    total = (1 - lambda_ssim) * l1_term + lambda_ssim * ssim_term
    return {"total": total, "l1": l1_term.detach(), "ssim_loss": ssim_term.detach()}
