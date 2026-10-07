"""
metrics_visual.py
LAYER 1 (CPU SANDBOX) -- STATUS: implemented, CPU-tested (PSNR/SSIM/MS-SSIM proxy).
LPIPS and real (non-proxy) MS-SSIM: implemented with automatic environment
detection -- NaN/proxy fallback in Environment A (no `lpips`/`pytorch-msssim`/
network access), REAL computed values in Environment B (Colab: GPU + network
access to download LPIPS's pretrained AlexNet weights). See `compute_lpips`
and `compute_ms_ssim_real` docstrings below.

Section A of the evaluation framework: IMAGE-SPACE / VIEW-SYNTHESIS
fidelity only. These numbers say nothing about metric 3-D accuracy -- see
metrics_geometry.py and metrics_metric_accuracy.py for that, and never let
a PSNR/SSIM/LPIPS number stand in for a geometric-accuracy claim anywhere in
the report or the demo.
"""
from __future__ import annotations

import dataclasses
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr


@dataclasses.dataclass
class VisualFidelityResult:
    psnr: float
    ssim: float
    ms_ssim_proxy: float     # REAL MS-SSIM if `pytorch-msssim` is installed (Environment B);
                             # otherwise the simplified pyramid proxy (Environment A) -- see
                             # compute_ms_ssim_real() / compute_ms_ssim_proxy() below. Field name
                             # kept for API stability; check environment to know which you got.
    valid_pixel_fraction: float  # fraction of GT pixels the reconstruction actually rendered anything for
    lpips: float = float("nan")  # NaN = not computed (Environment A); real value in Environment B


def compute_psnr_ssim(rendered: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None):
    """rendered, gt: (H,W,3) uint8. If valid_mask is given, PSNR/SSIM are computed only
    where the reconstruction actually produced a pixel (missing regions must not be
    silently scored as if they were black-vs-black matches)."""
    if valid_mask is None:
        valid_mask = np.ones(rendered.shape[:2], dtype=bool)
    if valid_mask.sum() < 16:
        return float("nan"), float("nan")
    r = rendered.astype(np.float64)
    g = gt.astype(np.float64)
    mse = np.mean((r[valid_mask] - g[valid_mask]) ** 2)
    psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 1e-12 else 99.0

    ys, xs = np.nonzero(valid_mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop_r = rendered[y0:y1, x0:x1]
    crop_g = gt[y0:y1, x0:x1]
    try:
        ssim_val = sk_ssim(crop_g, crop_r, channel_axis=2, data_range=255)
    except TypeError:
        ssim_val = sk_ssim(crop_g, crop_r, multichannel=True, data_range=255)
    return float(psnr), float(ssim_val)


def compute_ms_ssim_proxy(rendered: np.ndarray, gt: np.ndarray, scales=(1.0, 0.5, 0.25)) -> float:
    """
    A SIMPLIFIED multi-scale SSIM proxy: single-scale SSIM averaged over a small
    Gaussian-pyramid of downsampled copies, weighted toward coarser scales (which is
    the qualitative behaviour MS-SSIM is designed for -- more robust to small
    misalignment than single-scale SSIM). This is NOT the standard MS-SSIM
    algorithm (which uses a fixed 5-scale contrast/structure decomposition) --
    production numbers should use `pytorch-msssim` on GPU. Labelled a "proxy" in
    every report table for exactly this reason.
    """
    import cv2
    vals, weights = [], []
    for i, s in enumerate(scales):
        if s != 1.0:
            r = cv2.resize(rendered, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            g = cv2.resize(gt, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            r, g = rendered, gt
        if min(r.shape[:2]) < 8:
            continue
        try:
            v = sk_ssim(g, r, channel_axis=2, data_range=255)
        except TypeError:
            v = sk_ssim(g, r, multichannel=True, data_range=255)
        w = 1.0 + i  # weight coarser scales more, matching MS-SSIM's structure emphasis
        vals.append(v); weights.append(w)
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=weights))


def compute_lpips(rendered: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None):
    """
    LPIPS requires a pretrained perceptual network (AlexNet/VGG feature
    extractor with learned linear weights). Environment A (this CPU sandbox)
    has no GPU and no network access to the weight-hosting domain, so this
    used to unconditionally return NaN.

    Environment B (Colab) DOES have network access, so this now actually
    computes real LPIPS there: if `torch` + the `lpips` package are
    importable, it lazily builds one global AlexNet-backed LPIPS network
    (cached across calls -- rebuilding a network per call would be wasteful)
    and returns a real float. Falls back to the same honest
    NaN-with-explanation in any environment where that's not possible,
    rather than silently approximating perceptual similarity with something
    else and mislabeling it "LPIPS".
    """
    global _LPIPS_NET
    try:
        import torch
        import lpips as lpips_pkg
    except ImportError:
        return float("nan"), ("LPIPS not computed: requires `torch` + `lpips` packages, "
                               "not installed in this environment (expected in Environment A; "
                               "run `pip install lpips torch` in Environment B)")

    if valid_mask is None:
        valid_mask = np.ones(rendered.shape[:2], dtype=bool)
    if valid_mask.sum() < 16:
        return float("nan"), "LPIPS not computed: fewer than 16 valid (covered) pixels"

    if _LPIPS_NET is None:
        _LPIPS_NET = lpips_pkg.LPIPS(net="alex")  # downloads pretrained weights on first use
        _LPIPS_NET.eval()

    def to_tensor(img):
        # LPIPS expects NCHW, float in [-1, 1]. Zero out pixels the reconstruction
        # never rendered so they don't inject spurious perceptual "difference".
        masked = np.where(valid_mask[..., None], img, gt)  # neutralize uncovered pixels
        t = torch.from_numpy(masked).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t

    with torch.no_grad():
        d = _LPIPS_NET(to_tensor(rendered), to_tensor(gt))
    return float(d.item()), None


_LPIPS_NET = None  # lazy-initialized module-level cache, see compute_lpips()


def compute_ms_ssim_real(rendered: np.ndarray, gt: np.ndarray, win_size: int = 11):
    """
    Real (non-proxy) multi-scale SSIM via `pytorch-msssim`, used automatically
    in Environment B where that package is installed; returns None (caller
    falls back to `compute_ms_ssim_proxy`) if unavailable OR if the image is
    too small for the algorithm.

    `pytorch-msssim` does 4 successive 2x downsamplings internally and
    REQUIRES the smaller image dimension to exceed (win_size-1)*2**4 = 160px
    (for the default win_size=11) or it raises AssertionError. This CPU
    sandbox's default synthetic frames are 160x120 -- smaller side 120 <
    160 -- so real MS-SSIM structurally cannot run at that resolution,
    Colab GPU or not. This is a genuine constraint of the algorithm, not a
    bug in how it's called here: build a dataset with
    `build_dataset(..., width=W, height=H)` where min(W,H) > 160 (e.g.
    200x200) if you specifically want a real MS-SSIM number instead of the
    proxy -- see docs/GETTING_STARTED.md.
    """
    min_required = (win_size - 1) * (2 ** 4)
    if min(rendered.shape[0], rendered.shape[1]) <= min_required:
        return None
    try:
        import torch
        from pytorch_msssim import ms_ssim as real_ms_ssim
    except ImportError:
        return None
    t_r = torch.from_numpy(rendered).float().permute(2, 0, 1).unsqueeze(0)
    t_g = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0)
    try:
        with torch.no_grad():
            return float(real_ms_ssim(t_r, t_g, data_range=255, win_size=win_size))
    except AssertionError:
        return None  # belt-and-suspenders: some other internal size/shape constraint we didn't anticipate


def evaluate_visual_fidelity(rendered: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None) -> VisualFidelityResult:
    if valid_mask is None:
        valid_mask = np.ones(rendered.shape[:2], dtype=bool)
    psnr, ssim_v = compute_psnr_ssim(rendered, gt, valid_mask)

    ms_real = compute_ms_ssim_real(rendered, gt)
    ms = ms_real if ms_real is not None else compute_ms_ssim_proxy(rendered, gt)

    lpips_val, _reason = compute_lpips(rendered, gt, valid_mask)

    return VisualFidelityResult(psnr=psnr, ssim=ssim_v, ms_ssim_proxy=ms,
                                 valid_pixel_fraction=float(valid_mask.mean()), lpips=lpips_val)
