# Components That Cannot Realistically Be Validated in the CPU Sandbox

1. **Whether confidence-gated densification actually beats gradient-only
   densification on real Gaussian optimization.** This is THE central
   empirical claim and it fundamentally requires the differentiable
   rasterizer and an iterative optimize/densify/prune loop running at real
   scale (thousands-to-millions of Gaussians, hundreds of training
   iterations) — not reproducible in a Python loop on CPU. Blocked on
   Environment B.

2. **Real image-domain robustness under actual camera sensor noise,
   rolling-shutter effects, and lens distortion.** The synthetic renderer
   uses an idealized pinhole model with hand-designed degradation
   (motion blur kernel, JPEG re-encode, gamma/brightness). Real drone
   footage has correlated noise structure (rolling shutter, lens
   vignetting/distortion, sensor-specific compression artifacts,
   auto-exposure hunting) that a synthetic generator can approximate but
   not replace. Needs real or high-fidelity simulated (e.g. AirSim/Unreal)
   footage.

3. **LPIPS and true multi-scale SSIM.** ~~Both need either network access
   (pretrained weight download) or a GPU-side `pytorch-msssim` install —
   neither available in Environment A.~~ **UPDATE:** `metrics_visual.py` now
   auto-detects `lpips`/`pytorch-msssim` availability and computes REAL
   values when present (Environment B), falling back to the honest
   NaN/proxy behavior when absent (Environment A) — no code change needed
   between environments, just install the packages. Still genuinely
   unavailable in Environment A by design (no network access to weights).

4. **Classical photogrammetry / COLMAP, real FastGS, and a NeRF baseline.**
   All three need GPU and/or third-party toolchains not installable in a
   size- and time-constrained CPU-only sandbox. The ablation performed here
   isolates what Layer 1's OWN modules contribute to each other; it is not
   a substitute for the baseline comparison table in
   `docs/EXPERIMENT_PLAN.md`.

5. **Real-world georeferencing accuracy against actual RTK/GCP survey
   data.** The synthetic "control points" (building roof centers) are a
   reasonable methodological stand-in but are not real survey-grade ground
   truth; the metric-accuracy numbers in `outputs/run1/` validate the
   EVALUATION METHOD, not real-world accuracy. **Update:** real video +
   GPS log ingestion now exists (`layer1_cpu_sandbox/real_data/`), but it
   has no ground truth either — real georeferencing validation still needs
   an actual RTK/GCP survey to compare against, which nothing in this repo
   has access to.

6. **GPU throughput, memory, and real-time/near-real-time processing
   claims.** `metrics_efficiency.py` explicitly labels every timing number
   as "Environment A, single-core CPU — NOT representative of Environment B
   throughput." No FPS or latency number should be quoted for the final
   system until `colab_smoke_test.py`'s throughput check (and a full
   pipeline timing run) has actually executed on a Colab GPU.

7. **Dynamic-object segmentation precision at realistic image resolution.**
   The measured classical-baseline weakness (see
   `perception/dynamic_object_filter.py`) was found at 160×120px synthetic
   resolution; whether it remains a problem at 1080p/4K with more pixels
   per object is a genuinely open, untested question — plausible that it
   resolves itself at higher resolution, but that must be measured, not
   assumed.
