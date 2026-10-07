"""
setup.py -- builds the confidence_gaussian CUDA extension.
STATUS: implemented, NOT run in Environment A (no nvcc / no CUDA device here).

Run this ONLY in Environment B (Colab GPU runtime):
    cd layer3_gpu && python3 setup.py build_ext --inplace

Then in Python:
    import confidence_gaussian_cuda
    confidence_gaussian_cuda.compute_confidence(...)

See docs/SIH_IMPLEMENTATION_PLAN.md for the full Colab bootstrap sequence
(clone repo -> pip install -r requirements-gpu.txt -> this build step ->
smoke test via layer3_gpu/python/colab_smoke_test.py).
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="confidence_gaussian_cuda",
    version="0.1.0",
    description="Confidence-aware Gaussian densify/prune CUDA kernels for single-pass "
                "drone 3D reconstruction (SIH 2026 PS 26158).",
    ext_modules=[
        CUDAExtension(
            name="confidence_gaussian_cuda",
            sources=[
                "csrc/confidence_gaussian/confidence_gaussian.cpp",
                "csrc/confidence_gaussian/confidence_gaussian_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
