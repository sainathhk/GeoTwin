#!/usr/bin/env bash
# Colab bootstrap for Gaussian training followed by automatic VGGT geometry.
# Run from the extracted project root in a fresh Colab GPU runtime.
set -euo pipefail

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
python -m pip install -q -r requirements-gpu.txt
python -m pip install -q -r requirements-vggt.txt

python - <<'PY'
import numpy, torch
print(f"NumPy: {numpy.__version__} (project/VGGT pin: 1.26.1)")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if numpy.__version__ != "1.26.1":
    raise SystemExit("Wrong NumPy version. Restart the Colab runtime, rerun this bootstrap, and do not install NumPy 2.x.")
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU detected. In Colab select Runtime > Change runtime type > T4 GPU.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# Compile the project's confidence extension and install the external,
# unmodified Graphdeco differentiable rasterizer used by the 3DGS loop.
python -m pip install -q --no-build-isolation \
  "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
(cd layer3_gpu && python setup.py build_ext --inplace)
PYTHONPATH="${PWD}/layer3_gpu:${PYTHONPATH:-}" python layer3_gpu/python/colab_smoke_test.py

cat <<'EOF'
Bootstrap complete.
Run your normal train_gpu.py command. For --source real, the process will
finish Gaussian training first, then automatically run VGGT fusion and
Poisson depth-12 Trim001/Clean500. Outputs are saved under the same --out_dir
in the vggt_trim001_clean500/ subfolder.
EOF
