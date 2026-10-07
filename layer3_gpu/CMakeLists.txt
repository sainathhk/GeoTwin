# CMakeLists.txt -- standalone C++/libtorch build of the confidence_gaussian CUDA
# kernels, as a library (`confidence_gaussian_core`) usable from non-Python C++
# deployment code (e.g. an on-device/edge inference binary), independent of the
# Python torch-extension build in setup.py.
#
# STATUS: implemented, NOT configured/built anywhere (no CMake+CUDA toolchain
# exercised in Environment A). Build in Environment B:
#
#   cd layer3_gpu
#   mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
#   cmake --build . -j
#
# Requires: CUDA toolkit (nvcc) on PATH, libtorch (via the installed `torch` pip
# package's cmake_prefix_path, as above -- no separate libtorch download needed).

cmake_minimum_required(VERSION 3.18)
project(confidence_gaussian_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Reasonable default architectures for Colab GPUs (T4=75, A100=80, L4=89, V100=70).
# Override with -DCMAKE_CUDA_ARCHITECTURES=<arch> if Colab hands out something else;
# `docs/SIH_IMPLEMENTATION_PLAN.md` documents how to detect it at runtime.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
endif()

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(confidence_gaussian_core SHARED
    csrc/confidence_gaussian/confidence_gaussian.cpp
    csrc/confidence_gaussian/confidence_gaussian_kernel.cu
)

target_include_directories(confidence_gaussian_core PRIVATE csrc/confidence_gaussian)
target_link_libraries(confidence_gaussian_core PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
target_compile_options(confidence_gaussian_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
set_property(TARGET confidence_gaussian_core PROPERTY CXX_STANDARD 17)
set_property(TARGET confidence_gaussian_core PROPERTY POSITION_INDEPENDENT_CODE ON)

message(STATUS "confidence_gaussian_core configured. NOTE: this target has not been "
                "compiled or run in any environment yet -- build and smoke-test it in "
                "Environment B (Colab GPU) before relying on it.")
