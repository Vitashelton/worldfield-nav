# Environment Audit

Audit date: 2026-09-01
Purpose: Habitat-GS standalone smoke-test environment gate for mobile-robot visual navigation research. No installation was performed during this audit.

## Hardware gate

- Purchased specification to verify: `vGPU-48GB-350W`.
- Actual CUDA-visible GPU: `NVIDIA GeForce RTX 3090`.
- `nvidia-smi` visible memory: `49152 MiB` (48.00 GiB; approximately the specified 48 GB tier).
- PyTorch-visible memory: `47.41 GiB`.
- Power cap: `350 W`.
- Gate result: **PASS**. The visible CUDA memory is approximately 48 GB, so proceeding with the isolated-environment phase is permitted.

## GPU and CUDA

- NVIDIA driver: `580.105.08`.
- Driver-reported CUDA compatibility: `13.0`.
- CUDA compiler (`nvcc`): `12.1.105` (`/usr/local/cuda-12.1/bin/nvcc`).
- CUDA toolkit path: `/usr/local/cuda-12.1`.
- `CUDA_HOME`: unset in the login shell.
- No driver or system CUDA changes were made.

## Language and build tools

- Login-shell Python: `3.12.3`.
- Existing base PyTorch: `2.3.0+cu121`; CUDA runtime `12.1`; CUDA available: `True`.
- CMake: `3.22.1`.
- gcc: `11.4.0`.
- g++: `11.4.0`.

## EGL / OpenGL

- NVIDIA EGL runtime library detected: `libEGL_nvidia.so.0`.
- GLX runtime library detected: `libGLX_nvidia.so.0`.
- OpenGL runtime libraries detected: `libGL.so.1`, `libOpenGL.so.0`.
- `eglinfo` and `glxinfo` are not installed, so their diagnostic output was unavailable. This audit did not install them.

## Storage

| Mount | Total | Used | Available | Notes |
| --- | ---: | ---: | ---: | --- |
| `/` | 30G | 53M | 30G | System disk; no large artifacts allowed |
| `/root/autodl-tmp` | 210G | 12K | 210G | Fast data disk; project root |

## Conda state

- `conda`: `/root/miniconda3/bin/conda`.
- Existing environments: base only, at `/root/miniconda3`.
- Base environment was not modified.
- Required independent environment prefix for this project: `/root/autodl-tmp/worldfield_nav/envs/worldfield_nav`.

## Required project locations

- Project root: `/root/autodl-tmp/worldfield_nav`.
- Documentation: `/root/autodl-tmp/worldfield_nav/docs`.
- Environments: `/root/autodl-tmp/worldfield_nav/envs`.
- Caches: `/root/autodl-tmp/worldfield_nav/cache`.
