---
name: inurec-env
description: Diagnose Instant NuRec environment, install, CUDA JIT and import problems. Use on gsplat/nvdiffrast build or JIT failures, "ImportError: Self", CUDA/driver mismatches, missing pytest/ruff, HF download failures, or before proposing any dependency change. Related — inurec-train-eval (running), TROUBLESHOOTING.md (inference CLI errors).
---

# Instant NuRec Environment

## The env

Conda env `sjy-inurec`:
- Python 3.11.18, torch 2.7.0+cu128, pytorch-lightning 2.6.5, gsplat 1.5.3.
- `instant_nurec` installed editable from this repo, so `instant-nurec` and `instant-nurec-train` are
  on `PATH`.
- `uv` is not installed on this machine. The README / `setup.sh` flow (`uv sync --frozen`) documents
  the pinned set in `uv.lock` but is not how this env was built.
- No `pytest` or `ruff`. They are in pyproject's `dev` group. Ask before installing anything into a
  shared env.

Every pin in `pyproject.toml` is deliberate: "parity / runtime drift is attributable to a deliberate
dep bump". **Never upgrade a pinned dependency to make an error go away.** Report it and let the
user decide.

## gsplat / nvdiffrast JIT

The first render JIT-compiles CUDA kernels, which can take several minutes. The toolchain is in
`sjy-inurec` itself (conda `cuda-nvcc` 12.8 + `gxx_linux-64` 11.4, added 2026-11-20); `sjy-mapnn` is
no longer needed. Installed with (`openssl` pinned so the solver leaves it alone):

```bash
conda install -n sjy-inurec --freeze-installed -c nvidia/label/cuda-12.8.1 -c conda-forge \
  cuda-nvcc cuda-cudart-dev cuda-cccl libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev \
  gxx_linux-64=11.4 gcc_linux-64=11.4 openssl=3.5.9
```

The compiler alone is not enough: torch's `ATen/cuda/CUDAContextLight.h` includes `cusparse.h`,
`cublas_v2.h`, `cublasLt.h` and `cusolverDn.h` (`fatal error: cusparse.h: No such file`). Conda puts
them in `$CONDA_PREFIX/targets/x86_64-linux/include`. From the repo root:

```bash
conda activate sjy-inurec                      # sets CC / CXX to its gcc 11.4
export CUDA_HOME=$CONDA_PREFIX TORCH_EXTENSIONS_DIR=$PWD/.cache/torch_extensions \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_USE_COMM_NONBLOCKING=0 NCCL_COMM_BLOCKING=1   # multi-GPU: without these DDP hangs at random steps
```

| Symptom | Cause / fix |
|---|---|
| `ImportError: Self` (or any `typing` import error) in a subprocess | another env's `bin` (e.g. the old `sjy-mapnn` toolchain) is ahead of `sjy-inurec` on `PATH`, so its Python 3.10 shadows 3.11 |
| gsplat compile error about `nvcc` / the compiler | `sjy-inurec` not activated in *this* shell (it sets `CC` / `CXX`), or `CUDA_HOME` not exported; a detached job inherits only what was exported |
| Compiles every run | the extension cache dir differs between runs or workers. gsplat builds into torch's `_get_build_directory`, so keep `TORCH_EXTENSIONS_DIR` stable |
| Need to see the build | `VERBOSE=1 python -c "from gsplat.cuda._backend import _C"` (gsplat's own debug trigger). Explicit `BUILD_3DGUT` / `NUM_CHANNELS` env values override Instant NuRec's limited build |
| `no kernel image is available` | `TORCH_CUDA_ARCH_LIST` set to an arch other than the GPU's (H200 9.0); leave it unset so the build targets the visible GPUs |

## Other common errors

| Symptom | Cause |
|---|---|
| `PretrainedModelError: Could not download …` | no network access to HF. Set `INSTANT_NUREC_FULL_PT` to the local `checkpoints/pth/*.pth` (`TROUBLESHOOTING.md`) |
| `ModelCheckpointError … legacy traced-model archive` | wrong artifact for `INSTANT_NUREC_FULL_PT` |
| `Error(s) in loading state_dict` loading a fine-tune into the CLI | the CLI expects the released 5-class weights-only file. Load fine-tunes the way PMT `inurec_road_run.py` does |
| `apex` warning at startup | expected: `optimizer.implementation: auto` falls back to `torch.optim.Adam` |
| `ModuleNotFoundError: av2` / `nuscenes` | installed 2026-11-25 for `tools/index_export` (`--no-deps`, README there); reinstall with that recipe, not a plain `pip install`, which would pull `opencv-python` over the env's headless `cv2` |

## Before proposing a dependency change

State which pin changes, why, what else in `uv.lock` moves with it, and whether the released
checkpoints and the gsplat commit pinned in `[tool.uv.sources]` still work. That gsplat commit fixes
F-theta culling and exposes world-ray 3DGUT, which calibrated rendering needs. Then ask.
