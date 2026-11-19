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

The first render JIT-compiles CUDA kernels, which can take several minutes. The toolchain comes from
`sjy-mapnn`:

```bash
B=/home/irteam/miniconda3/envs/sjy-mapnn/bin
export CUDA_HOME=/home/irteam/miniconda3/envs/sjy-mapnn PATH=$PATH:$B TORCH_CUDA_ARCH_LIST=9.0 \
       CC=$B/x86_64-conda-linux-gnu-gcc CXX=$B/x86_64-conda-linux-gnu-g++ \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

| Symptom | Cause / fix |
|---|---|
| `ImportError: Self` (or any `typing` import error) in a subprocess | `sjy-mapnn/bin` was **prepended** to `PATH`, so its Python 3.10 shadows 3.11. Append it instead |
| gsplat compile error about `nvcc` / the compiler | `CUDA_HOME`, `CC` or `CXX` not exported in *this* shell; a detached job inherits only what was exported |
| Compiles every run | the extension cache dir differs between runs or workers. gsplat builds into torch's `_get_build_directory`, so keep `TORCH_EXTENSIONS_DIR` stable |
| Need to see the build | `VERBOSE=1 python -c "from gsplat.cuda._backend import _C"` (gsplat's own debug trigger). Explicit `BUILD_3DGUT` / `NUM_CHANNELS` env values override Instant NuRec's limited build |
| `no kernel image is available` | `TORCH_CUDA_ARCH_LIST` doesn't match the GPU (the H200s are 9.0) |

## Other common errors

| Symptom | Cause |
|---|---|
| `PretrainedModelError: Could not download …` | no network access to HF. Set `INSTANT_NUREC_FULL_PT` to the local `checkpoints/pth/*.pth` (`TROUBLESHOOTING.md`) |
| `ModelCheckpointError … legacy traced-model archive` | wrong artifact for `INSTANT_NUREC_FULL_PT` |
| `Error(s) in loading state_dict` loading a fine-tune into the CLI | the CLI expects the released 5-class weights-only file. Load fine-tunes the way PMT `inurec_road_run.py` does |
| `apex` warning at startup | expected: `optimizer.implementation: auto` falls back to `torch.optim.Adam` |
| `ModuleNotFoundError: av2` / `nuscenes` | expected in this env. Exporters run in `sjy-rogs` |

## Before proposing a dependency change

State which pin changes, why, what else in `uv.lock` moves with it, and whether the released
checkpoints and the gsplat commit pinned in `[tool.uv.sources]` still work. That gsplat commit fixes
F-theta culling and exposes world-ray 3DGUT, which calibrated rendering needs. Then ask.
