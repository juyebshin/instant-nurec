# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

NVIDIA Instant NuRec: a feed-forward model that turns a calibrated multi-camera driving log into a
3D Gaussian Splatting scene in one forward pass. It outputs a static Gaussian layer, dynamic layers,
a sky cubemap and per-camera ISP affines. It is a DAv3 ViT encoder with per-image camera tokens and
DPT decoders. This checkout is on branch **`pmt-finetune`**. The branch adds fine-tuning on non-NCore
logs (AV2 / nuScenes through PseudoMapTrainer index pickles) and an 8-class semantic head.
**Read `docs/PMT_FINETUNE.md` first.** It holds the branch diff, the AV2 pilot results, the known
issues and the multi-rig plan that is the next step.

## Project goal

Use the feed-forward Gaussians as one scene representation that several autonomous-driving tasks
consume: **HD maps** (road Gaussians that initialise RoGS in PseudoMapTrainer, geometry for mapnn),
**occupancy**, and **simulation** (NuRec initialisation, novel-view rendering). How far each one is:
- HD maps: the road path is the only one fine-tuned and measured.
- Occupancy: nothing is built.
- Simulation: upstream's export path, unmodified.

`.claude/rules/research-goals.md` holds the capability table and the standing constraints. Never
describe an unbuilt capability as existing.

## Commands

The environment is conda **`sjy-inurec`** (Python 3.11, torch 2.7.0+cu128, Lightning 2.6.5,
gsplat 1.5.3). The repo is installed editable there. The README's `uv` / `.venv` flow does not apply
on this machine, because `uv` is not installed. `pytest` and `ruff` are not installed in `sjy-inurec`
either; they are in pyproject's `dev` group.

gsplat JIT-compiles on the first render and borrows a CUDA toolchain from `sjy-mapnn`. **Append**
that toolchain to `PATH`; prepending it shadows this env's Python (`ImportError: Self`):

```bash
conda activate sjy-inurec
B=/home/irteam/miniconda3/envs/sjy-mapnn/bin
export CUDA_HOME=/home/irteam/miniconda3/envs/sjy-mapnn PATH=$PATH:$B TORCH_CUDA_ARCH_LIST=9.0 \
       CC=$B/x86_64-conda-linux-gnu-gcc CXX=$B/x86_64-conda-linux-gnu-g++ \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Inference with released weights (`run.sh` and `run_inference.py` are thin wrappers around this):

```bash
instant-nurec --model pa-multiview --ncore-path <seq.json|manifest.lst> --output-dir <dir> \
  [--merge [--n-gaussians N]] [--include-dynamic [--motion-source learned]] [--render-preview | --merge --render-video]
```

The public CLI loads only the released weights-only `.pth` (`INSTANT_NUREC_FULL_PT` overrides the
path) into the 5-class config. To run a fine-tuned Lightning `.ckpt`, strip `model.`, filter to the
`KelvinStaticCore` keys and build the decoder with `n_semantic_classes=8`. That is what PMT
`RoGS/scripts/inurec_road_run.py` and mapnn `tools/analysis_tools/inurec_zeroshot_run.py` do.

Training:

```bash
INSTANT_NUREC_RUN_ID=<run> CUDA_VISIBLE_DEVICES=<gpu> instant-nurec-train --config configs/training/pmt_av2_smoke.yaml
python -c "from pathlib import Path; from instant_nurec.training.run import load_training_config as L; c=L(Path('configs/training/pmt_av2_pilot.yaml')); print(c.phase, c.system.max_epochs, c.model.decoder.n_semantic_classes)"   # validate a config, no GPU
```

`pmt_av2_smoke.yaml` is the one-step mechanics check. `pmt_av2_pilot.yaml` is the 2-epoch AV2 run
(~30 h on one shared H200). The upstream recipes are `full_model_front_{context,render}.yaml`, and
`docs/TRAINING.md` covers them, including resume and the smoke-test contract.

Tests and lint (install `pytest` / `ruff` into the env first; the training extra is needed to collect
the full suite):

```bash
python -m pytest tests/ -q
python -m pytest tests/test_training_data.py -q -k <name>      # one file / one test
ruff check .
```

`conftest.py` puts the repo root on `sys.path`. The ~600 tests are mostly CPU unit tests of upstream
code. **None of them cover this branch's additions** (`PMTIndexDataset`, the 8-class widening,
`semantic_class` targets), so for those the evidence is a smoke run.

**This is a shared machine.** The four H200s are shared with other users. Check `nvidia-smi` before
launching. Never `pkill` / `killall`; kill only PIDs you started, after checking `/proc/<pid>/cmdline`.
Launch anything longer than ~2 minutes detached (`setsid nohup … > log 2>&1 &`) with a log file.

## Architecture

**One set of weights, two compositions.** These are the main cross-file invariant.
- Training: `TrainingSystem` (`training/system.py`, Lightning, manual optimisation) owns
  `KelvinInstantNuRec` (`model/kelvin.py`), which has a *learned* sky decoder (`CubemapDecoderSky`).
- Inference: `make()` (`model/__init__.py`) builds `KelvinStaticCore` (`model/static_core.py`, no
  learned sky; it builds an observed-sky cubemap from the inputs). `KelvinInferenceModel`
  (`model/inference.py`, masking, dynamic layers, packaging) and `GaussiansInstantNuRecSystem`
  (`model/system.py`, predict loop) wrap it.
- Both instantiate the same `KelvinDAv3Encoder` + `KelvinDPTDecoder` + `PerCameraAffinePostProcessing`.
  Any head-shape change must still load into both, and inference loads with `strict=True`.
  `KelvinStaticCore` hard-codes the EGO/SKY/ROAD/MOVABLE indices.

**Geometry is calibration-anchored.**
- Gaussian centres are `ray_origin + ray_dir × predicted distance`. The network predicts only
  distance along known rays.
- Pose and FoV enter as a per-image camera token. The world frame is the first camera's first
  context frame, and the scene is scaled by `scene_rescale = 0.15`.
- There are no fixed camera slots: `camera_idxs` only groups frames for the per-camera affine token.
  The camera count (1/3/5) and image shape are enforced by the CLI and the inference wrapper, not
  by the core.
- Depth labels are **ray distance, not camera z**. `pmt_index.z_to_ray_distance` converts.

**Data contract** (`utils/batch.py`).
- A dataset returns an `InstantNuRecDataBatch`: a per-item `context` list of `DataAndRenderingBatch`,
  optional independently sampled `supervision` frames, and optional `cuboid_tracks`.
- `CameraFrameLabels` carries `rgb`, `metric_distance`, `normals`, `flags` (the `RayFlags` IntFlag in
  `utils/types.py`) and, on this branch, `semantic_class`. When `semantic_class` is set it replaces
  the flag-derived semantic target; 255 means ignore and EGO pixels stay EGO.
- Missing normals are derived from depth in `KelvinInstantNuRec.prepare_context`.
- Datasets: `NCoreInstantNuRecDataset` (NCore V4 plus `.aux.*.zarr` sidecars),
  `NCoreMixtureDataset`, and `PMTIndexDataset` (this branch). `training/data.py::_make_dataset`
  picks one by config type.
- Sampling is deterministic per item (SHA256 of `rng_epoch`, index and seed). A failed item is
  retried up to 10 times with a replacement. `SkipBatchSampler` makes training resumable mid-epoch.
- `CameraSubsampler` (`datasets/instantnurec_base.py`) aspect-crops and then resizes, and the
  intrinsics follow. External tools that prepare inputs must crop the same way. The swapped-branch
  crop bug in PMT and mapnn (`PMT_FINETUNE.md`, fixed 2026-11-19) silently moved the depth scale.

**Config.**
- Configs are pydantic schemas (`config_schema/`). A YAML file goes through
  `TrainingConfig.model_validate`.
- The decoder is a discriminated union on `name` (`dpt-decoder` | point-query). The training composition
  rejects point-query.
- `TrainingConfig.model_post_init` silently overrides YAML values. With `phase: render` it forces
  `enable_render_global_step = 0`, fills in the render-phase loss weights when `loss` is omitted, and
  upgrades `strategy: auto` to `ddp_find_unused_parameters_true`. `INSTANT_NUREC_RUN_ID` overrides
  `run_id`.
- Initialisation: `init_weights_paths.dav3` starts phase 1 from DAv3. `init_weights_paths.full`
  starts from a full checkpoint and normally reinitialises the Gaussian head and the affine. This
  branch's `init_full_keep_heads` keeps both and widens a 5-class semantic head to 8
  (`_widen_semantic_head`).

**Training step.**
- Reconstruct, giving primitives plus a `SupervisionPack`.
- Context losses: distance, distance gradient, normal, semantic CE, RGB, sky, velocity.
- From `enable_render_global_step` on, also render (`training/renderer.py`: gsplat 3DGUT with
  calibrated world rays, sky composite, affine ISP) and add the render losses: RGB MSE, LPIPS,
  inverse distance, background.
- A loss weight of 0 disables that loss. A nonzero weight with a missing label raises, except for
  normal, render-distance and velocity.
- The batch is reduced by mean, not sum.

**Primitives** (`primitives/kelvin_primitive.py`).
- A `KelvinInstantNuRecPrimitive` holds:
  - a `KelvinStaticLayer`: positions, rotations, scales, densities, rgb, optional `semantic_class`
    (uint8) and normals;
  - a list of `KelvinDynamicLayer`s: piecewise-linear keyframe positions with opacity falloff;
  - the sky cubemap and the per-camera affine.
- `KelvinSemanticClass` is positional, so channel index = enum value. The EGO and SKY probabilities
  zero the opacity.
- `KelvinStaticLayer.voxelize` is the KL-optimal merge behind `--merge`.
- `predict/` writes the PLY, the sky `.npz` / `.png`, the dynamic `.npz` and the preview/video renders.

**Sibling repos on this machine** that this one feeds or depends on:

| Where | What |
|---|---|
| `../PseudoMapTrainer/RoGS/scripts/inurec_finetune_index.py` | writes the index pickles `PMTIndexDataset` reads; runs in `sjy-rogs` (AV2 / nuScenes devkits) |
| `../PseudoMapTrainer/RoGS/scripts/inurec_road_{run,test}.py` | road Gaussians for RoGS init, and the road test (dz, IoU vs LiDAR) |
| `../PseudoMapTrainer/RoGS/output/inurec_finetune/` | indexes, runs and checkpoints of the fine-tune |
| `../mapnn/tools/analysis_tools/inurec_{zeroshot_diag,zeroshot_run,render_eval}.py` | geometry vs LiDAR per rig; next-frame render eval |
| `../mapnn/docs/instant-nurec-backbone-study.md` | zero-shot study of this model as a geometry backbone |
| `../DVGT/public_datasets/data_annotation_moge3/` | LiDAR-aligned MoGe-3 depth labels shared with DVGT |

**Style.** ruff, line length 120, isort settings in `pyproject.toml`. Upstream files carry an SPDX
header. Upstream contributions need DCO sign-off (`git commit -s`, `CONTRIBUTING.md`). `.claude/`
is gitignored here, so the agents and skills below are local unless force-added.

## Project agents & skills

`.claude/agents/` and `.claude/skills/` are adapted from `../mapnn/.claude` for this repo. Prefer
them over generic equivalents here.

| Skill | Use for |
|---|---|
| `inurec-train-eval` | smoke / pilot runs, resume, checkpoints, reading PSNR and the downstream evals |
| `inurec-config` | YAML ↔ pydantic schema, `model_post_init` overrides, adding a field, init paths |
| `inurec-data` | the NCore V4 and PMT index contracts, labels, adding a dataset or rig |
| `inurec-env` | `sjy-inurec`, the gsplat JIT toolchain, pinned deps, import / build errors |
| `multi-rig-training` | the `PMT_FINETUNE.md` §3 plan: dataset mixing, varying cameras / aspect, focal jitter, held-out rigs |
| `hdmap-from-gaussians` | road / lane / crosswalk / boundary Gaussians, RoGS init, the road test |
| `occupancy-from-gaussians` | deriving or training 3D semantic occupancy from the Gaussians (not built yet) |
| `simulation-export` | PLY / sky / dynamic export, NuRec init, renders, next-frame render eval |
| `agent-catalog` | which agent to spawn, and on which model tier |

| Agent | Model | Use for |
|---|---|---|
| `inurec-reviewer` | opus | pre-commit diff review: the four principles plus this repo's correctness checks |
| `inurec-explorer` | sonnet | read-only: tracing a value or tensor across files and the sibling repos |
| `inurec-debugger` | sonnet | 5-phase debugging of crashes, NaNs, OOM, metric regressions |
| `data-pipeline-engineer` | sonnet | index exporters, new datasets or rigs, label alignment |
| `task-head-integrator` | sonnet | a new downstream readout or head (occupancy, map classes), default-off |
| `experiment-scientist` | opus | experiment design and result interpretation |
| `scene-researcher` | fable | literature-grounded method research; the only web-enabled agent |

`.claude/rules/` holds the working conventions: `research-goals.md`, `experiments.md`, `testing.md`
and `data-licensing.md`. Read the relevant one before working in that area.

---

# Behavioral guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
