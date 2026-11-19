---
name: inurec-config
description: Author or modify Instant NuRec training/predict YAML configs and the pydantic schemas behind them. Use when adding a config variant, adding a config field, changing cameras/resolution/frames/loss weights, choosing init_weights_paths vs resume_from_checkpoint, or when a config validates but the run does something other than what the YAML says. Related — inurec-train-eval (running it), inurec-explorer agent (tracing a field to its use).
---

# Instant NuRec Config & Schema

## How a YAML becomes a run

`instant-nurec-train --config X.yaml` → `yaml.safe_load` → `TrainingConfig.model_validate`
(`config_schema/train.py`) → `TrainingSystem` and `TrainingDataModule`. The schema *is* the API, and
there is no registry. To find what a key does, find its `Field` in `config_schema/` and grep the
attribute name.

| Schema | File | Notes |
|---|---|---|
| `TrainingConfig`, `TrainingSystemConfig`, `TrainingLossConfig`, optimizer / scheduler / logger | `config_schema/train.py` | `TrainingLossConfig.context_phase()` / `.render_phase()` are the upstream weights |
| `KelvinModelConfig`, encoder / decoder / sky / post-processing | `config_schema/models.py` | decoder = discriminated union on `name` |
| Dataset splits: `PMTIndexDatasetConfig`, `NCoreInstantNuRecDatasetConfig`, `NCoreMixtureDatasetConfig` | `config_schema/dataset.py` | union *without* a discriminator; pydantic picks by fields |
| Predict / CLI | `config_schema/predict.py`, `config_schema/instantnurec.py` | built by `cli.py` from the release profile |

## Silent overrides: check the resolved values

`TrainingConfig.model_post_init` rewrites the YAML:
- `phase: render` → `system.enable_render_global_step = 0`, whatever the YAML says.
- `phase: render` and no `loss:` block → `TrainingLossConfig.render_phase()`. A `loss:` block
  replaces the defaults **entirely**. Unlisted weights take the *field* defaults, which are the
  context-phase values: `rgb`, `lpips`, `distance` and `background` all default to **0**. A
  render-phase `loss:` block that omits `rgb` therefore trains without the render RGB loss. List
  every weight, as `pmt_av2_pilot.yaml` does.
- `phase: render` and `strategy: auto` → `ddp_find_unused_parameters_true`.
- The `INSTANT_NUREC_RUN_ID` env var → `run_id`.

The dataset union has no discriminator. A typo in a PMT field can make pydantic try another member
and fail with a confusing error about NCore fields. Read the error for the member you meant.

## Initialisation vs resume

| Key | Effect |
|---|---|
| `model.init_weights_paths.dav3` | phase-1 start from the DAv3 safetensors. Context and Gaussian heads are zero-initialised |
| `model.init_weights_paths.full` | new run from a full checkpoint (released `.pth` or Lightning `.ckpt`). The Gaussian head and affine are **re-initialised** |
| `+ model.init_full_keep_heads: true` | (branch) keep the Gaussian head and affine; widen a 5-class semantic head to `decoder.n_semantic_classes` |
| `resume_from_checkpoint` | continue the same run: optimiser, scheduler, step and data cursor |

Fine-tuning the released model means `full` + `init_full_keep_heads: true`, as in `pmt_av2_pilot.yaml`.
Without `keep_heads` you start the render phase from a random Gaussian head.

## Adding a field

- **Additive only.** A new field needs a default that reproduces current behaviour, so the released
  `parsed.yaml`, the upstream configs and every `resolved.yaml` still load and mean the same thing.
- Never rename or repurpose a field. Constrain it with `Literal` or `Field(ge=…)` the way the
  neighbours do (`n_semantic_classes: Literal[5, 8]`).
- A model-shape field (head channels, class count) must also reach the inference composition
  (`KelvinStaticCore`) and the downstream loaders that rebuild `KelvinModelConfig` by hand (PMT
  `inurec_road_run.py`, mapnn `inurec_zeroshot_run.py`).
- New config files: copy the closest existing YAML and change only what the experiment varies. Say
  what changed in the header comment, as the PMT configs do.

## Values that fan out

| Value | Must agree with |
|---|---|
| `camera_subsampler.{frame_width,frame_height}` | the released profile (504×280 for `pa-multiview`), the input prep of every external eval script, `patch_shape` 14 divisibility |
| `n_frames_per_sample`, `frame_gap_us` | depth coverage (`context_requires_depth`: 2 Hz keyframes only), memory |
| `supervision_frame_batch.camera_subsampler` | memory (504×280 supervision OOMed at ~36 GB), and the render `distance` loss sees depth only if `include_context_frames: true` |
| `decoder.n_semantic_classes` | `KelvinSemanticClass`, the exporter's `semantic_lut`, the inference loaders |
| `system.save_every_n_train_steps` | the `last.ckpt` gap (see `inurec-train-eval`) |
| `system.train_num_workers` | CPU decode cost (~20 s per AV2 sample on one core) |

## Checklist before running

- [ ] `load_training_config` passes, and the printed resolved values match your intent.
- [ ] Index paths point at the intended split; the test split is not in train or val.
- [ ] Init path and `init_full_keep_heads` are what you mean.
- [ ] The YAML header comment says what this config varies.
