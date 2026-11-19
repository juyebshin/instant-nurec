---
name: multi-rig-training
description: Plan and execute the multi-dataset, multi-rig fine-tune of Instant NuRec (docs/PMT_FINETUNE.md §3) — mixing AV2/nuScenes/DDAD/KITTI/OpenScene, varying camera count/frame count/aspect per batch, focal-length and aspect augmentation, held-out rigs (Waymo, HRDX), and the success criteria. Use when adding a dataset to the mix, changing sampling, adding augmentation, or deciding whether the run is ready to launch. Related — inurec-data (per-dataset contracts), data-pipeline-engineer agent (exporters), experiment-scientist agent (protocol).
---

# Multi-Rig Training

`docs/PMT_FINETUNE.md` §3 is the plan of record. This skill is its working checklist. If the
two disagree, the doc wins; update this file.

## Objective and why

The objective is one feed-forward geometry model whose metric scale and shape hold on rigs it was
not trained on.

- Zero-shot, the scale error depends on the rig: range ratio 0.47 on Waymo, 0.70–0.79 on AV2 and
  1.01 on HRDX.
- The AV2 fine-tune fixed AV2 only.
- The scale moves with focal length and aspect, which a single-rig fine-tune never varies.

## Success criteria (stated in advance)

- **Held-out Waymo and HRDX:** range ratio within ±5%, and 10–20 m error approaching the MoGe-3
  control (0.60 / 1.20 m). Scored by mapnn `inurec_zeroshot_diag.py`.
- **In-domain:** no regression vs step 9000 on the AV2 road test and the AV2 mapnn geometry read.
- **Renders:** the next-frame render stays above hold.

## Feasibility pass (do in order, before any long run)

| # | Step | Pass criterion |
|---|---|---|
| 0 | Save on train end (`training/run.py::make_checkpoint_callback`) | a 3-step run with `save_every_n_train_steps: 2` leaves a checkpoint holding step 3 |
| 1 | Export DDAD indexes (PMT exporter), then a DDAD-only smoke (6 cameras, 1936×1216 source) | smoke criteria in `inurec-train-eval`; depth in metres matches the DVGT label on a spot check |
| 2 | Varying camera count and aspect through the training path, step to step | two consecutive steps with different V and aspect run forward, backward and step without recompiling or reshaping errors (the pilot was always 7 cameras at 504×280) |
| 3 | Short AV2 + nuScenes + DDAD probe | peak memory and s/step recorded per dataset; fits the chosen GPU budget |

Report each step's evidence before starting the next one.

## Design pieces and where they go

- **Data.**
  - Use `PMTIndexDataset` as the single loader. Add sources in the PMT exporter.
  - Use DVGT's filtered annotation lists, because label quality varies by dataset.
- **Mixing.**
  - Per-dataset weights are new; there is no PMT mixture yet.
  - `NCoreMixtureDataset` (`datasets/mixture.py`) is the existing pattern for weighted mixing with
    deterministic RNG. Follow it rather than inventing a second scheme.
- **Variable V / T / aspect per batch.**
  - Borrow only the *idea* of DVGT's `DynamicBatchSampler` (`../DVGT/dvgt/datasets/sampler.py`):
    per batch, a random camera count, frame count and aspect under a fixed image budget.
  - Do not borrow its parquet loader.
  - The model has no fixed camera slots, but `KelvinInstantNuRec._grab_metainfo` asserts that every
    item in a batch has the same image and view count. Variability is per *batch*, not within one.
- **Augmentation.** Random crop and scale with focal jitter (DVGT: fx/fy 0.8–1.2, pixel aspect
  0.8–1.0) and a varying output aspect. Apply it inside `CameraSubsampler`-consistent code, so the
  intrinsics follow the pixels.
- **Losses.** Distance, distance gradient, normal and render losses on every dataset. The semantic
  loss applies only where M2F masks exist; elsewhere the target is 255, which is already ignored.

## Open decisions (ask the user; do not choose silently)

- Held-out rigs: Waymo and HRDX, or Waymo in training.
- Keep the 8-class head, or train geometry only.
- Initialise from step 9000 or from the released weights.
- GPU budget. The pilot took ~30 h on one shared H200, and the GPUs are mostly occupied.

## Risks to keep visible in reports

- The semantic head sees only AV2 and nuScenes, so it may drift on other rigs. Report the semantic
  share on a held-out rig.
- Mixing changes the effective data per epoch. An in-domain regression may be dilution, not
  interference. Say which one the run can distinguish.
