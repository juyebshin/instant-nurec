# Fine-tuning Instant NuRec on PseudoMapTrainer data

Branch `pmt-finetune`. Goal: a feed-forward multi-view geometry model that works across sensor
rigs, whose Gaussians initialise RoGS road reconstruction in PseudoMapTrainer (PMT). Status as of
2026-11-19: single-dataset AV2 fine-tune done (section 2); multi-dataset fine-tune planned
(section 3).

## 1. What this branch adds

| File | Change |
| --- | --- |
| `instant_nurec/datasets/pmt_index.py` (new) | `PMTIndexDataset`: non-NCore logs from per-log index pickles (below). Same batch layout as `NCoreInstantNuRecDataset`: rig = ego, world = first camera's first context frame, global-shutter pinhole cameras, no cuboid tracks. |
| `config_schema/dataset.py` | `PMTIndexDatasetConfig`. `index_paths` entries can be pickles or directories of pickles. |
| `training/data.py` | Builds `PMTIndexDataset` for that config. |
| `config_schema/dataset.py`, `datasets/mixture.py` | A `mixture` component may be a `PMTIndexDatasetConfig`, so datasets with different camera sets (AV2 7, nuScenes 6) mix in one run. |
| `config_schema/models.py`, `model/backbone/decoders.py` | `KelvinDPTDecoderConfig.n_semantic_classes: 5 \| 8`. `KelvinModelConfig.init_full_keep_heads`: fine-tune the released model, keeping its Gaussian head and affine post-processing. |
| `primitives/kelvin_primitive.py` | `KelvinSemanticClass` gains LANE 5, CROSSWALK 6, BOUNDARY 7. With them, ROAD means road surface without paint. |
| `training/system.py` | `_widen_semantic_head`: a released 5-class head is widened to 8. LANE and CROSSWALK start from the ROAD row, BOUNDARY from the OTHERS row, all with bias −2 so the argmax at step 0 is unchanged. |
| `training/losses.py`, `utils/batch.py` | Optional `CameraFrameLabels.semantic_class` (255 = ignore). When set, it is the semantic-loss target instead of the class implied by the ray flags. EGO pixels stay EGO. |
| `configs/training/pmt_av2_smoke.yaml`, `pmt_av2_pilot.yaml` (new) | One-step smoke config and the 2-epoch AV2 pilot. |
| `configs/training/pmt_av2_nusc.yaml`, `pmt_av2_nusc_smoke.yaml` (new) | AV2 + nuScenes fine-tune with 504×280 supervision, and its smoke at the same contract. |

**Index pickles.** These are written by PMT `RoGS/scripts/inurec_finetune_index.py`, which runs
in env `sjy-rogs` because that env has the AV2 and nuScenes devkits; this env does not. The
exporter currently handles AV2 and nuScenes.

```
cameras: {name: {K, wh, T_cam_ego}}             # OpenCV camera frame
rig: {timestamps_us, T_ego_world}               # ego pose at every image timestamp
frames: {name: [{t_us, img, seg, depth}, ...]}  # every image; seg / depth may be None
semantic_lut: (256,) uint8                      # Mapillary Vistas v2 id -> KelvinSemanticClass, 255 = ignore
```

**Labels.**
- **Depth:** DVGT's LiDAR-aligned MoGe-3 depth (`DVGT/public_datasets/data_annotation_moge3/align_depth`).
  It is stored as log-encoded uint16 camera-z PNGs and converted to ray distance in
  `pmt_index.read_log_depth` / `z_to_ray_distance`.
- **Normals:** derived from that depth in `KelvinInstantNuRec.prepare_context` (`model/kelvin.py`),
  so the `primitive_normal` loss is active.
- **Semantics:** M2F Mapillary-v2 masks from the RoGS label dirs, mapped to the 8 classes.
- **Coverage:** depth exists only on 2 Hz keyframes. `context_requires_depth` draws context
  frames from those.

AV2 index: 695 logs with MoGe-3 depth, split 491 / 101 / 103 into train / val / test under
`PseudoMapTrainer/RoGS/output/inurec_finetune/index/av2/{train,val,test}`. Test is held out.

nuScenes index (2026-11-20, `logs/export_nusc_all.sh`): the official v1.0-trainval split, 700 train / 150 val
scenes under `.../index/nusc/{train,val}`. Every keyframe has MoGe-3 depth (DVGT's filter lists keep all
34,149) and every frame, sweeps included, has an M2F mask. 6 cameras at 1600×900, ~40 depth keyframes per
camera per scene.

**Environment.** `sjy-inurec` alone. Since 2026-11-20 it carries its own gsplat JIT toolchain (conda
`cuda-nvcc` 12.8, `gxx_linux-64` 11.4); until then it borrowed `sjy-mapnn`'s. From the repo root:

```bash
conda activate sjy-inurec                      # sets CC / CXX to its gcc 11.4
export CUDA_HOME=$CONDA_PREFIX TORCH_EXTENSIONS_DIR=$PWD/.cache/torch_extensions \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_USE_COMM_NONBLOCKING=0 NCCL_COMM_BLOCKING=1   # multi-GPU: without these DDP hangs at random steps
INSTANT_NUREC_RUN_ID=<run> CUDA_VISIBLE_DEVICES=<gpu> instant-nurec-train --config configs/training/pmt_av2_pilot.yaml
```

`scripts/train_pmt_av2_nusc.sh` wraps this for the AV2 + nuScenes run and writes checkpoints and the
console log under `checkpoints/training/` in this repo. (The AV2 pilot configs still write to the PMT
output directory, where the pilot's checkpoints are.)

## 2. AV2 pilot (`pmt_av2_pilot.yaml`, run `pmt-av2-pilot-2ep`)

**Setup.**
- Released `instant_nurec_pa_multiview_1.1.0.pth`, encoder unfrozen, 8-class head.
- Context: T=8 frames × 7 ring cameras at 504×280, 0.5 s apart.
- Supervision: 1 random frame per camera plus the context frames (`include_context_frames`), at
  252×140. Full 504×280 supervision ran out of memory at ~36 GB on the shared GPU.
- Render phase from step 0. lr 2e-5 (1/5 of upstream, because the encoder is unfrozen), warmup 500.
- 2 epochs × 4910 steps at batch 1. Peak ~25 GB, ~11 s/step on a shared H200 (~30 h).

**Run.**
- Finished all 9,820 steps. Final val PSNR 22.99 (22.72 after epoch 1).
- Outputs in `PseudoMapTrainer/RoGS/output/inurec_finetune/runs/pmt-av2-pilot-2ep`.

**Known issues.**
- **Final weights not saved.** `last.ckpt` holds step 9000, not 9820. `make_checkpoint_callback`
  (`training/run.py`) uses `every_n_train_steps` with `save_last`, which never saves at train end,
  so up to `save_every_n_train_steps − 1` steps are lost. The LR schedule moves only per epoch
  (`current_epoch / max_epochs`), so those were full-rate steps. **Fixed 2026-11-20**
  (`_StepCheckpoint`: also saves the final step; verified on a 4-step DDP run saving every 3).
- **Sky head (`sky.*`) is trained from scratch.** It is the render background
  (`training/renderer.py`) and reads `encoded_latent` without detach. Released inference builds an
  observed-sky cubemap instead. Left as is.
- **Lane and crosswalk classes are rare.** At step 9000, LANE is argmax on 0.4–1.1% of pixels and
  CROSSWALK on ≤ 0.1%. At step 1000 neither class was ever argmax.

### Evaluation (all on logs outside the 695 training/val/test logs, or in val)

**Road test** (PMT `RoGS/scripts/inurec_road_run.py` + `inurec_road_test.py vis`). ROAD-surface
Gaussians from T=8 windows over a whole log, against the LiDAR road in a 0.1 m BEV grid. dz is the
median height error vs LiDAR. Log 7fab is in no PMT split; 02678d04 and 0b5142c1 are in val.

| Log | Model | dz | median \|dz\| | IoU |
| --- | --- | --- | --- | --- |
| 7fab | zero-shot | +0.66 m | 0.66 | 0.485 |
| 7fab | step 1000 | −0.07 m | 0.07 | 0.608 |
| 7fab | step 9000 | −0.02 m | 0.025 | 0.584 |
| 02678d04 | MoGe-3 labels | −0.01 m | 0.038 | 0.500 |
| 02678d04 | zero-shot | +0.66 m | 0.66 | 0.455 |
| 02678d04 | step 1000 | −0.06 m | 0.06 | 0.503 |
| 02678d04 | step 9000 | +0.01 m | 0.027 | 0.488 |
| 0b5142c1 | MoGe-3 labels | −0.00 m | 0.026 | 0.665 |
| 0b5142c1 | zero-shot | +0.59 m | 0.60 | 0.535 |
| 0b5142c1 | step 1000 | −0.03 m | 0.06 | 0.689 |
| 0b5142c1 | step 9000 | +0.03 m | 0.039 | 0.652 |

- **Height:** step 9000 reaches the MoGe-label level.
- **Coverage:** IoU is 2–4 points below step 1000. The road pixel share is unchanged, so the loss
  is in where road points land, not in the semantics. Cause not found.

**Next-frame render** (mapnn `tools/analysis_tools/inurec_render_eval.py --ckpt`).
- Data: 100 AV2 samples from 150 logs, none of them among the 695 PMT logs.
- Method: render the 8-frame context's static Gaussians into the held-out next frame (+0.5 s).
- Score: static pixels; PSNR / SSIM / LPIPS.

| | zero-shot (fixed crop) | step 9000 |
| --- | --- | --- |
| rendered next frame | 15.52 / 0.469 / 0.571 | 19.43 / 0.624 / 0.357 |
| copy previous frame ("hold") | 17.07 / 0.454 / 0.336 | 17.14 / 0.457 / 0.335 |
| re-render newest input | 16.57 / 0.529 / 0.509 | 22.41 / 0.725 / 0.286 |

- The rendered next frame now beats hold on PSNR (+2.3 dB) and SSIM. Zero-shot was −1.55 dB behind.
- LPIPS still favours hold.
- Hold differs between the columns because the static mask is each model's own semantic head.

**Visualisations.** Per camera: input, depth, LiDAR depth, semantic and normal, plus BEV and PLY.
- Step 9000: `PseudoMapTrainer/RoGS/output/inurec_road/av2_ft9000/{vis,render}/png`.
- Zero-shot with the fixed crop: `.../inurec_road/av2/vis/png`.
- Script: `.../inurec_road/ft9000_vis.sh`.

**Train/inference contract check.** Same frames, same checkpoint: the training path and the
road-test input path agree. Median predicted / label distance on road is 1.020 vs 1.013.

**Input-crop bug (fixed 2026-11-19).** The aspect-crop branches in the road-test and mapnn
`prepare_view` were swapped. With the `min()` clamp, the result was no crop at all, so images were
resized anisotropically to 504×280:

| Dataset | Squash |
| --- | --- |
| AV2 ring | 1.36× |
| AV2 portrait front | 2.38× |
| Waymo | 1.20× |
| HRDX | 1.16× |
| nuScenes | 1.01× |

Training's `CameraSubsampler` was never affected. All numbers above use the fix. The earlier
"−0.47 m overshoot" after fine-tuning was this bug.

## 3. Plan: multi-dataset fine-tune across sensor rigs

**Objective.** One feed-forward geometry model whose metric depth scale and shape hold on rigs it
was not trained on.

**Why.**
- Zero-shot, the released model's scale error depends on the rig. The AV2 fine-tune fixed AV2 only.
- Zero-shot results (mapnn `docs/instant-nurec-backbone-study.md`, fixed crop; median xy error
  vs LiDAR at 10–20 m, T=1 / T=8, and median predicted / LiDAR range):

  | Rig | Instant-NuRec zero-shot | range ratio | MoGe-3 control (T=1) |
  | --- | --- | --- | --- |
  | AV2 (7 ring, portrait front) | 3.93 / 2.78 m | 0.70 / 0.79 | 0.85 m |
  | nuScenes (6 cams) | 4.20 m | — | 1.07 m |
  | Waymo (5 cams) | 6.91 / 6.97 m | 0.47 | 0.60 m |
  | HRDX | 1.91 / 1.60 m | 1.01 | 1.20 m |

- With a per-camera scale correction the AV2 error is 1.89 m, as it was before the crop fix. The
  crop bug therefore moved the depth scale, not the shape: the scale moves with focal length and
  aspect, which a single-rig fine-tune never varies.

**What already supports it.**
- **The model has no fixed camera slots.** `camera_idxs` only groups each camera's frames for the
  per-camera affine token (`model/post_processing.py`). Any camera count works without
  architecture changes.
- **Upstream trained multiview with a random 1/3/5-camera, 12/16/18-frame curriculum.** That
  curriculum is not in this repo (`docs/TRAINING.md`).
- **The labels are shared with DVGT.** DVGT's LiDAR-aligned MoGe-3 depth already exists for:

  | Dataset | Aligned MoGe-3 depth | M2F semantics |
  | --- | --- | --- |
  | AV2 | 695 logs | yes |
  | nuScenes | keyframes | yes |
  | DDAD | 150 scenes | no |
  | KITTI | 5 date folders; 74.7k frames in the filtered train list | no |
  | OpenScene | 9 entries | no; completeness unchecked |

  DVGT's filtered annotation lists are in `DVGT/data_annotation/meta/moge3_vitl/moge_pred_filter_parquet`.
- **Waymo and HRDX have no aligned MoGe-3 labels here.** That makes them natural held-out rigs,
  and mapnn's `inurec_zeroshot_diag.py` already scores them against LiDAR.

**Design.**
1. **Data.**
   - Extend the PMT exporter to DDAD, then KITTI and OpenScene if their labels are complete. Keep
     `PMTIndexDataset` as the single loader, since its training contract is checked.
   - Mix datasets with per-dataset weights.
   - Borrow only DVGT's `DynamicBatchSampler` (`dvgt/datasets/sampler.py`) idea: per batch, a random
     camera count, frame count and aspect ratio under a fixed image budget. Not its parquet loader.
2. **Augmentation.** Random crop and scale with focal-length jitter (DVGT: fx/fy 0.8–1.2, pixel
   aspect 0.8–1.0) and a varying output aspect. This targets the focal/aspect sensitivity directly.
3. **Losses.**
   - Distance, distance-gradient, normal and render losses on every dataset.
   - Semantic loss only where M2F masks exist (AV2, nuScenes). Elsewhere the target is 255, which
     the loss already ignores.
4. **Initialisation.** The step-9000 AV2 checkpoint or the released weights (open, below).
5. **Prerequisite.** Save on train end (the `last.ckpt` gap above).

**Success criteria.**
- **Held-out Waymo and HRDX:** range ratio within ±5%, and the 10–20 m error approaches the MoGe-3
  control (0.60 / 1.20 m).
- **In-domain:** no regression vs step 9000 on the AV2 road test and the AV2 mapnn read.
- **Renders:** next-frame render stays above hold.

**Feasibility pass, before any long run.**
1. Export DDAD indexes, then a smoke test with DDAD alone (6 cameras, 1936×1216).
2. Verify that the training path accepts a varying camera count and aspect from step to step.
   The pilot always used 7 cameras at 504×280.
3. Run a short AV2 + nuScenes + DDAD probe for peak memory and s/step.

**AV2 + nuScenes, before DDAD (2026-11-20, `pmt_av2_nusc.yaml`).** Same recipe as the pilot except the
two-dataset mixture (4,910 AV2 + 4,900 nuScenes samples per epoch) and supervision at the 504×280
profile resolution instead of 252×140. Val: a fixed draw of 25 AV2 val logs + 25 nuScenes val scenes, so
its `val/psnr` is not comparable to the pilot's.
- Smoke (`pmt_av2_nusc_smoke.yaml`, GPU 0 unshared): both camera sets train at batch 1, so the camera
  count varies from step to step (7 ↔ 6) with no code change. Aspect stays 504×280; varying aspect is
  still unverified. Every loss finite (velocity skipped: no tracks). Resume from a mid-epoch checkpoint
  reproduces the same losses and crosses the epoch boundary. Peak ~50 GB (nvidia-smi).
- Throughput probe (full config, 16 workers, 30 steps): 7.8 s/step including start-up, so 2 epochs
  (19,620 steps) ≈ 42 h on one unshared H200.
- Launch: `scripts/train_pmt_av2_nusc.sh` (sets the NCCL blocking env below; `GPUS` overrides the
  default 0,1,2,3). Config: 4 GPUs × batch 2 (total 8), lr 5.7e-5 (the pilot's 2e-5 × √8), 1,227 steps /
  epoch. Measured so far: 2 GPUs × batch 1 on free GPUs 8.7 s/step (~51 GB); batch 2 peaks ~80 GB per GPU;
  s/step at batch 2 not measured.
- **4-GPU DDP hung intermittently; fixed by blocking NCCL comms (2026-11-20).** Without the fix the full
  config on GPUs 0–3 hung in 6 of 7 runs, at a random step and rank. Native stacks (py-spy) of the stuck
  rank: its main thread blocked inside NCCL's own enqueue (`ncclCommGetAsyncError` → `ncclGroupJobComplete`,
  the non-blocking-communicator path) on a tiny per-step collective; the other ranks wait for it. NCCL's
  per-op log shows identical collective sequences on all ranks, and `TORCH_DISTRIBUTED_DEBUG=DETAIL` found no
  mismatch. Not the cause: NVLS, `expandable_segments`, the pin-memory thread. **Fix:** launch with
  `TORCH_NCCL_USE_COMM_NONBLOCKING=0 NCCL_COMM_BLOCKING=1`; 2 of 2 runs then completed 60 steps on 4 GPUs.
  Side finding: `model/backbone/sky.py:140` builds a buffer on `torch.device("cuda")` in `__init__`, so
  every rank opens a ~600 MB context on GPU 0.
- **4-GPU throughput is poor here:** 24.6 s/step at batch 4 (6.2 s/sample) vs 7.8 s/sample on one GPU,
  with GPU 3 shared with another user's job (80–100% util). Two epochs ≈ 34 h on 4 GPUs vs ≈ 42 h on 1.
- **Per-GPU batch > 1 (2026-11-20).** A batch must share its context image count (AV2 56, nuScenes 48),
  so a mixture with `train_batch_size` / `val_batch_size` > 1 uses `GroupedSkipBatchSampler`
  (`training/data.py`): one component per batch, the batch list a function of (seed, epoch), dealt
  `rank::world` with padding so every rank runs the same steps. Batch 1 keeps the old sampler. Peak
  (nvidia-smi): 1/GPU ~50 GB; 2/GPU 80.1 GB on AV2 alone, ~70 GB for a nuScenes batch; 4/GPU ~140 GB
  (extrapolated, ~30 GB per sample), which does not fit. 2/GPU is at the limit for 80 GB cards.
  Verified: unit tests (`tests/test_grouped_batch_sampler.py`) and a 2-GPU × batch-2 mixture smoke.
- Resume edge case: a checkpoint written on an epoch's last batch resumes into the next epoch with the
  batch cursor still at the end, and that epoch trains no steps (`training/data.py` `SkipBatchSampler`).
  Not hit when the epoch length is not a multiple of `save_every_n_train_steps`.

**Risks.**
- **Compute.** The single-dataset pilot took ~30 h on one shared GPU, and the box's GPUs are
  mostly occupied.
- **Label quality varies by dataset.** Use DVGT's filtered lists.
- **The semantic head sees only AV2 and nuScenes.** It may drift on other rigs.

**Open decisions.**
- Held-out rigs: Waymo and HRDX, or Waymo in training.
- Keep the 8-class semantic head or train geometry only.
- Initialise from step 9000 or from the released weights.
- GPU budget.
