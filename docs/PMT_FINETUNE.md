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
| `config_schema/models.py`, `model/backbone/decoders.py` | `KelvinDPTDecoderConfig.n_semantic_classes: 5 \| 8`. `KelvinModelConfig.init_full_keep_heads`: fine-tune the released model, keeping its Gaussian head and affine post-processing. |
| `primitives/kelvin_primitive.py` | `KelvinSemanticClass` gains LANE 5, CROSSWALK 6, BOUNDARY 7. With them, ROAD means road surface without paint. |
| `training/system.py` | `_widen_semantic_head`: a released 5-class head is widened to 8. LANE and CROSSWALK start from the ROAD row, BOUNDARY from the OTHERS row, all with bias −2 so the argmax at step 0 is unchanged. |
| `training/losses.py`, `utils/batch.py` | Optional `CameraFrameLabels.semantic_class` (255 = ignore). When set, it is the semantic-loss target instead of the class implied by the ray flags. EGO pixels stay EGO. |
| `configs/training/pmt_av2_smoke.yaml`, `pmt_av2_pilot.yaml` (new) | One-step smoke config and the 2-epoch AV2 pilot. |

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

**Environment.** `sjy-inurec`, plus the gsplat JIT toolchain from `sjy-mapnn`:

```bash
B=/home/irteam/miniconda3/envs/sjy-mapnn/bin
export CUDA_HOME=/home/irteam/miniconda3/envs/sjy-mapnn PATH=$PATH:$B TORCH_CUDA_ARCH_LIST=9.0 \
       CC=$B/x86_64-conda-linux-gnu-gcc CXX=$B/x86_64-conda-linux-gnu-g++ \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INSTANT_NUREC_RUN_ID=<run> CUDA_VISIBLE_DEVICES=<gpu> instant-nurec-train --config configs/training/pmt_av2_pilot.yaml
```

Append `sjy-mapnn/bin` to `PATH` rather than prepending it. Otherwise its Python 3.10 shadows this
env's Python in scripts that call `python` (`ImportError: Self`).

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
  (`current_epoch / max_epochs`), so those were full-rate steps. Not fixed yet.
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
