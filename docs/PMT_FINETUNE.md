# Fine-tuning Instant NuRec on PseudoMapTrainer data

Branch `pmt-finetune`. Goal: a feed-forward multi-view geometry model that works across sensor
rigs, whose Gaussians initialise RoGS road reconstruction in PseudoMapTrainer (PMT). Status as of
2026-11-19: single-dataset AV2 fine-tune done (section 2); multi-dataset fine-tune planned
(section 3).

## 1. What this branch adds

| File | Change |
| --- | --- |
| `instant_nurec/datasets/pmt_index.py` (new) | `PMTIndexDataset`: non-NCore logs from per-log index pickles (below). Same batch layout as `NCoreInstantNuRecDataset`: rig = ego, world = first camera's first context frame, global-shutter pinhole cameras, no cuboid tracks (unless `cuboid_tracks` is set, below). |
| `config_schema/dataset.py` | `PMTIndexDatasetConfig`. `index_paths` entries can be pickles or directories of pickles. |
| `training/data.py` | Builds `PMTIndexDataset` for that config. |
| `config_schema/dataset.py`, `datasets/mixture.py` | A `mixture` component may be a `PMTIndexDatasetConfig`, so datasets with different camera sets (AV2 7, nuScenes 6) mix in one run. |
| `config_schema/models.py`, `model/backbone/decoders.py` | `KelvinDPTDecoderConfig.n_semantic_classes: 5 \| 8`. `KelvinModelConfig.init_full_keep_heads`: fine-tune the released model, keeping its Gaussian head and affine post-processing. |
| `primitives/kelvin_primitive.py` | `KelvinSemanticClass` gains LANE 5, CROSSWALK 6, BOUNDARY 7. With them, ROAD means road surface without paint. |
| `training/system.py` | `_widen_semantic_head`: a released 5-class head is widened to 8. LANE and CROSSWALK start from the ROAD row, BOUNDARY from the OTHERS row, all with bias −2 so the argmax at step 0 is unchanged. |
| `training/losses.py`, `utils/batch.py` | Optional `CameraFrameLabels.semantic_class` (255 = ignore). When set, it is the semantic-loss target instead of the class implied by the ray flags. EGO pixels stay EGO. |
| `configs/training/pmt_av2_smoke.yaml`, `pmt_av2_pilot.yaml` (new) | One-step smoke config and the 2-epoch AV2 pilot. |
| `configs/training/pmt_av2_nusc.yaml`, `pmt_av2_nusc_smoke.yaml` (new) | AV2 + nuScenes fine-tune with 504×280 supervision, and its smoke at the same contract. |
| `configs/training/pmt_geo_av2_nusc_waymo.yaml`, `pmt_geo_waymo_smoke.yaml` (new) | AV2 + nuScenes + Waymo on the geo splits PMT and mapnn use (§3), and a smoke of the Waymo path. |
| `config_schema/dataset.py`, `datasets/pmt_index.py` | `PMTIndexDatasetConfig.semantic_remap` (default empty): class → class, composed onto the index's 8-class `semantic_lut`. `{5: 3, 6: 3, 7: 0}` trains the released 5-class head (2026-11-22). |
| `datasets/pmt_index.py`, `training/losses.py`, `config_schema/train.py` | Index cameras with lens `distortion` (undistorted on load), a separate `depth_grid` and (OpenScene) `seg_grid`; the semantic loss is skipped, not NaN, on a batch with no mask; `loss.background_requires_semantics` (default off). |
| `config_schema/dataset.py`, `datasets/pmt_index.py` | `PMTIndexDatasetConfig.cuboid_tracks` (default off, 2026-11-23): an index's annotated boxes (`tracks`) become cuboid tracks as in `NCoreInstantNuRecDataset._compute_cuboid_tracks`, which turns on the upstream motion supervision (`primitive_velocity`) and box-driven dynamic layers. |
| `configs/training/pmt_openscene_motion.yaml`, `pmt_openscene_motion_smoke.yaml` (new) | OpenScene-only fine-tune on its geo split with the motion head supervised by the boxes, and its smoke. |
| `cli.py`, `model/__init__.py`, `model/system.py`, `datasets/datamodule.py`, `datasets/pmt_index.py`, `config_schema/{dataset,predict}.py`, `predict/{export_ply,export_cameras,render_preview}.py` | `run_inference.py --index-path` (2026-11-24): index logs through the NCore predict path (chunking, merge, PLY / sky / dynamic export, preview, video); `--checkpoint` runs a `.pth` or a fine-tune's Lightning `.ckpt`. Index runs also write `.cameras.npz` and a per-Gaussian `semantic_class` PLY attribute. NCore runs are unchanged. Details below. |
| `tools/index_export/`, `scripts/index/` (new, 2026-11-25) | The index exporter and its batch scripts, moved from PMT (logic unchanged; runs in `sjy-inurec`); the semantic LUT as a table. The index tree is `data/index/`. |
| `tools/viewer.py` (new) | nerfstudio-style viser viewer of an exported scene, rendered on the GPU with the model's 3DGUT renderer. |

**Index pickles.** Step by step from raw downloads: `docs/prepare_data.md`. These are written by
`tools/index_export/export_index.py` (README there) into
`data/index/`, with the batch scripts in `scripts/index/`. The exporter handles AV2, nuScenes, (2026-11-21)
Waymo and (2026-11-23) OpenScene. Until 2026-11-25 it was PMT `RoGS/scripts/inurec_finetune_index.py`, run in
`sjy-rogs`, and the tree was PMT `RoGS/output/inurec_finetune/index` (now a symlink to `data/index`). The move
changed no pickle: all 4646 in use were re-exported with the moved code and compared field by field; they are
identical except OpenScene box rotations, which differ by ≤ 4.4e-16 (numpy 1.26 vs 1.24).

```
cameras: {name: {K, wh, T_cam_ego}}             # OpenCV camera frame
rig: {timestamps_us, T_ego_world}               # ego pose at every image timestamp
frames: {name: [{t_us, img, seg, depth}, ...]}  # every image; seg / depth may be None
semantic_lut: (256,) uint8                      # Mapillary Vistas v2 id -> KelvinSemanticClass, 255 = ignore
# optional per camera (Waymo): distortion {K, D, wh}  -> img files are raw, undistorted onto K / wh on load
#                              depth_grid {K, wh}     -> the depth files' own pinhole grid, resampled onto K / wh
# optional (OpenScene): tracks {id: {label, dims (l, w, h), timestamps_us, T_box_world}}  -> annotated 3D boxes
```

**Labels.**
- **Depth:** LiDAR-aligned MoGe-3 depth made by DVGT's pipeline, copied 2026-11-21 to this repo's
  `data/moge3/align_depth` with its labeling code in `tools/moge_labels/` (README there: provenance, data
  layout). Index pickles reference only `instant-nurec/data/` (symlinks to the datasets and Mask2Former
  outputs, plus that copy); pickles exported earlier were rewritten by `logs/rewrite_index_paths.py`.
  It is stored as log-encoded uint16 camera-z PNGs and converted to ray distance in
  `pmt_index.read_log_depth` / `z_to_ray_distance`.
- **Normals:** derived from that depth in `KelvinInstantNuRec.prepare_context` (`model/kelvin.py`),
  so the `primitive_normal` loss is active.
- **Semantics:** M2F Mapillary-v2 masks from the RoGS label dirs, mapped to the 8 classes. AV2 and
  nuScenes have them. Waymo masks come from PMT `Mask2Former/demo/inference_waymo.py` (in progress
  2026-11-21) into `PMT/data/m2f_infer_waymo/<segment>/seg_<camera>/<t_us>.png`, predicted on the raw
  distorted jpgs; the loader undistorts them nearest-neighbour onto the image grid. Until a segment's
  masks exist (then `scripts/index/export_geo.sh` with `REEXPORT_WAYMO=1`), its frames carry `semantic_class` 255.
- **Coverage:** depth exists only on 2 Hz keyframes. `context_requires_depth` draws context
  frames from those.

AV2 index: 695 logs with MoGe-3 depth, split 491 / 101 / 103 into train / val / test under
`data/index/av2/{train,val,test}` (list: `scripts/index/av2_logs.txt`). Test is held out. These 695
are exactly the AV2 geo-train logs. DVGT added depth for the other 305 logs on 2026-11-20 (same 2 Hz
keyframes), so all 1000 AV2 logs now have it; `export_av2.sh` still exports only the listed 695, because
`pmt_av2_pilot.yaml` and `pmt_av2_nusc.yaml` read these directories whole.

nuScenes index (2026-11-20, `logs/export_nusc_all.sh`): the official v1.0-trainval split, 700 train / 150 val
scenes under `.../index/nusc/{train,val}`. Every keyframe has MoGe-3 depth (DVGT's filter lists keep all
34,149) and every frame, sweeps included, has an M2F mask. 6 cameras at 1600×900, ~40 depth keyframes per
camera per scene.

Waymo index (2026-11-21): `data/waymo_ns` → `/home1/irteam/data/waymo_ns`, converted by DVGT (moved out of DVGT
`public_datasets/` 2026-11-21; nerfstudio layout, 1000 segments, 5 cameras
at 10 Hz) with DVGT's MoGe-3 depth on every 5th frame (2 Hz; 35–40 per camera per segment). Three things
differ from AV2 / nuScenes:
- **Lens distortion** (k2 ≈ −0.3). The images are undistorted on load onto
  `cv2.getOptimalNewCameraMatrix(alpha=0)`, which keeps the full raw frame with no black border.
- **Depth on DVGT's own grid.** DVGT undistorts, resizes and crops around the principal point, then
  resizes back to the raw aspect, so its pixels are anisotropic (side cameras fx 756 vs fy 1364). The
  exporter reproduces that grid with DVGT's own functions and asserts it matches the PNG size; the loader
  resamples depth onto the image grid. The side cameras' principal point sits high (cy ≈ 240 of 886), so
  their depth covers only the upper ~55% of the frame; the lower band, mostly near road, has no depth.
- **Poses.** Ego = the vehicle pose at the frame timestamp (waymo_ns gives all cameras the lidar
  timestamp) times the static extrinsic. waymo_ns also stores a per-image camera pose, which differs from
  this by up to 0.2 m on the segment checked (rolling shutter / trigger time). DVGT aligns its depth with the
  same static-extrinsic convention. Rolling shutter is not modelled.
Checked on one segment: depth overlaid on the undistorted images lines up at pole, tree, car and wall edges.

Geo-split tree (2026-11-21, `scripts/index/export_geo.sh`): `.../index/geo/{av2,nusc,waymo}/{train,val}`, from the
"near" lists, copied to `configs/splits/geo_near/` (README there: PMT's AV2 / nuScenes lists, identical to
mapnn's, and mapnn's Waymo list, commit 8ef6637). AV2 695 / 150; nuScenes 554 / 99 (the lists name
657 / 124, but 103 / 25 of those scenes are from the official test split, which has no MoGe depth); Waymo
700 / 151. AV2 geo-train and nuScenes are symlinks to the pickles above; geo-test is not indexed.

OpenScene index (2026-11-23, `scripts/index/export_openscene.sh`, for `pmt_openscene_motion.yaml`): `data/openscene` →
`/home1/irteam/data/openscene` (NAVSIM log pickles, 2 Hz, 8 cameras at 1920×1080). The unit is a geo-split run
`<log>:<first>-<last>` (mapnn's OpenScene "near" list, copied to `configs/splits/geo_near/openscene`; runs cut
logs, ≥ 70 m between train and val frames), one pickle per run; runs under 8 frames are not exported, leaving
443 train / 307 val runs. 32 runs drop a frame or two (1.0 / 1.5 s gaps); the sampler steps over them.
- **Depth:** DVGT's MoGe-3 (`moge3_vitl`) labels of the 240 "lvpgh" logs (140 Pittsburgh, 100 Las Vegas), on
  every frame and camera, hard-linked into `data/moge3/align_depth/openscene` (no extra disk). DVGT's quality
  filter for them was still running; the index takes every label file. Lens distortion (k1 = −0.36) and the
  depth grid are handled as for Waymo (DVGT `open_scene_loader.py`, num_tokens 2500); the reproduced grid
  equals the intrinsics DVGT stored for each image.
- **Masks:** from PMT `Mask2Former/demo/inference_openscene.py` (being rerun 2026-11-24) into
  `data/m2f/openscene/<log>/seg_<CAM>/<token>.png` (symlink to PMT `data/m2f_infer_openscene`), Vistas v2 ids.
  The exporter reads each camera's mask grid off its first mask: raw size (1920×1080) = predicted on the raw
  image, undistorted on load as for Waymo; 846×476 = predicted on DVGT's MoGe image, so the camera gets
  `seg_grid` (= `depth_grid`) and the loader resamples masks like depth (255 outside DVGT's crop, ~7% of the
  frame). Any other size fails the export. The MoGe-grid case was checked on one run from the first mask pass
  (masks line up with the undistorted images at cars, poles, kerbs and paint). The index picks masks up on
  re-export (`REEXPORT=1 scripts/index/export_openscene.sh`).
- **Boxes:** NAVSIM `anns` (x, y, z, l, w, h, yaw in the lidar frame, which equals the ego frame here, plus
  `track_tokens`), stored per track as box → world poses. Checked on one run: lifted MoGe-3 points fall inside
  the vehicle boxes 1.6–1.9× more often with the true yaw than with yaw + 90°, so boxes, calibration and
  poses agree.

**Inference on index logs (2026-11-24).** `run_inference.py --index-path` takes index pickles (or
directories of them) instead of `--ncore-path` and runs them through the same predict path:
- Each log is cut into sequential chunks that cover it (`AdaptiveSequentialFrameBatchSampler`, as for NCore
  clips). Every image is a context candidate; labels are not read.
- One dataloader batch holds every chunk of one log, so `--merge` always merges the whole log.
- Chunk merge, voxelisation, and the PLY / `.sky.npz` / `.dynamic.npz` export are the NCore ones.
  `--render-preview` and `--render-video` work through a pinhole branch in `render_composited_frame`
  (before this, those renders were F-theta only). `--render-video` renders every frame of the first camera
  in the log; checked on the Waymo val log below (198 FRONT frames, 10 fps, 19.8 s). Since 2026-11-25, when
  `--max-chunks` truncates the log the video stops at the end of the last reconstructed chunk (a warning names
  the frame count and the `--max-chunks` for the whole log); before, it raised and wrote no video.
- Cameras default to all of the index's cameras in its order, which is the fine-tune's order, at any count.
  The released model's 1 / 3 / 5 camera check applies to NCore input only.
- OpenScene boxes become cuboid tracks, so `--motion-source tracks` uses them. The other datasets have no
  boxes, so their dynamic layer is the MOVABLE pixels with learned motion.
- Since 2026-11-25 `--index-path` sets `track_dynamic_any_chunk`: a box that is DYNAMIC (travel > 1.5 m in the
  chunk's window) in any chunk of the log is DYNAMIC in every chunk. Before, a vehicle stopped in one chunk went
  to that chunk's static layer, and the merge's frustum ownership, which is spatial only, dropped it wherever a
  later chunk saw the spot from closer after it had left. A stopped bus (OpenScene
  `2021.09.14.18.43.41_veh-45_00885_00952_0-133`, 27.5–31.0 s) was missing from the video until it drove off. The
  field is off by default, so training configs are unchanged.
- The world is moved to the log's first rig pose; OpenScene's UTM coordinates (~4e6 m) would lose ~0.3 m to
  float32. `.cameras.npz` stores the offset (`world_origin`; dataset world = PLY + `world_origin`) next to
  every context camera's pose and K, for the viewer.
- `--n-frames` (default 18) and `--max-frame-gap-us` (default 750000) set the chunking. The PMT
  fine-tunes trained with 8 and 500000. At OpenScene's 2 Hz, the 500000 gap makes the sampler repeat a frame
  in about half the chunks, as the NCore sampler does on sparse clips.
- `--checkpoint` works with either input. It loads a weights-only `.pth` or a Lightning `.ckpt`, and reads
  the semantic head width (5 / 8) from the weights.

```bash
INSTANT_NUREC_RUN_ID=geo-2576 CUDA_VISIBLE_DEVICES=0 python run_inference.py --model pa-multiview \
  --index-path data/index/geo/waymo/val/10226164909075980558_180_000_200_000.pkl \
  --checkpoint checkpoints/training/pmt-geo-av2-nusc-waymo-2ep/checkpoints/last.ckpt \
  --n-frames 8 --max-frame-gap-us 500000 --max-chunks 12 --output-dir outputs/predict \
  --merge --include-dynamic --render-preview
CUDA_VISIBLE_DEVICES=0 python tools/viewer.py outputs/predict/geo-2576/ply/<seq>/<seq>.ply --port 8080
```

**Viewer.** `tools/viewer.py` serves a viser page on `127.0.0.1:<port>`; from another machine, use
`ssh -L 8080:localhost:8080 <host>`. Like nerfstudio's viewer, each view is rendered here and streamed to the
page: a reduced size while the camera moves, then full size once it rests.
- Renderer: gsplat 3DGUT with calibrated pinhole rays, the call the model's preview uses. Its render of a
  context camera matches the `--render-preview` PNG at 50.4 dB PSNR (AV2 val log, step 2576).
- Render modes: rgb (over the sky cubemap, through the chosen camera's ISP affine), rgb without sky, semantic
  class, depth and opacity.
- Controls: per-class visibility, the dynamic layer at a time slider (with play), and context-camera frustums.
  Clicking a frustum looks through that frame, setting its pose, FoV, time and ISP camera.
- Ego trajectory: `.cameras.npz` carries the rig poses the chunks were built on (`ego_T_world`,
  `ego_timestamps_us`, PLY frame; exports from 2026-11-24 on). The viewer draws them as a line, with the part
  up to the time slider highlighted. The ego at that time is a box + axes; the box is a nominal 4.6 × 1.9 ×
  1.6 m car, 1.4 m ahead of the ego origin (rear axle on the ground). Older files fall back to the first
  camera's path.
- Semantic colours need the `semantic_class` attribute, which `--index-path` runs write. NCore PLYs carry only
  road / sky masks.
- Cost per view (2026-11-24, OpenScene scene, 2.0 M static + 8.7 M dynamic Gaussians, shared H200): render,
  readback and JPEG take 10–20 ms. A frame is 22 kB moving (480 px) and 95 kB refined (1280×720). The
  bottleneck was the depth image viser composites frustums with: a lossless PNG of 2 MB per 1280×720
  frame (~20× the JPEG), 40 ms to encode, then the transfer and the browser's decode. It is now off by
  default ("frustums behind geometry"), and when on it goes with refined frames only. The Gaussian set
  (class filter + dynamic layer at the current time) is cached between frames. No distance or occlusion
  culling: gsplat already skips off-screen Gaussians and stops each pixel once it is opaque, so culling would
  lose far geometry for little gain.
- `viser` 1.1.1 was pip-installed into `sjy-inurec` on 2026-11-24. It is not in `pyproject.toml` / `uv.lock`.
- The viewer sets the gsplat JIT toolchain itself: the repo build cache, the env's `CUDA_HOME`, `bin/` on
  PATH, and the env's `CC` / `CXX` / nvcc `-ccbin`, as `conda activate` sets them. With `CXX` unset, torch
  writes `cxx = c++` and nvcc falls back to a `gcc` this machine lacks. Ninja then sees changed compile
  commands, so it rebuilds and fails (`gcc: No such file or directory`).

**ffmpeg (2026-11-24).** `--render-video` needs ffmpeg with libx264 on PATH. `conda install -c conda-forge
ffmpeg` fails in `sjy-inurec`: the `gdk-pixbuf` post-link script picks up the env's cross-compiler activation
and looks for `ld.gold` in the base env. The rollback also deletes `conda-meta/history`, which was recreated
(with a note) after each attempt. The env's ffmpeg is instead the static build from pip `imageio-ffmpeg`
0.8.1 (ffmpeg 9.0.2, libx264), linked as `bin/ffmpeg`.

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
  is in where road points land, not in the semantics. Likely cause (2026-11-22, §3 "Class BEV and boundary
  in the road test"): the GT includes LiDAR boundary-labelled points. Against GT road surface only, step
  9000 beats step 1000 on all 3 logs.

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

  DVGT's filtered annotation lists: `data/moge3/meta/moge_pred_filter_parquet` (copied from DVGT for AV2,
  nuScenes and Waymo).
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
- **Result (run `pmt-av2-nusc-2ep`, 2026-11-20 16:59 → 11-21 23:14, GPUs 0–3, instant-nurec `24f1a8b`).**
  Both epochs completed (2,454 steps, ~45 s/step, about 19.6k samples). The final step is saved
  (`checkpoints/training/pmt-av2-nusc-2ep/checkpoints/epoch=epoch=02-step=step=2454.ckpt`). `val/psnr`:
  20.85 after epoch 1, 21.56 after epoch 2. Evals on step 2454, with the step-9000 protocol and the same
  scripts (unchanged since; PMT `9e6eee3`, mapnn `ede8699`; sha256 in
  `RoGS/output/inurec_road/av2nusc_2ep_run.log`). Single seed, three road-test logs. **Confounded vs the
  pilot:** data mix, batch 8 vs 1, and steps all differ.

  | road test | dz | median \|dz\| | IoU | IoU ≤ 15 m |
  | --- | --- | --- | --- | --- |
  | 7fab: step 9000 → AV2+nuS | −0.018 → −0.026 | 0.025 → 0.031 | 0.584 → 0.578 | 0.505 → 0.500 |
  | 02678d04 | +0.008 → −0.005 | 0.027 → 0.031 | 0.488 → 0.486 | 0.504 → 0.496 |
  | 0b5142c1 | +0.029 → −0.006 | 0.039 → 0.032 | 0.652 → 0.670 | 0.506 → 0.517 |

  | next-frame render, static (PSNR / SSIM / LPIPS) | step 9000 | AV2+nuS 2454 |
  | --- | --- | --- |
  | rendered next frame | 19.43 / 0.624 / 0.357 | 18.75 / 0.582 / 0.308 |
  | copy previous frame ("hold") | 17.14 / 0.457 / 0.335 | 17.13 / 0.456 / 0.335 |
  | re-render newest input | 22.41 / 0.725 / 0.286 | 22.19 / 0.736 / 0.209 |

  - **Road height:** stays at the MoGe-label level.
  - **Road IoU:** within about a point of step 9000, and still below step 1000. The open IoU drop is
    unchanged.
  - **Render:** the rendered next frame loses 0.7 dB PSNR, but its LPIPS now beats hold (0.308 vs 0.335);
    at step 9000 LPIPS still favoured hold.
  - **Semantic share on 7fab:** CROSSWALK is argmax on 0.6% of pixels, against 0.1% at step 9000.

  Outputs: `RoGS/output/inurec_road/av2nusc_2ep/{vis,render}` and `labelcheck/<log>/av2nusc_2ep`.
- **Class BEV and boundary in the road test (2026-11-22).** Scripts `RoGS/output/inurec_road/cls_bev_run.sh` and
  `cls_bev_lidar_run.sh`.
  - **What the scripts save now.** `inurec_road_run.py` also keeps BOUNDARY Gaussians (8-class head), with
    a per-Gaussian `cls`.
  - **Visualisations.** The RGB BEV shows road surface plus boundary. `vis --rogs-dir <RoGS scene dir>`
    writes `road_bev_cls_*.png`:
    - the Gaussians' road / lane / crosswalk / boundary (opacity-weighted vote, RoGS colours) on RoGS's
      own BEV grid (0.05 m, trajectory ±7 m);
    - below it, that RoGS run's `images/final/bev_label.png` and its LiDAR road-point label render
      `road_point/bev_label.png`;
    - per-class IoU vs the LiDAR labels in `road_bev_cls_stats.json`. The script checks that both
      grids match.
  - **Frames.** The road-test world is not RoGS's AV2 world: it is anchored at the first LiDAR sweep,
    RoGS at the first `ring_rear_left` frame, 0.03–0.55 m apart on these logs. The class BEV corrects for
    this.
  - **Established metrics unchanged.** `all` and `range<=15m` (road surface vs every GT point) are
    bit-identical to before for zero-shot, MoGe, step 1000, step 9000 and AV2+nuS on the 3 logs.
  - **Likely cause of the step-1000 → 9000 IoU drop.** The LiDAR road GT includes boundary-labelled
    points (29% / 45% / 29% of points on 7fab / 02678d04 / 0b5142c1), and the established IoU scores
    road surface against all of them. Step 1000 labels more curb and sidewalk as road. With that
    mismatch removed, step 9000 is ahead on every log. Single seed, 3 logs.

    | IoU (7fab / 02678d04 / 0b5142c1) | established | surface vs GT surface | surface+boundary vs all GT |
    | --- | --- | --- | --- |
    | step 1000 | 0.608 / 0.503 / 0.689 | 0.739 / 0.579 / 0.794 | 0.756 / 0.623 / 0.676 |
    | step 9000 | 0.584 / 0.488 / 0.652 | 0.786 / 0.605 / 0.833 | 0.780 / 0.637 / 0.688 |
    | AV2+nuS 2454 | 0.578 / 0.486 / 0.670 | 0.788 / 0.619 / 0.824 | 0.779 / 0.633 / 0.672 |
    | MoGe-3 labels | – / 0.500 / 0.665 | – / 0.620 / 0.839 | – |

  - **Class IoU vs the LiDAR road-point labels on RoGS's grid.** Road / lane / crosswalk / boundary, on
    7fab:

    | 7fab | road | lane | crosswalk | boundary |
    | --- | --- | --- | --- | --- |
    | step 1000 | 0.63 | 0.00 | 0.00 | 0.52 |
    | step 9000 | 0.67 | 0.47 | 0.00 | 0.58 |
    | AV2+nuS | 0.67 | 0.32 | 0.09 | 0.57 |
    | RoGS `single_trip_av2` | 0.69 | 0.51 | 0.69 | 0.33 |

    Comparing the two:
    - RoGS is ahead on lane and crosswalk.
    - The Gaussians are ahead on boundary.
    - Road is within 0.03.

    Two caveats:
    - The Gaussians are rasterised as point centres, which leaves far-range speckle, while RoGS is
      splatted. This counts against the Gaussians.
    - RoGS is a per-scene optimisation over the whole log.
  - **More logs (2026-11-22).**
    - Runs: `vislogs_run.sh` covers the logs of the mapnn views / BEV vis samples: 0031 = 366ceb72,
      0101 = a059b6b9, 0171 = ded5ef6e. `heldout_run.sh` adds 5 logs. All use the same protocol and
      scripts (sha256 in the logs), with the same four models. `road_summary.py` builds the table.
    - The 5 added logs: 05fa5048, 47286726, 65387aee, 6c932547, e42aa296. They were drawn with a fixed
      seed from the 25 AV2 val ∩ geo-val logs that have LiDAR road GT and a RoGS run, so they are held out
      from every fine-tune, the geo run included.
    - 366ceb72 and a059b6b9 are in AV2 train; the fine-tunes trained on them, so they are not counted.
    - Held-out total: 9 logs (7fab, 02678d04, 0b5142c1, ded5ef6e and the 5 added).

    | 9 held-out logs, median | median \|dz\| | IoU established | IoU surface vs GT surface |
    |---|---|---|---|
    | zero-shot | 0.634 | 0.548 | 0.637 |
    | pilot step 1000 | 0.060 | 0.608 | 0.684 |
    | pilot step 9000 | 0.031 | 0.586 | 0.730 |
    | AV2+nuS step 2454 | 0.032 | 0.586 | 0.734 |

    - **IoU-drop explanation holds.** Step 9000 and AV2+nuS beat step 1000 on surface vs GT surface on
      9/9 logs, and lose to it on the established IoU on 8/9. Single seed per model.
    - **Crosswalk.** The fine-tunes score ≤ 0.16 class IoU vs LiDAR on every log. RoGS scores 0.33–0.69 on
      the 5 held-out logs where both the LiDAR labels and RoGS have crosswalk (7fab, 05fa5048, 65387aee,
      6c932547, e42aa296).
    - **Lane.** Step 9000 scores 0.14–0.47 and AV2+nuS 0.07–0.34, against RoGS's 0.24–0.59. Boundary is at
      or above RoGS on 7 of the 9 logs; it is below on 0b5142c1 and ded5ef6e.

**AV2 + nuScenes + Waymo on geo splits (2026-11-21, `pmt_geo_av2_nusc_waymo.yaml`).** The user chose to
train on Waymo, on the geographically disjoint splits PMT and mapnn train and evaluate on, so numbers are
comparable with theirs. Same recipe as `pmt_av2_nusc.yaml` except: geo splits; Waymo added; 695 AV2 × 5 +
554 nuScenes × 6 + 700 Waymo × 5 = 10,299 samples per epoch (about a third each), 1,288 steps / epoch on
4 GPUs × batch 2; val a fixed draw of 25 geo-val logs per dataset; `background_requires_semantics: true`.
- **Waymo masks (pending).** The current Waymo pickles have none, so a Waymo batch skips the semantic loss
  (it used to return NaN on an all-ignore target) and trains no sky cubemap. Without
  `background_requires_semantics` the background loss would treat Waymo sky as foreground (target opacity
  1) and push the model to paint the sky. Once `inference_waymo.py` has run, re-export the Waymo pickles
  (`REEXPORT_WAYMO=1 scripts/index/export_geo.sh`) and Waymo trains semantics and sky like AV2; the flag then only
  matters for frames without a mask. A run started before the re-export keeps the mask-less pickles.
- **Waymo is no longer a held-out rig.** HRDX still is. Waymo geo-test (149 segments) stays held out by
  log, and mapnn's zero-shot Waymo number is now in-domain for this model.
- **Road test logs:** 7fab is AV2 geo-val; 02678d04 and 0b5142c1 are geo-train, so on this model the
  road test scores two training logs.
- Verified: unit tests (`tests/test_pmt_mixed_labels.py`), all 2,349 index pickles load, and one item per
  dataset through the full config on CPU (context 56 / 48 / 40 images; depth valid 95% / 89% / 75%).
  Smoke (`pmt_geo_waymo_smoke.yaml`, one Waymo geo-train segment, batch 1, GPU 2 shared with the
  AV2 + nuScenes run): 2 steps + val, every loss finite, `last.ckpt` written, val PSNR 15.70; skipped as
  designed: background, primitive_semantics, primitive_velocity. `primitive_distance` is 11.2 / 10.4 m at
  steps 0 / 1 (AV2 + nuScenes smoke: total loss 6.7–8.0), in line with the released model's zero-shot
  Waymo range ratio of 0.47 (§3). Peak memory and s/step not measured (GPU shared). The full 3-dataset
  config on 4 GPUs has not been run.
- **5-class head (decided 2026-11-22).** The 8-class runs barely predict CROSSWALK (class IoU vs LiDAR
  0.00–0.09 on 7fab, against RoGS's 0.69), so this run keeps the released 5-class head (OTHERS, EGO, SKY,
  ROAD, MOVABLE) unchanged, with no widening. The index pickles stay 8-class; `semantic_remap:
  {5: 3, 6: 3, 7: 0}` folds LANE and CROSSWALK into ROAD (paint is road surface) and BOUNDARY (curb,
  sidewalk, median, traffic island, …) into OTHERS. ROAD_SEMANTIC flags are unchanged by this, because
  BOUNDARY was never road surface. Map classes are to be assigned after the fact, either by projecting the
  Gaussians into the M2F masks and voting, or by RoGS optimisation. Neither is built yet. The road test on
  this model saves ROAD only (`inurec_road_run.py` keeps BOUNDARY for 8 classes only), so compare it on the
  road-surface metrics.
  - Verified on CPU: on 2 AV2 and 2 nuScenes geo-val items, the remapped labels contain only classes 0–4.
    They equal the 8-class labels with the fold applied, pixel for pixel (ROAD share = old ROAD + LANE +
    CROSSWALK share).
  - Smoke (`pmt_av2_nusc_smoke.yaml` with the 5-class head and remap, batch 1, GPU 1 shared): released
    head loaded with 0 unexpected keys. 4 steps + val, every loss finite (total 6.7–8.0, the 8-class smoke's
    range), `primitive_semantics` 0.17–0.27. Checkpoints at steps 2 and 4 and `last.ckpt` written; val
    PSNR 14.48. Only `primitive_velocity` skipped. Smoke only, no eval.
  - Comparison: against `pmt-av2-nusc-2ep` this run changes the data and splits as well as the head, so
    the head's effect cannot be read from it alone.
- **Licence.** Waymo Open Dataset terms now apply to these checkpoints; they are not yet recorded in
  mapnn `docs/data-provenance.md` ("Other datasets").

**Risks.**
- **Compute.** The single-dataset pilot took ~30 h on one shared GPU, and the box's GPUs are
  mostly occupied.
- **Label quality varies by dataset.** Use DVGT's filtered lists.
- **The semantic head sees only AV2 and nuScenes.** It may drift on other rigs, Waymo included.

**Open decisions.**
- Held-out rigs: decided 2026-11-21 for `pmt_geo_av2_nusc_waymo.yaml`: Waymo in training (geo split),
  HRDX held out.
- Keep the 8-class semantic head: decided 2026-11-22. Train the released 5-class head, and assign map
  classes after the fact.
- Initialise from step 9000 or from the released weights.
- GPU budget.
