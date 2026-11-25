---
name: inurec-data
description: Data contracts for Instant NuRec training — NCore V4 sequences with aux sidecars, and the PseudoMapTrainer per-log index pickles read by PMTIndexDataset. Use when preparing or debugging a dataset, adding a new dataset or sensor rig, checking depth/semantic/normal labels, verifying frame conventions, or when a loss is skipped for missing labels. Related — multi-rig-training (mixing datasets), data-pipeline-engineer agent (implementing an exporter), inurec-config (dataset config fields).
---

# Instant NuRec Data

## Two input paths

| Path | Loader | Source of truth |
|---|---|---|
| NCore V4 sequence `.json` (+ `<shard>.aux.{semantic_segmentation,depth,egomask}.zarr`) | `datasets/instantnurec_ncore.py::NCoreInstantNuRecDataset`, `datasets/mixture.py` | `docs/TRAINING.md` "NCore V4 input contract" |
| PMT per-log index pickle | `datasets/pmt_index.py::PMTIndexDataset` | `docs/PMT_FINETUNE.md` §1 and the exporter docstring |

The plan is to keep **one loader for non-NCore data** (`PMTIndexDataset`). A new source gets an
exporter branch in PMT, not a new dataset class here.

## The PMT index

The pickles are written by `tools/index_export/export_index.py` (batch: `scripts/index/export_*.sh`) in
`sjy-inurec`, into `data/index/`. Before 2026-11-25 this was PMT `RoGS/scripts/inurec_finetune_index.py` in
`sjy-rogs`; the old index path is a symlink to `data/index`. One pickle per log:

```
cameras: {name: {K, wh, T_cam_ego}}             # OpenCV camera frame
rig: {timestamps_us, T_ego_world}               # ego pose at every image timestamp
frames: {name: [{t_us, img, seg, depth}, ...]}  # every image; seg / depth may be None
semantic_lut: (256,) uint8                      # Mapillary Vistas v2 id -> KelvinSemanticClass, 255 = ignore
```

Conventions that `PMTIndexDataset` relies on. Check all of them for a new source:
- **Frames.** The rig is the ego and the world is the first camera's first context frame. The first
  entry of `camera_ids` is the reference camera.
- **Cameras.** Global-shutter pinhole only (`pinhole_parameters`). Distorted or fisheye sources must
  be undistorted in the exporter, or extend the loader deliberately.
- **Depth.** Log-encoded uint16 **camera-z** PNG. `read_log_depth` decodes it and `z_to_ray_distance`
  converts it to the **ray distance** the losses expect. Getting this wrong biases scale by the
  per-pixel cosine and does not crash.
- **Depth coverage.** Only 2 Hz keyframes have depth. With `context_requires_depth`, context frames
  are drawn from those only. With `include_context_frames: true`, the render `distance` loss sees
  depth.
- **Semantics.** The M2F Mapillary-v2 mask is mapped through `semantic_lut` into the 8 classes, and
  255 is ignored by the CE loss.
- **No cuboid tracks**, so `primitive_velocity` is skipped (allowed).
- **Normals** are derived from depth in `KelvinInstantNuRec.prepare_context`.

AV2 today: 695 logs with MoGe-3 depth, split 491 / 101 / 103 into train / val / test under
`data/index/av2/{train,val,test}` (`scripts/index/av2_logs.txt`). **The test split is
held out.**

## Labels available per dataset

| Dataset | Aligned MoGe-3 depth | M2F semantics | Status |
|---|---|---|---|
| AV2 | 695 logs | yes | indexed, trained |
| nuScenes | keyframes | yes | exporter supports it, not trained |
| DDAD | 150 scenes | no | next to export (`research-goals.md` roadmap) |
| KITTI | 74.7k frames in filtered list | no | after DDAD, if labels are complete |
| OpenScene | 9 entries, completeness unchecked | no | unverified |
| Waymo | `data/moge3/align_depth/waymo` (2 Hz, all 1000 segments) | Mask2Former in progress (`data/m2f/waymo`) | trains on its geo split since 2026-11-21 |
| HRDX | none here | — | held-out rig, scored by mapnn's zero-shot diag |

Depth labels live in `data/moge3/align_depth` (copied from DVGT 2026-11-21; code in `tools/moge_labels/`).
Use DVGT's filtered lists in `data/moge3/meta/moge_pred_filter_parquet`, because label quality
varies by dataset.

## Image preparation must match `CameraSubsampler`

`datasets/instantnurec_base.py::CameraSubsampler` crops to the target aspect, then resizes, and the
intrinsics follow. Every external script that feeds the model must do the same. The 2026-11-19 bug,
where swapped crop branches meant no crop and an anisotropic squash, moved the depth scale with
no error. The squash factor ranged from 1.01× (nuScenes) to 2.38× (AV2 portrait front).

## Inspecting before training

Load one item and look at it before a long run:

```python
from pathlib import Path
from instant_nurec.training.run import load_training_config
from instant_nurec.datasets.pmt_index import PMTIndexDataset
cfg = load_training_config(Path("configs/training/pmt_av2_smoke.yaml"))
ds = PMTIndexDataset(cfg.dataset.train, global_seed=cfg.seed)
item = ds[0]   # inspect: context cameras/frames, label shapes, depth range (m), semantic histogram incl. 255
```

Check:
- depth in plausible metres (ray distance > camera z off-axis);
- a semantic histogram with no class that never appears;
- context timestamps 0.5 s apart;
- camera count and resolution as configured.

## Failure modes

| Symptom | Likely cause |
|---|---|
| `primitive_semantics` skipped or all-ignore | `semantic_segmentation: false`, `seg` None in the index, or a LUT mapping everything to 255 |
| Depth scale off by a constant | z vs ray distance, wrong K after crop, or crop mismatch in an external eval |
| Very few samples, or retries in the log | `context_requires_depth` with too few keyframes for T × gap; corrupt image paths |
| Train-val leakage | split by frame instead of by log; check that the index directories are disjoint |
