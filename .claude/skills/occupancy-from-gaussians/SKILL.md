---
name: occupancy-from-gaussians
description: Derive or train 3D semantic occupancy from Instant NuRec's feed-forward Gaussians, and evaluate it on Occ3D-nuScenes or SurroundOcc. Use when asked about occupancy, voxel IoU/mIoU, Gaussian-to-voxel aggregation, free-space estimation, or reusing the gsdiff / gsformer (GaussianFormer) occupancy code and data on this machine. Nothing here is built yet — this skill is the plan and the pitfalls. Related — task-head-integrator agent (implementing), experiment-scientist agent (protocol), inurec-data (frames and labels).
---

# Occupancy from Gaussians

**Status: not implemented.** This repo has no occupancy head, no occupancy GT loader and no
occupancy metric. `KelvinStaticLayer.voxelize` is a KL-optimal *Gaussian merge* for export. It is not
occupancy. Say so in every plan.

## What already exists on this machine

| Thing | Where | Use |
|---|---|---|
| Occ3D-nuScenes GT | `../gsdiff/data/occ3d/gts/<scene>/<sample_token>/labels.npz` (also `../gsformer/data/occ3d`) | `semantics`, `mask_lidar`, `mask_camera`; ego frame |
| SurroundOcc GT | `../gsdiff/data/surroundocc/samples` | denser GT, LiDAR frame |
| Occ3D protocol | `../gsdiff/config/prob/nuscenes_gspixel_finetune_diff.py` | `pc_range [-40,-40,-1,40,40,5.4]`, voxel 0.4 m, `use_ego=True`, empty label 17 |
| mIoU with camera mask | `../gsdiff/tools/misc/metric_util.py::MeanIoU` | `use_mask` → `mask_camera` |
| Gaussian-to-voxel aggregation (CUDA) | `../gsdiff/model/head/{localagg,localagg_prob,…}`, `GaussianHead` in `model/head/gaussian_head.py` | GaussianFormer-style semantic Gaussian splatting into voxels |
| Upstream GaussianFormer / GaussianFormer-2 | `../gsformer/` | reference implementation |
| Pixel-aligned Gaussian lifter | `../gsdiff` `GaussianLifterPx` | the closest existing analogue to Instant NuRec's per-pixel Gaussians |

Env for those repos: `gsdiff`. Read `../gsdiff/CLAUDE.md` before using its code. Its anchors are
stored in sigmoid / logit space, unlike Instant NuRec's metric world-space Gaussians.

## Mismatches to resolve before any number

1. **Frame.**
   - Instant NuRec's world frame is the first camera's first context frame, scaled by
     `scene_rescale`; exported primitives are in metres.
   - Occ3D is the ego frame at the sample's timestamp.
   - Transform with the rig pose at that timestamp. Check it by overlaying Occ3D occupied voxels
     on the Gaussians for one sample before computing any metric.
2. **Time.** Context covers T frames, but occupancy is per timestamp. Static layer: use as is.
   Dynamic layers: `KelvinDynamicLayer.interpolate(t_us)` at the sample time. Without
   `include_dynamic`, movers are missing. Say which.
3. **Taxonomy.**
   - The semantic head has 8 classes: OTHERS / EGO / SKY / ROAD / MOVABLE / LANE / CROSSWALK /
     BOUNDARY.
   - Occ3D-nuScenes has 17 classes plus free.
   - Only a coarse mapping exists. ROAD+LANE+CROSSWALK go to driveable surface, and MOVABLE to the
     union of vehicle / pedestrian / cyclist classes. OTHERS covers sidewalk, terrain, vegetation,
     manmade and barrier.
   - Report **binary geometry IoU first**, then semantic mIoU only over classes the mapping can
     express. `n_semantic_classes` is `Literal[5, 8]`; widening it again is a model change with its
     own run (`task-head-integrator`).
4. **Visible surfaces only.**
   - Pixel-aligned Gaussians sit on observed surfaces. The space behind them is unknown, not free.
   - Evaluate with `mask_camera`.
   - Do not fill occluded volume. Free space comes from ray traversal: voxels between the camera and
     the predicted distance are free.
5. **Dataset.**
   - Occ3D-nuScenes needs nuScenes inputs in the PMT index format. The exporter supports nuScenes,
     but no nuScenes index has been trained on or validated here.
   - The released / AV2 checkpoints are out of domain on nuScenes (zero-shot 10–20 m error 4.20 m,
     `PMT_FINETUNE.md` §3). Expect geometry, not the readout, to dominate.

## Recommended order

Cheapest first. Each step has its own pass criterion.

1. **Frame check (no metric).**
   - Export Gaussians for ~5 nuScenes val samples.
   - Transform them to the Occ3D ego frame and overlay them on `labels.npz` occupied voxels.
   - Pass: road Gaussians lie in the driveable-surface voxels to within one voxel of height.
2. **Readout baseline (no training).**
   - Voxelise at 0.4 m: a voxel is occupied if the density-weighted Gaussian mass in it passes a
     threshold, and its class is the argmax of the summed per-class probability.
   - Free = voxels crossed by camera rays before the predicted distance.
   - Score binary IoU, and mIoU over the mappable classes, with `mask_camera`, on a fixed sample
     list.
   - This is the baseline every later step is measured against (`research-goals.md`: readout
     before new head).
3. **Geometry control.** The same readout using LiDAR-aligned MoGe-3 depth lifted with calibration
   instead of predicted distance. If that control matches the model, the model adds nothing beyond
   depth.
4. **Trained occupancy.** Only after steps 2–3 show where the gap is. Options:
   - a GaussianFormer-style aggregation loss (`localagg_prob`) on the existing Gaussians;
   - a light occupancy head reading `encoded_latent`.

   Either one is default-off, with the released and AV2 checkpoints still loading strictly.
   `scene-researcher` should ground the design first.

## Rules

- **Never compare against published Occ3D numbers as a baseline.** Use the step-2 readout on the
  same sample list. Published numbers are context only.
- State the sample count, the voxel size, the range, whether `mask_camera` was used, and whether
  dynamic layers were included.
- Any change to the readout threshold or the free-space rule is a protocol change, and it re-runs
  the baseline.
- Licensing: Occ3D and SurroundOcc labels inherit nuScenes terms (`.claude/rules/data-licensing.md`).
