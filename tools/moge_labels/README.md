# MoGe-3 depth labels (copied from DVGT)

LiDAR-aligned MoGe-3 depth for AV2, nuScenes and Waymo: the `metric_distance` labels `PMTIndexDataset`
reads. The labels in `data/moge3/` were **copied** from DVGT on 2026-11-21, not regenerated. This directory
holds the code that made them, so a future relabel can run from this repo.

## Provenance

- Code: DVGT `tools/data_process/` at commit `51cf3f6`, **working tree as run** (several files were modified
  and uncommitted in DVGT then). Apache-2.0 (DVGT `LICENSE`). Copied: `main.py`; the `argoverse`, `nuscene`
  and `waymo` loaders with `base_dataset` and `factory`; `utils/{image,io,alignment,log,tools,transformer_np}`;
  `preprocess/{argoverse,waymo}.py`. `utils/geometry.py` holds `closed_form_inverse_se3` from DVGT
  `dvgt/utils/geometry.py`.
- Changes: import paths, the loaders' default data paths (`data/...`), and `main.py` inserting the repo root
  first on `sys.path`. Logic is unchanged. MoGe comes from the upstream `moge` package
  (`microsoft/MoGe@74fbce0`, MoGe-3); DVGT's vendored `align_depth_affine` is identical to it.
- Run scripts: `run_waymo.sh`, `run_av2.sh`, `run_nuscenes.sh` are DVGT's
  `output/{waymo_moge3,av2_moge3_rest,moge3_relabel}/run.sh` with paths changed. AV2 preprocessing uses
  `sjy-rogs` (has the av2 API) instead of DVGT's `sjy-mapnn`.
- Model: `checkpoints/moge-3-vitl/model.pt`, copied from DVGT `ckpt/moge-3-vitl/model.pt` (Hugging Face
  download, sha256 `9b41b7b9…fc1fb7925`).

## What was run (in DVGT, 2026-11)

| dataset | split(s) | logs | inputs |
|---|---|---|---|
| AV2 | `geo_near_train`, `geo_rest` | 695 + 305 = 1000 | `preprocess/argoverse.py` from `/home1/irteam/data/av2` |
| nuScenes | `train`, `val` | 700 + 150 (trainval only) | `data/nuscenes`, `data/nuscenes_occ_world_pkl` |
| Waymo | `train_2hz`, `val_2hz` | 798 + 202 = 1000 | `preprocess/waymo.py` from `/home1/irteam/data/waymo` tfrecords into `waymo_ns` |

All with `--moge_version=3 --batch_size=2 --gen_meta --gen_image --gen_proj_depth --gen_pred_depth
--gen_align_depth`, every 2 Hz keyframe. Where a frame has fewer than 2 LiDAR points the "aligned" depth falls
back to the empty LiDAR projection, so it stores no valid depth.

## Data layout (`data/`, gitignored)

| path | what |
|---|---|
| `data/moge3/align_depth/{argoverse,nuscene,waymo}/` | the labels (copy) |
| `data/moge3/meta/*.parquet` | per-image intrinsics / extrinsics / poses written with them; `moge_pred_filter_parquet/` = DVGT's quality-filtered lists |
| `data/av2`, `data/nuscenes` | symlinks to `/home1/irteam/data/{av2,nuscenes}` |
| `data/waymo_ns` | symlink to `/home1/irteam/data/waymo_ns`, the only copy of the converted Waymo. Moved there from DVGT `public_datasets/waymo_ns` on 2026-11-21; DVGT keeps a symlink. `run_waymo.sh` preprocesses into it. |
| `data/nuscenes_occ_world_pkl/` | copy of DVGT's OccWorld nuScenes infos (the nuScenes loader reads them) |
| `data/m2f/{av2,nuscenes,waymo}` | symlinks to the PMT Mask2Former outputs |

Not copied: DVGT's `image/`, `proj_depth/`, `pred_depth/` label kinds and `argoverse2_process/`. A relabel
regenerates them.

## Running

From the repo root, in the MoGe-3 env (`/home1/irteam/workspace/venv-dvgt-moge3`), with a GPU:

```bash
GPU=3 bash tools/moge_labels/run_nuscenes.sh
LOG_LIST=<av2 log ids> SPLIT=<name> GPU=3 bash tools/moge_labels/run_av2.sh
GPU=3 bash tools/moge_labels/run_waymo.sh
```

Only the imports were checked here (the loaders register in that env). No relabel has been run from this repo.
A relabel overwrites `data/moge3`; treat it as a label change (`.claude/rules/experiments.md`).
