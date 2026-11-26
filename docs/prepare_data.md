# Preparing non-NCore training data (AV2 / nuScenes / Waymo / OpenScene)

How raw driving logs become the index pickles that `PMTIndexDataset` (`instant_nurec/datasets/pmt_index.py`)
trains on. NCore V4 clips need none of this; they are read directly (`docs/TRAINING.md`).

```
raw dataset  ──►  data/<dataset>                      (step 1: download, symlink)
                     │
                     ├─► data/moge3/align_depth/...   (step 2: MoGe-3 depth labels; required)
                     ├─► data/m2f/<dataset>/...       (step 3: Mask2Former masks; optional, not built here)
                     ▼
              scripts/index/export_*.sh  ──►  data/index/<dataset>/...pkl   (step 4)
                     ▼
              configs/training/pmt_*.yaml  (index_paths)
```

An index pickle holds only calibration, ego poses, file paths and (OpenScene) boxes. Images and labels are read
when a sample is drawn, so every file a pickle names must stay where it was at export time. The pickles store
absolute paths under this repo's `data/`, so keep the dataset symlinks stable rather than moving the data.

Everything under `data/` is gitignored.

## Layout of `data/`

| Path | What | Read by |
|---|---|---|
| `data/av2` | Argoverse 2 Sensor Dataset: `{train,val,test}/<log>/{sensors/cameras,calibration,city_SE3_egovehicle.feather}` | exporter, MoGe-3 labeling |
| `data/nuscenes` | nuScenes v1.0-trainval: `v1.0-trainval/`, `samples/`, `sweeps/` | exporter, MoGe-3 labeling |
| `data/nuscenes_occ_world_pkl` | OccWorld nuScenes infos (copied from DVGT) | MoGe-3 labeling only |
| `data/waymo_ns` | Waymo converted to a nerfstudio layout: `<segment>/{transforms.json,images/<CAM>/<t_us>.jpg}` | exporter |
| `data/openscene` | OpenScene / NAVSIM: `navsim_logs/trainval/<log>.pkl`, `sensor_blobs/all_sensor_blobs/<log>/<CAM>/<token>.jpg` | exporter |
| `data/moge3/align_depth/{argoverse,nuscene,waymo,openscene}` | MoGe-3 depth labels (step 2) | exporter, dataloader |
| `data/m2f/{av2,nuscenes,waymo,openscene}` | Mask2Former masks (step 3, optional) | exporter, dataloader |
| `data/index/` | the index tree (step 4) | training configs |

## Step 1: raw data and symlinks

Each dataset needs registration and has its own terms. Accept them yourself; do not use mirrors
(`.claude/rules/data-licensing.md`). Download to a data disk, then symlink it into `data/` under the names
above. On this machine the downloads live in `/home1/irteam/data/`:

```bash
cd /home1/irteam/workspace/instant-nurec
mkdir -p data
ln -sfn /home1/irteam/data/av2       data/av2
ln -sfn /home1/irteam/data/nuscenes  data/nuscenes
ln -sfn /home1/irteam/data/openscene data/openscene
ln -sfn /home1/irteam/data/waymo_ns  data/waymo_ns     # made in step 2 from the Waymo tfrecords
```

| Dataset | Get | What the exporter needs |
|---|---|---|
| Argoverse 2 Sensor | <https://www.argoverse.org/av2.html> (download instructions in its user guide) | the `train` / `val` / `test` log directories, cameras + calibration + poses; LiDAR only for depth labeling |
| nuScenes | <https://www.nuscenes.org/nuscenes#download> (registration) | v1.0-trainval metadata, `samples/` and `sweeps/` camera blobs (the index lists sweeps too) |
| Waymo Open Dataset (perception) | <https://waymo.com/open/download> (registration) | the `training` / `validation` tfrecords (here `/home1/irteam/data/waymo/{training,validation}`), converted to `data/waymo_ns` in step 2 |
| OpenScene (NAVSIM) | <https://github.com/OpenDriveLab/OpenScene> | `navsim_logs/trainval/*.pkl` and the trainval camera sensor blobs; `sensor_blobs/all_sensor_blobs/` is a directory of per-log symlinks into the downloaded parts |

### Optional: OpenScene at 10 Hz (nuPlan sweeps)

OpenScene keeps nuPlan's frames at 2 Hz; nuPlan recorded its 8 cameras at 10 Hz. The in-between frames are
meant as **supervision only** (context stays on the labelled 2 Hz frames), giving the render losses views the
model did not see as input. They come from nuPlan v1.1 itself, which needs a nuplan.org account and acceptance
of its terms:

```bash
# on the download machine; standard-library Python, copy the two files over
python3 scripts/data/fetch_openscene_sweeps.py --logs scripts/data/openscene_lvpgh_logs.txt --out ~/data/openscene --dry-run
python3 scripts/data/fetch_openscene_sweeps.py --logs scripts/data/openscene_lvpgh_logs.txt --out ~/data/openscene
```

It writes `~/data/openscene/nuplan_db/<log>.db` (image table with per-camera timestamps, 100 Hz ego poses,
calibration) and `~/data/openscene/sensor_blobs/sweeps/<log>/<CAM>/<token>.jpg` (every frame not already in
`sensor_blobs/trainval/`; OpenScene keeps nuPlan's file names). It fetches only the listed logs' bytes: the
`.db` archives are zips (read by member), and the camera "zips" are tar files whose images are grouped by log,
which the script locates by probing and then streams. Reruns resume. At the end it checks every log against its
database's image table (`nuplan_fetch/summary.json`). Checked on one log (2021.05.12.22.00.38_veh-35_01008_01518,
in `train_camera_0`): 5100 images per camera at 100 ms, the OpenScene 2 Hz frames among them under the same
names. About 9 GB of images for that log; the 240 logs are roughly 0.9 TB (estimate from the 2 Hz sizes).

Downloaded on 2026-11-26: all 240 logs complete (5.42M database images, none missing; 25 GB of databases).

The index with the sweeps is a separate tree, `data/index/geo/openscene_sweeps/{train,val}` (same runs as
`geo/openscene`), written by Step 4's OpenScene script with `SWEEPS=1`:

```bash
SWEEPS=1 bash scripts/index/export_openscene.sh     # export_index.py --openscene-sweeps
```

Each run gets every database image between its first and last keyframe as a frame with its own camera timestamp,
`depth=None` and `seg=None`, and the database's ~100 Hz ego poses in the rig. The 2 Hz part (cameras, keyframes,
their poses, boxes) is identical to the `geo/openscene` pickle; checked field by field on one run. The database's
lidar-keyed ego pose equals NAVSIM's `ego2global` (0 m), and warping keyframes into adjacent sweeps with MoGe
depth has its lowest photometric error at the sweep's own timestamp (±20 / 50 / 100 ms offsets are worse), so the
sweep timestamps and poses line up. `configs/training/pmt_openscene_motion_sweeps.yaml` trains on it.

Known gaps: the keyframes keep NAVSIM's lidar timestamp, while the cameras fire −21 to +23 ms from it (median
ego motion in that time 5 cm, up to 25 cm); unchanged here so the context is as before. Sweeps have no masks,
so their ego-vehicle pixels are not masked out of the render loss (≤ 3% of CAM_L0 / CAM_R0).

## Step 2: MoGe-3 depth labels (required)

The depth labels are LiDAR-aligned MoGe-3 predictions: log-encoded uint16 PNGs of camera z-depth
(`near` / `far` in the PNG text). The loader converts them to ray distance. Training needs them: context
frames are drawn from frames with depth, and the distance losses need a target.

| Dataset | Where | Frames |
|---|---|---|
| AV2 | `data/moge3/align_depth/argoverse/<log>/<cam>/<t_ns>.png` | 2 Hz keyframes, all 1000 logs |
| nuScenes | `data/moge3/align_depth/nuscene/samples/<CAM>/<stem>.png` | trainval keyframes |
| Waymo | `data/moge3/align_depth/waymo/<segment>/images/<CAM>/<t_us>.png` | 2 Hz, 1000 segments |
| OpenScene | `data/moge3/align_depth/openscene/<log>/<CAM>/<token>.png` | every frame of the 240 "lvpgh" logs |

The labeling code is in `tools/moge_labels/` (README there: provenance, environments, run scripts). It runs in
the MoGe-3 venv on a GPU:

```bash
GPU=3 bash tools/moge_labels/run_nuscenes.sh
LOG_LIST=<av2 log ids> SPLIT=<name> GPU=3 bash tools/moge_labels/run_av2.sh
GPU=3 bash tools/moge_labels/run_waymo.sh     # also converts the Waymo tfrecords into data/waymo_ns (sjy-waymo env)
```

The labels in `data/moge3` today were **copied from DVGT** (2026-11-21; OpenScene hard-linked on 2026-11-23),
not regenerated here. There is no OpenScene run script in this repo yet: its labels came from DVGT's
`open_scene_loader.py` (num_tokens 2500), and the exporter reproduces that loader's depth grid. A relabel is a
label change; treat it as one (`.claude/rules/experiments.md`).

## Step 3: semantic masks and the LUT (optional)

**This step is optional, and this repo cannot do it yet.** The masks come from a Mask2Former inference run
(Swin-L IN21k, Mapillary Vistas v2 semantic config `maskformer2_swin_large_IN21k_384_bs16_300k.yaml`), which
lives in PseudoMapTrainer's `Mask2Former/` (`demo/inference_waymo.py`, `demo/inference_openscene.py`; AV2 and
nuScenes masks are RoGS's label directories). `data/m2f/*` are symlinks to those outputs.

Without masks the index still exports and trains. A frame without a mask has `seg = None`, and the loader then:
- sets `semantic_class` = 255 (ignored), so the semantic loss is skipped on that frame;
- skips the background loss when `background_requires_semantics: true`;
- gives the sky cubemap no target and does not blank depth on sky;
- weights every moving pixel 0.1 in `primitive_velocity` (vehicle pixels get 1.0 only with a mask).

If you produce masks, write them where the exporter looks and re-export (step 4, `REEXPORT=1` /
`REEXPORT_WAYMO=1`):

| Dataset | Mask path |
|---|---|
| AV2 | `data/m2f/av2/<split>/<log>/sensors/cameras/seg_<cam>/<t_ns>.png` |
| nuScenes | `data/m2f/nuscenes/{samples,sweeps}/seg_<CAM>/<stem>.png` |
| Waymo | `data/m2f/waymo/<segment>/seg_<CAM>/<t_us>.png` |
| OpenScene | `data/m2f/openscene/<log>/seg_<CAM>/<token>.png` |

Format: a single-channel uint8 PNG of **Vistas v2 class ids** (0–123), not of model classes. Waymo and
OpenScene masks may be predicted on the raw, distorted image (same size as the image; undistorted on load). An
OpenScene mask may instead be predicted on DVGT's MoGe image (846×476); the exporter tells the two apart by size
and rejects any other size.

**The LUT.** `tools/index_export/semantic_lut.py` maps Vistas v2 ids to the model's classes (OTHERS, EGO, SKY,
ROAD, MOVABLE, LANE, CROSSWALK, BOUNDARY; 255 = ignore). Every pickle stores it as `semantic_lut`, whether or
not masks exist; it only matters for frames that have one. The training config then folds it onto the head
it trains: the released 5-class head uses `semantic_remap: {5: 3, 6: 3, 7: 0}` (LANE and CROSSWALK → ROAD,
BOUNDARY → OTHERS). The map-class assignment follows RoGS's Vistas v2 remap, so both models agree on road,
lane, crosswalk and boundary; the table records only the resulting ids, with their Vistas names. Changing
the table changes every semantic label. Re-export after any change.

## Step 4: export the index

In `sjy-inurec`, which has `av2` and `nuscenes-devkit` for the AV2 and nuScenes readers
(`tools/index_export/README.md` has the install recipe; a plain `pip install` would replace the env's `cv2`).
CPU only. Each script skips pickles that already exist and writes each one to a temp directory before moving
it into place, so it can be rerun after an interruption.

```bash
cd /home1/irteam/workspace/instant-nurec
bash scripts/index/export_av2.sh        # data/index/av2/{train,val,test}: the 695 logs in scripts/index/av2_logs.txt
bash scripts/index/export_nusc.sh       # data/index/nusc/{train,val}: v1.0-trainval, 700 / 150
bash scripts/index/export_geo.sh        # data/index/geo/{av2,nusc,waymo}/{train,val}: needs the two above
bash scripts/index/export_openscene.sh  # data/index/geo/openscene/{train,val}: independent
```

| Variable | Default | Use |
|---|---|---|
| `INDEX_DIR` | `data/index` | write somewhere else, e.g. a scratch tree for a check |
| `JOBS` | `8` | parallel exporter processes |
| `REEXPORT_WAYMO=1` | off | `export_geo.sh`: rewrite the Waymo pickles (new masks) |
| `REEXPORT=1` | off | `export_openscene.sh`: rewrite every OpenScene pickle (new masks) |

What each tree holds:
- `geo/` is the split of record (`configs/splits/geo_near/`, the "near" lists shared with PMT and mapnn). Geo
  test is not indexed: it is held out.
- `geo/av2/train` and `geo/nusc/*` are relative symlinks into `av2/` and `nusc/`. AV2 geo-val and all of Waymo
  are exported into `geo/` directly. nuScenes scenes from the official test split have no depth and are
  skipped (geo train 554 of 657, val 99 of 124).
- OpenScene is exported per geo run `<log>:<first>-<last>`, so one log can have runs in both train and val.
  Runs shorter than 8 frames are skipped: at 2 Hz no T=8 context fits.
- `av2/` is pinned to `scripts/index/av2_logs.txt`. All 1000 AV2 logs have depth now, but
  `pmt_av2_pilot.yaml` and `pmt_av2_nusc.yaml` read `av2/{train,val}` whole, so adding logs changes their
  data. Extend the list on purpose.

One log by hand, e.g. to try a change:

```bash
python tools/index_export/export_index.py --dataset openscene \
  --scenes 2021.05.12.22.00.38_veh-35_01008_01518:183-256 --out-dir /tmp/index_test
```

`--dataset` is `av2` (with `--split train|val|test|auto`), `nusc`, `waymo` or `openscene`. The script prints
each camera's frame count and how many frames have a mask and depth.

### Check an export

Load an item through the training loader. This needs no GPU:

```bash
python - <<'EOF'
from pathlib import Path
from instant_nurec.training.run import load_training_config
from instant_nurec.datasets.pmt_index import PMTIndexDataset
c = load_training_config(Path("configs/training/pmt_openscene_motion_smoke.yaml"))
ds = PMTIndexDataset(c.dataset.train, global_seed=38); ds.set_rng_epoch(0)
lab = ds[0].context[0].data.camera.labels
print(len(ds), lab.rgb.shape, lab.metric_distance.shape, lab.semantic_class.unique())
EOF
```

`semantic_class` all 255 means the export found no masks. For a new dataset or rig, also overlay depth on the
images and train a smoke config (`.claude/agents/data-pipeline-engineer.md`, `.claude/rules/testing.md`).

## Point a config at the index

`PMTIndexDatasetConfig.index_paths` takes pickle files or directories of them (`configs/training/pmt_*.yaml`).
`camera_ids` must exist in every listed pickle, in the order the model sees them. Inference takes the same
pickles: `run_inference.py --index-path <pkl|dir>` (`docs/PMT_FINETUNE.md`).
