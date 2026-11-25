# Index export (AV2 / nuScenes / Waymo / OpenScene)

Writes the per-log index pickles that `PMTIndexDataset` (`instant_nurec/datasets/pmt_index.py`) reads. The
pickle format and what the loader does with it are described in that module's docstring and in
`docs/PMT_FINETUNE.md`.

| File | What |
|---|---|
| `export_index.py` | one pickle per log / scene / OpenScene geo run: `--dataset {av2,nusc,waymo,openscene} --scenes ... --out-dir ...` |
| `semantic_lut.py` | Mapillary Vistas v2 id → `KelvinSemanticClass`, stored in every pickle as `semantic_lut` |
| `../../scripts/index/export_{av2,nusc,geo,openscene}.sh` | batch export into `data/index/` (`INDEX_DIR`, `JOBS` override) |

The index tree is `data/index/` (gitignored): `av2/{train,val,test}`, `nusc/{train,val}`,
`geo/{av2,nusc,waymo,openscene}/{train,val}`. `geo/av2/train` and `geo/nusc/*` are relative symlinks into
`av2/` and `nusc/`. Run order for a full rebuild: `export_av2.sh`, `export_nusc.sh`, then `export_geo.sh`;
`export_openscene.sh` is independent.

## Environment

`sjy-inurec`. The AV2 and nuScenes readers use `av2==0.2.1` and `nuscenes-devkit==1.1.11` (the versions
`sjy-rogs` had), installed into `sjy-inurec` on 2026-11-25 with `--no-deps`, plus their runtime deps:
`pyarrow==17.0.0 pyquaternion==0.9.9 matplotlib shapely descartes cachetools fire pycocotools scikit-learn
numba pyproj rich click joblib av tqdm`. Nothing already in the env changed version. `opencv-python` was
deliberately left out: it would replace the env's `cv2` (opencv-python-headless 4.10.0.84), which the
dataloader uses for undistortion and resizing. `pip check` therefore reports the devkits' unmet
`opencv-python`, `nox`, `matplotlib<3.6` and `Shapely<2` requirements; the exporter's code paths don't need
them.

## Provenance

- Moved on 2026-11-25 from PseudoMapTrainer `RoGS/scripts/inurec_finetune_index.py` (untracked in that repo)
  and the launch scripts in its gitignored `RoGS/output/inurec_finetune/logs/` (`export_av2_all.sh`,
  `export_nusc_all.sh`, `export_geo.sh`, `export_openscene.sh`). Changes: repo-relative paths, `sjy-inurec`
  instead of `sjy-rogs`, the LUT read from `semantic_lut.py`, `INDEX_DIR` / `JOBS`. Export logic unchanged.
- `semantic_lut.py` is the PMT exporter's LUT written out as a table. Its map classes follow RoGS's
  `MapillaryVistasV2` remap (AGPL-3.0 in PMT); only the resulting id → class assignment is recorded here.
- Check of the move: every existing pickle (AV2 695, nuScenes 850, geo AV2 val 150, Waymo 851, OpenScene
  750, plus the geo symlinks) was re-exported with this code and compared field by field with the pickle in
  use. See `docs/PMT_FINETUNE.md` for the result.
