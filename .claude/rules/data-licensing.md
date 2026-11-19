# Data & Weight Licensing

A checkpoint inherits the terms of every dataset it trained on, every label source and every
initialisation. Terms differ, and some restrict commercial use. The constraint attaches at training
time, not at release time.

## Rules

- **Flag, do not decide.** Licensing is the user's call. Name the specific term and the affected
  runs, and let the user choose. Do not silently include or exclude a source.
- **Do not restate terms from memory.** Link the actual terms and record the version you read.
- **Record provenance at ingestion.** When a dataset, label set or checkpoint enters the project,
  record its source, version, licence and any non-commercial or share-alike obligation in the same
  change.
- **Never bypass access gates.** Waymo, nuScenes, AV2 and others need registration. Ask the user
  to obtain access; do not look for mirrors.
- **Keep data and weights out of git.** `checkpoints/`, `*.pth`, `*.ckpt` and `*.pkl` are
  gitignored; keep it that way.

## Known terms to check in this repo

| Source | Where the terms are | Why it matters |
|---|---|---|
| nvdiffrast (`render` / `training` extras) | `THIRD_PARTY_LICENSE.txt` (NVIDIA Source Code License, 1-Way Commercial) | non-NVIDIA use limited to non-commercial research or evaluation; every training run uses it |
| Released Instant NuRec weights | Hugging Face model card `nvidia/instant-nurec` | every fine-tune initialises from them |
| DAv3 Base initialisation | Hugging Face `depth-anything/DA3-BASE` | phase-1 initialisation |
| Waymo Open Dataset | waymo.com/open/terms | named in `docs/TRAINING.md`; candidate held-out rig |
| Argoverse 2, nuScenes, DDAD, KITTI, OpenScene | each dataset's own terms | training data for the multi-rig plan |
| MoGe-3 depth labels (via DVGT) and Mask2Former Mapillary-v2 masks | the MoGe-3, Mask2Former and Mapillary Vistas terms | derived labels inherit the terms of the model and data that produced them |

The sibling repo `../mapnn/docs/data-provenance.md` already records several of the driving
datasets. Read it before re-deriving terms.
