---
name: hdmap-from-gaussians
description: HD-map uses of Instant NuRec Gaussians — road-surface and road-marking (lane/crosswalk/boundary) Gaussians, the 8-class semantic head, initialising RoGS in PseudoMapTrainer, and the road test (height error and BEV IoU vs LiDAR). Use when working on road reconstruction, map-class semantics, rare-class problems, RoGS init, or interpreting road-test numbers. Related — inurec-train-eval (running the road test), multi-rig-training (geometry across rigs), experiment-scientist agent.
---

# HD Maps from Gaussians

## Pipeline today

1. **Semantics.** The 8-class head:
   - OTHERS 0, EGO 1, SKY 2, ROAD 3, MOVABLE 4, LANE 5, CROSSWALK 6, BOUNDARY 7.
   - With classes 5–7, ROAD means road surface *without* paint.
   - The released 5-class head is widened by `_widen_semantic_head` (`training/system.py`). LANE and
     CROSSWALK start from the ROAD row and BOUNDARY from the OTHERS row, all with bias −2, so the
     argmax at step 0 is unchanged.
2. **Labels.** M2F Mapillary-v2 masks, mapped by the index `semantic_lut` (built in PMT
   `inurec_finetune_index.py`), with 255 = ignore. They exist only for AV2 and nuScenes.
3. **Road Gaussians.**
   - PMT `inurec_road_run.py` runs non-overlapping T=8 windows over a whole log.
   - It keeps every pixel-aligned Gaussian whose semantic argmax is ROAD, in the RoGS world frame
     (the scene's first chassis pose), and writes `road_gaussians.npz`.
   - This is the seed meant for RoGS.
4. **Road test.** `inurec_road_test.py vis` compares them against the LiDAR road on a 0.1 m BEV grid:
   - dz = median height error;
   - median |dz|;
   - IoU of road coverage.

Today the run keeps ROAD only. Lane, crosswalk and boundary Gaussians are not exported for RoGS, and
the PLY export carries only `road_mask` / `sky_mask`. Using them is new work.

## Current numbers (AV2, `PMT_FINETUNE.md` §2)

- **Height:** step 9000 is at MoGe-label level (|dz| 0.025–0.039 m). Zero-shot was +0.59 to +0.66 m.
- **Coverage:** IoU is 2–4 points *below* step 1000 on all three logs. The road pixel share is
  unchanged, so the loss is in where the road points land, not in the semantics. **Cause not
  found.** This is the main open HD-map question.
- **Rare classes:** at step 9000, LANE is argmax on 0.4–1.1% of pixels and CROSSWALK on ≤ 0.1%. At
  step 1000 neither was ever argmax.

## Working on it

- **Diagnose IoU before tuning semantics.** Hypotheses consistent with "same road share, lower IoU":
  - points spread farther or nearer (a range-dependent scale change);
  - more road Gaussians at long range, where 0.1 m BEV cells are sparse;
  - window-boundary misalignment.

  Cheap probes first: IoU per range band, BEV density maps at step 1000 vs 9000 on the same log.
  `experiment-scientist` designs the comparison.
- **Rare classes.** Candidates, each a one-variable run:
  - class-weighted or focal CE for 5–7;
  - more `primitive_semantics` weight (already 0.1, 10× upstream);
  - crosswalk-rich log sampling.

  Measure per-class argmax share *and* a per-class road-test-style overlay, not only the total loss.
- **Exporting map classes to RoGS.** An additive change in PMT `inurec_road_run.py`: keep LANE,
  CROSSWALK and BOUNDARY with their class id. Check that RoGS's init reads a class channel before
  promising it does. That code lives in `../PseudoMapTrainer/RoGS`.
- **Geometry gates every map metric.** Before reading a road-test change on a new rig, run the
  zero-shot geometry read on that rig (`multi-rig-training`).

## Rules

- Road-test logs are fixed: 7fab (in no PMT split), 02678d04 and 0b5142c1 (val). Add logs; do not
  swap them. Always report the MoGe-3-labels row as the label ceiling.
- Report dz, median |dz| and IoU per log, not an average over three.
- A semantic-head change must keep `KelvinSemanticClass` positional and append-only. Reordering
  breaks every checkpoint and the exporter's LUT.
