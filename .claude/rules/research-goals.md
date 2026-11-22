# Project Goal & Standing Constraints

## North star

Build on Instant NuRec's feed-forward Gaussians as **one scene representation shared by several
autonomous-driving tasks**, trained to hold across sensor rigs:

1. **Geometry that transfers.** Metric depth scale and shape that hold on rigs the model was not
   trained on. Every downstream task reads positions from the Gaussians, so a scale error hurts
   all of them.
2. **HD maps.** Road-surface and road-marking Gaussians that initialise RoGS in PseudoMapTrainer,
   and geometry that mapnn can consume.
3. **Occupancy.** 3D semantic occupancy derived from, or trained on, the same Gaussians.
4. **Simulation.** Static and dynamic layers plus sky that render well from novel views and
   initialise NuRec per-scene refinement.

When a change is ambiguous, prefer the option that keeps the representation shared over the one
that adds a task-specific branch.

## Where the repo actually is

Be accurate about this in every plan and report. Sources: `docs/PMT_FINETUNE.md`, `docs/TRAINING.md`,
`README.md`.

| Capability | Status |
|---|---|
| Feed-forward static + dynamic 3DGS, sky cubemap, per-camera ISP from NCore V4 | implemented (upstream) |
| Released profiles `pa-front`, `pa-multiview` (1/3/5 cameras, 504×280, T=18), `pq-front` | implemented (upstream) |
| Two-phase dense front-camera training | implemented (upstream) |
| Upstream random 1/3/5-camera, 12/16/18-frame multiview curriculum | **not in this repo** |
| Fine-tune on non-NCore logs (`PMTIndexDataset`; the exporter handles AV2, nuScenes and Waymo) | implemented. AV2 pilot done: 2 epochs, val PSNR 22.99, but `last.ckpt` holds step 9000 of 9820. AV2 + nuScenes (`pmt_av2_nusc.yaml`) done 2026-11-21: 2 epochs, final step 2454 saved, val PSNR 21.56; road test level with step 9000, next-frame render −0.7 dB PSNR but LPIPS now beats hold (confounded vs the pilot). AV2 + nuScenes + Waymo on geo splits (`pmt_geo_av2_nusc_waymo.yaml`, 2026-11-21): indexes built, Waymo path smoke-tested only; 5-class head since 2026-11-22; **not trained yet.** |
| 8-class semantic head (+ LANE, CROSSWALK, BOUNDARY) | implemented, dropped for the next run (2026-11-22). Class IoU vs LiDAR on 7fab: CROSSWALK 0.00–0.09 (RoGS 0.69). At step 9000, LANE is argmax on 0.4–1.1% of pixels and CROSSWALK on ≤ 0.1%. BOUNDARY not reported. |
| HD map: ROAD Gaussians vs LiDAR road (road test, 3 AV2 logs) | measured. Height reaches the MoGe-label level (median \|dz\| 0.025–0.039 m). IoU is 2–4 points below step 1000. Likely cause (2026-11-22): the LiDAR GT includes boundary points. Against GT road surface, step 9000 is ahead on all 3 logs (single seed). |
| RoGS training initialised from these Gaussians | **not reported here.** Do not claim it. |
| Multi-rig geometry | **planned** (`PMT_FINETUNE.md` §3). Zero-shot range ratio is 0.47 (Waymo) to 1.01 (HRDX). Waymo moved into training (geo split) 2026-11-21; HRDX is the held-out rig. |
| Index export for DDAD / KITTI / OpenScene | **not implemented** |
| Varying camera count or aspect from step to step in training | camera count verified (7 ↔ 6 at batch 1, AV2 + nuScenes smoke, 2026-11-20). **Aspect unverified**: always 504×280. |
| Save on train end | fixed 2026-11-20 (`training/run.py::_StepCheckpoint`) |
| Occupancy: head, GT loader, metric | **not implemented.** `KelvinStaticLayer.voxelize` merges Gaussians; it is not occupancy. |
| Simulation export: PLY, sky, dynamic `.npz`, calibrated preview / video | implemented (upstream), released weights only. The public CLI does not load fine-tuned `.ckpt`. |
| Next-frame render on AV2 (static pixels) | measured. Step 9000: 19.43 dB vs hold 17.14. LPIPS still favours hold. |
| Tests for this branch's additions | **none** |

Never describe an unbuilt capability as existing, in code comments, docs or a report. If a plan
depends on one, call it a prerequisite.

## Roadmap

From `PMT_FINETUNE.md` §3. The order is agreed. Do it before any long run:

0. **Save on train end**, which closes the `last.ckpt` gap. Prerequisite for every later run.
1. **Export DDAD indexes, then a DDAD-only smoke run** (6 cameras, 1936×1216).
2. **Verify varying camera count and aspect** through the training path from step to step.
3. **Short AV2 + nuScenes + DDAD probe** for peak memory and s/step.
4. Then the multi-rig fine-tune, with HRDX held out as a rig (Waymo trains on its geo split since
   2026-11-21, user decision).

Occupancy and simulation work comes after step 4 unless the user reorders it. Both read the same
geometry, and that geometry does not yet transfer across rigs.

Open decisions belong to the user. Do not settle them inside a code change:
- held-out rigs (decided 2026-11-21: Waymo in training on its geo split, HRDX held out);
- keeping the 8-class head (decided 2026-11-22: the next run trains the released 5-class head; map classes
  are assigned after the fact by projection + voting or RoGS optimisation, which is not built yet);
- initialising from step 9000 or from the released weights;
- the GPU budget.

## Standing constraints

**Do not break the released contract.**
- Released checkpoints must still load with `strict=True` into `KelvinStaticCore`.
- The public CLI must produce the same outputs for the same inputs.
- New behaviour goes behind a default-off config field, as `init_full_keep_heads` and
  `n_semantic_classes` do.

**Comparability outranks novelty.** The road test, the render eval and the zero-shot diag are
protocols. A change to the input crop, the frame selection, the static mask or the sample set
invalidates every earlier number. The 2026-11-19 crop bug moved the depth scale silently, and the
"−0.47 m overshoot" it produced was read as a model effect for days. Announce such changes, date
them, and re-run the baseline. See `experiments.md`.

**Geometry before tasks.** A downstream metric (road IoU, occupancy IoU, render PSNR) sits on top of
depth scale. Check the geometry read before attributing a task-level change to the task head.

**Readout before new head.** For a new task, first try reading it out of the existing Gaussians
(semantic argmax, densities, voxelisation) with no training. That is the honest baseline for "does a
trained head help", and it costs no GPU.

**Labels are partial by construction.**
- Semantic masks exist for AV2 and nuScenes only. Depth exists on 2 Hz keyframes only.
- A missing label is ignored (`semantic_class = 255`) or its loss weight is 0. It is never filled
  with a class.
- Missing data must not crash a batch or silently drop a sample.

**Held-out stays held out.**
- HRDX is the held-out rig. Waymo has MoGe-3 labels since 2026-11-20 and trains on its geo split; its
  geo-test segments stay held out.
- Geo splits (PMT / mapnn "near" lists) are the split of record for comparisons with PMT and mapnn.
- The AV2 test split (103 logs) is held out.
- Split by whole log, never by frame.

**One loader for non-NCore data.** Extend `PMTIndexDataset` and the PMT exporter. Do not add a
parallel dataset class per source; its training contract is the one that was checked.
