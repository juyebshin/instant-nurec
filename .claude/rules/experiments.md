# Experiment Hygiene

Binding rules for any run that is meant to support a claim.

## One variable

An experiment changes **one** thing against a named baseline. If several things must move, say so
and label the result confounded. For example, more cameras can force half-resolution supervision.

## Pin the baseline

- Report every result as a delta against a specific run: config path, commit, checkpoint step, and
  the eval script with *its* repo and commit.
- Not against a README number, a paper number, or memory.
- The released `pa-multiview` (zero-shot) and the AV2 step-9000 checkpoint are the baselines that
  exist today. Their numbers are in `docs/PMT_FINETUNE.md`.

## The evals and what each one measures

| Eval | Where | Measures | Caveat |
|---|---|---|---|
| `val/psnr` | training log, render phase | render quality on val supervision frames | phase-1 `val/psnr = 0.1 × epoch` is plumbing, never quality |
| Road test | PMT `inurec_road_run.py` + `inurec_road_test.py vis` | ROAD Gaussians vs LiDAR road, 0.1 m BEV grid: dz, median \|dz\|, IoU | T=8 windows over a whole log; logs 7fab / 02678d04 / 0b5142c1 |
| Next-frame render | mapnn `inurec_render_eval.py --ckpt` | static-pixel PSNR / SSIM / LPIPS of the held-out next frame (+0.5 s) vs "hold" | static mask = each model's own semantic head, so hold differs per model |
| Zero-shot geometry | mapnn `inurec_zeroshot_diag.py` | median xy error vs LiDAR in 0–10 / 10–20 / 20–30 m bands, range ratio | 200 samples per rig; MoGe-3 at T=1 is the control |
| Semantic share | ad hoc | % of pixels where a class is argmax | not accuracy; use it only to detect a class that never fires |

## Hold the protocol fixed

Across an A/B, hold these fixed unless one of them *is* the variable:
- `seed`
- index paths and splits
- camera set and context resolution (504×280)
- T and frame gap
- supervision resolution
- `include_context_frames`
- batch size, GPU count
- steps actually run
- the eval script's input crop, sample list and static mask

Record all of them with the result.

## Cross-repo evals

The evals live in PMT and mapnn, so a number depends on three commits. Record the commit of each
repo that touched the number. An input-prep change in either sibling repo is a protocol change, and
the crop bug came from there.

## Seeds and noise

Single-seed results are suggestive, not conclusive. Say "single seed" next to the number. Three
road-test logs are a small sample; say so when a difference is within the spread across logs.

## Negative results are results

Record what did not work, how far it ran and the likeliest reason, with the same care as a positive
result. The unexplained IoU drop from step 1000 to step 9000 is an open finding. Do not drop it.

## What must accompany a reported number

- config path, commit, checkpoint path and step
- command run, GPU, steps or epochs completed (say so if the final weights were not saved)
- the eval script, its repo commit, and the sample or log set
- the raw metric table, not only the headline
- the baseline it is compared against
- anything that makes it non-comparable: a confound, reduced resolution, a partial run, a single seed
