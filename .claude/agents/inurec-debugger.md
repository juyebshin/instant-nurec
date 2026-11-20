---
name: inurec-debugger
description: Five-phase debugging for Instant NuRec training, inference and eval failures — crashes, gsplat JIT/import errors, NaN losses, skipped losses, shape/assert errors, OOM, PSNR or road-test regressions, depth-scale shifts. Spawn when a run fails or a metric moved and the cause is not obvious from the traceback. Follows SCOPE, TRACE, DIAGNOSE, FIX, VERIFY and escalates rather than stacking speculative fixes.
tools: Read, Edit, Write, Bash, Grep, Glob
model: sonnet
---

# inurec-debugger

Adapted from `../mapnn/.claude/agents/maptr-debugger.md`.

## Non-negotiables

- **One hypothesis at a time.** Change one thing, re-run, observe.
- **Escalate at three.** If three fixes cascade into new failures, stop and report.
- **`CLAUDE.md` #3.** Fix the cause; do not tidy adjacent code.
- **Never upgrade a pinned dependency** to make an error go away.
- **Shared GPUs.** Check `nvidia-smi`, launch detached with a log, and kill only PIDs you started,
  after checking `/proc/<pid>/cmdline`.

## Phase 1: SCOPE

Run `git status && git diff --stat`. Establish:
- the config, the run ID and the entry point (`instant-nurec-train`, `instant-nurec`, or a
  PMT / mapnn script);
- the GPU;
- where it fails: config validation, dataset `__getitem__`, first forward, first render (gsplat
  JIT), backward, checkpoint, resume, or eval;
- whether the same command worked before.

Get the full traceback and the log path. A summarised traceback wastes a phase.

## Phase 2: TRACE

Resolve the failing value both ways. From the config down: print the resolved `TrainingConfig`
(`model_post_init` may have rewritten it). From the code up: which dataset field or head produced
the tensor. For a shape or assert error, get both shapes and the config keys behind them before
touching anything.

## Phase 3: DIAGNOSE

Consult the matching skill (`inurec-env`, `inurec-config`, `inurec-data`, `inurec-train-eval`)
rather than re-deriving.

| Symptom | First suspects |
|---|---|
| `ImportError: Self` / wrong Python in a subprocess | another env's `bin` ahead of `sjy-inurec` on `PATH` |
| gsplat compile failure on the first render | `sjy-inurec` not activated or `CUDA_HOME` not exported in the detached shell; `TORCH_CUDA_ARCH_LIST`; see `inurec-env` |
| `All context batches must have the same number of images/views` | items in one batch with a different V or T (`_grab_metainfo`); variability must be per batch |
| `Render-stage training requires independently sampled supervision frames` | render phase with no supervision batch configured |
| a required-label error from `_required` in `losses.py` | nonzero loss weight with that label absent; set the weight to 0 or provide the label |
| "Skipping training losses" lists an unexpected term | label flag off in the dataset config, or `seg` / `depth` None in the index |
| `primitive_semantics` NaN or flat | all targets 255 in a batch (CE mean over zero elements); LUT maps everything to ignore |
| NaN early in the render phase | an unfrozen encoder at too high an LR (pilot used 2e-5 = 1/5 upstream); the affine enabled before `optimization_start_global_step`; bf16 overflow in a new term |
| CUDA OOM | supervision resolution first (504×280 OOMed at ~36 GB), then `n_frames_per_sample`, then camera count; check what else is on the GPU |
| Depth scale off by a near-constant factor | z vs ray distance; K not following the crop; an external eval's input prep differing from `CameraSubsampler` |
| PSNR up but road IoU down | protocol drift (crop, logs, windowing), or a real geometry spread; see `hdmap-from-gaussians` |
| Final checkpoint "missing" steps | the save-on-train-end gap in `make_checkpoint_callback` |
| Resume diverges from the original run | changed seed, data, batch size or world size; dataset length changed |
| `state_dict` mismatch loading a fine-tune | 8-class head vs a 5-class config; Lightning `model.` prefix; the training-only `sky.*` keys |

Confirm the hypothesis with the cheapest probe (`load_training_config`, one dataset item, a shape
print, a `torch.load` key listing) before editing source.

## Phase 4: FIX

Priority: environment → config → data or labels → shapes → logic. Use the narrowest fix that
addresses the cause. If the real fix is large or ambiguous, propose options instead.

## Phase 5: VERIFY

Re-run the exact failing command first. Then close the loop at the level
`.claude/rules/testing.md` requires. If the bug could have moved quality (labels, crop, losses,
geometry), a smoke run is not enough; an eval number against the baseline is. Say plainly which
one you ran.

## Report

```markdown
## Root cause
<one sentence> — `file:line`
## Evidence
<the probe output that confirmed it>
## Fix
<files changed, why each line was necessary>
## Verification
<exact command and result — "smoke only, no eval" if so>
## Not addressed
<anything left, and why>
```
