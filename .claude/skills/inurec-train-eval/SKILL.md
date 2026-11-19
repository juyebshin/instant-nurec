---
name: inurec-train-eval
description: Run and verify Instant NuRec training and evaluation in this repo. Use when asked to train, fine-tune, smoke-test, resume, pick a checkpoint, or verify that a model/loss/data change actually works. Covers instant-nurec-train, the PMT smoke and pilot configs, the context/render phases, checkpoint layout and the last.ckpt gap, reading val/psnr, and the three downstream evals (road test, next-frame render, zero-shot geometry). Related — inurec-config (editing YAML), inurec-env (toolchain and import errors), multi-rig-training (the next run).
---

# Instant NuRec Train & Eval

A run is the test for anything this branch added: no unit test covers it (`.claude/rules/testing.md`).
State the pass criterion before you launch.

## Before launching

1. `nvidia-smi`: the GPUs are shared. Pick a free one and pass it in `CUDA_VISIBLE_DEVICES`.
2. Export the gsplat toolchain block from `CLAUDE.md` (append `sjy-mapnn/bin` to `PATH`, do not
   prepend it).
3. Validate the config and print the values that `model_post_init` may have rewritten:
   ```bash
   python -c "from pathlib import Path; from instant_nurec.training.run import load_training_config as L; c=L(Path('<cfg>')); print(c.phase, c.system.enable_render_global_step, c.system.strategy, c.loss)"
   ```
4. Set `INSTANT_NUREC_RUN_ID` to a readable name. Otherwise the run directory is a random shortuuid.

## Commands

```bash
# Smoke: one step at 4 frames x 196x112, then checkpoint
INSTANT_NUREC_RUN_ID=smoke-$(date +%m%d-%H%M) CUDA_VISIBLE_DEVICES=<gpu> \
  instant-nurec-train --config configs/training/pmt_av2_smoke.yaml

# Long run: always detached, always logged
INSTANT_NUREC_RUN_ID=<run> CUDA_VISIBLE_DEVICES=<gpu> setsid nohup \
  instant-nurec-train --config <cfg> > <out_dir>/<run>.log 2>&1 &
```

Outputs go to `<out_dir>/<run_id>/`: `resolved.yaml`, `checkpoints/{last.ckpt, epoch=…-step=….ckpt}`
and the logger directory. The PMT runs use `logger.name: csv` and live under
`../PseudoMapTrainer/RoGS/output/inurec_finetune/runs/`.

**Resume** = copy that run's `resolved.yaml`, add `resume_from_checkpoint: <…>/last.ckpt`, and keep
the seed, data, batch size, world size and schedule unchanged (`docs/TRAINING.md`). **Initialising a
new run** from a checkpoint is `model.init_weights_paths.full` instead; optimiser state and step start
fresh.

## Smoke-run pass criteria

- The optimiser step completes, and `checkpoints/last.ckpt` is written.
- Resuming it runs one more step (the `docs/TRAINING.md` contract).
- Every configured loss term is present and finite.
- The "Skipping training losses with unavailable labels" warning lists only what you expected, such
  as `primitive_velocity` (PMT logs have no cuboids).

## Known gap: final weights are not saved

`make_checkpoint_callback` (`training/run.py`) uses `every_n_train_steps` with `save_last`, which does
not save at train end. Up to `save_every_n_train_steps − 1` steps are lost; the AV2 pilot's
`last.ckpt` is step 9000 of 9820. Until this is fixed, report the step a checkpoint *actually* holds
(`torch.load(p, weights_only=False)["global_step"]`), not the planned length.

## Reading training metrics

- `train/psnr` and `val/psnr` mean something only in the render phase. In phase 1, `val/psnr = 0.1 × epoch`
  is checkpoint plumbing.
- PSNR is one global `10·log10(1/MSE)` over `RGB_LABEL` and non-`INVALID` pixels, after the sky
  composite and the affine.
- The LR schedule moves only per epoch (`current_epoch / max_epochs`), so a mid-epoch stop is at full
  LR.
- The affine output is detached until `post_processing.optimization_start_global_step` (1000).

## Downstream evals

These run in sibling repos and other envs; see `.claude/rules/experiments.md` for what each one
measures.

```bash
# Road test (HD map). The export and vis steps run in sjy-rogs; run is in sjy-inurec.
cd ../PseudoMapTrainer/RoGS
python scripts/inurec_road_test.py export --dataset av2 --scene <log> --out-dir output/inurec_road/<tag>
CUDA_VISIBLE_DEVICES=<gpu> python scripts/inurec_road_run.py --requests output/inurec_road/<tag>/requests.pkl --ckpt <ckpt>
python scripts/inurec_road_test.py vis --out-dir output/inurec_road/<tag>

# Next-frame render and zero-shot geometry: export + score in sjy-mapnn, forward in sjy-inurec
cd ../mapnn   # see the docstrings of tools/analysis_tools/inurec_{zeroshot_diag,zeroshot_run,render_eval}.py
```

Check each script's docstring for the exact flags before running it. They are the source of truth,
not this file.

## Reporting

Report:
- the config, the run ID and the GPU;
- the steps actually run, and the step the evaluated checkpoint holds;
- the eval script and its repo commit;
- the raw table.

Say "smoke only, no eval" when that is what happened.
