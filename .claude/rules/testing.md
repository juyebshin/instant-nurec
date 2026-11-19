# Testing & Verification

`tests/` is upstream's suite: about 600 tests, mostly CPU, collected through `conftest.py`. The
training extra is needed to collect all of them. `pytest` is not installed in `sjy-inurec`; install
it into the env before running. **No test covers this branch's additions** (`PMTIndexDataset`,
`_widen_semantic_head`, `semantic_class` targets, `init_full_keep_heads`).

State the success criterion before running, then produce the run (`CLAUDE.md` #4).

## Evidence ladder

| Change | Required evidence |
|---|---|
| Docs, comments | none; say so plainly |
| Config value only | `load_training_config(Path(cfg))` validates, and the printed resolved values are what you meant (watch the `model_post_init` overrides) |
| Upstream code with existing tests | the relevant `tests/test_*.py` passes before and after |
| New module, loss, dataset or head | smoke run: one optimiser step, `last.ckpt` written, resume for one more step (`pmt_av2_smoke.yaml` pattern, `docs/TRAINING.md` "bounded end-to-end smoke test"); every loss term finite |
| Anything that can move quality (geometry, losses, labels, crop, sampling) | an eval with numbers against the baseline checkpoint (`experiments.md`) |
| Train vs inference input path | same frames and same checkpoint through both paths: predicted-to-label distance on road agrees, as the `PMT_FINETUNE.md` §2 contract check did (1.020 vs 1.013) |
| Change to the released inference path | the released `.pth` still loads with `strict=True`, and the relevant CLI / inference tests pass |

## Reporting rules

- Name the exact command, config, GPU and how far it actually ran.
- A smoke run is not a quality result. Write "smoke only, no eval" rather than implying otherwise.
- If a run was cut short, crashed or was skipped for lack of a GPU, say that instead of reasoning
  about what it would have shown.
- Training is not bitwise deterministic, even with the seeded sampler. Hold the seed fixed anyway.

## If you add tests

- Put them under `tests/` and match the existing style: plain pytest, CPU, synthetic tensors.
- Good unit-test surfaces on this branch:
  - `pmt_index.read_log_depth` / `z_to_ray_distance` / `pinhole_parameters`, on a synthetic K and image;
  - `_semantic_targets` with `semantic_class` set (255 ignored, EGO override);
  - `_widen_semantic_head` (argmax unchanged at step 0).
- Anything that needs real index pickles, a GPU or gsplat is an integration run. Do not fake it with
  mocks.
