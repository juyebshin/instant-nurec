# Geographically disjoint "near" splits

Copied 2026-11-21 so this repo does not read the sibling repos. Do not hand-edit; re-copy from the
source if it changes, and treat any change as a protocol change (`.claude/rules/experiments.md`).
These are the splits PseudoMapTrainer and mapnn train and evaluate on, so geo-split numbers here are
comparable with theirs.

| dataset | train / val / test | source (unchanged copy) |
|---|---|---|
| `av2/` | 695 / 150 / 150 log ids (5 of 1000 logs unlisted) | PseudoMapTrainer `MapVR/custom_tools/maptrv2/av2/geosplits/near`, commit `cf022f5`; identical to mapnn's |
| `nusc/` | 657 / 124 / 137 scene names (82 of 1000 unlisted) | PseudoMapTrainer `MapVR/custom_tools/maptrv2/nusc/geosplits/near`, commit `cf022f5`; identical to mapnn's |
| `waymo/` | 700 / 151 / 149 segment names (a full partition) | mapnn `tools/maptrv2/waymo/geosplits/near`, commit `8ef6637`; how it was built: `waymo/MANIFEST.md` |
| `openscene/` | 496 / 336 frame runs `<log>:<first>-<last>` (no test), over the 240 MoGe-3-labelled logs (140 Pittsburgh, 100 Las Vegas) | mapnn `tools/maptrv2/openscene/geosplits/near`, commit `ede8699` (copied 2026-11-23); `openscene/MANIFEST.md`. Runs split logs, with ≥ 70 m between train and val frames |

"near" = held-out regions in the same cities as training. nuScenes lists also name official-test
scenes (103 / 25 / 15), which have no MoGe-3 depth here and so are not indexed.
