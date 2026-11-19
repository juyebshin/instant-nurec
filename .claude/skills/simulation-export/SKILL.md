---
name: simulation-export
description: Use Instant NuRec outputs for simulation — 3DGS PLY export, sky cubemap sidecars, dynamic-layer export, merge/voxelization, calibrated preview/video renders, NuRec per-scene refinement init, and novel-view (next-frame) render evaluation. Use when exporting scenes, rendering, checking render quality, comparing a fine-tuned checkpoint's renders, or preparing PLYs for NuRec/AlpaSim. Related — inurec-train-eval (render-phase PSNR during training), hdmap-from-gaussians (road subset).
---

# Simulation Export & Rendering

## What exists (upstream, released weights)

```bash
instant-nurec --model pa-multiview --ncore-path <seq.json|.lst> --output-dir <out> \
  [--merge [--n-gaussians 2000000]] [--include-dynamic [--motion-source tracks|learned]] \
  [--render-preview] [--merge --render-video]
```

The outputs go to `out/<run_id>/ply/<sequence_id>/<stem>.*`:

| File | Writer | Content |
|---|---|---|
| `<stem>.ply` | `predict/export_ply.py` | static 3DGS in the NCore world frame, with per-Gaussian `road_mask` and `sky_mask`. **Only those two masks: the 8-class semantics are not in the PLY** |
| `<stem>.sky.npz`, `.sky.png` | `predict/export_sky.py` | observed-sky cubemap plus per-camera affine ISP |
| `<stem>.dynamic.npz` | `predict/export_dynamic.py` | per layer: keyframe positions and timestamps, rotations, scales, rgb, max densities, falloff (`format_version 1`, `ncore_world`, `wxyz`) |
| `<stem>.render.png` | `--render-preview` | first context frame with sky composited |
| `<stem>.render.mp4` | `--merge --render-video` | every original frame of the first context camera; needs ffmpeg with libx264 |

- `--merge` merges the chunks by frustum ownership, then runs the KL-optimal voxelisation, searching
  the voxel size until it reaches `--n-gaussians`.
- Renders need the `render` extra (gsplat, nvdiffrast) and use calibrated rays and rolling-shutter
  poses, not a pinhole approximation.
- The PLY is the initialisation for NuRec per-scene refinement (`README.md` "What's next?"). View it
  in SuperSplat or NuRec's `ply_viewer`, not MeshLab.

## Limits that matter here

- **The CLI loads only released weights**, and only NCore V4 input. A fine-tuned `.ckpt` or AV2 /
  nuScenes logs need a custom driver. PMT `inurec_road_run.py` and mapnn `inurec_render_eval.py`
  show how to build `KelvinStaticCore` from a Lightning checkpoint and render with gsplat 3DGUT. Do
  not change the CLI's checkpoint check to make this work; add a separate path.
- **Learned sky vs observed sky.** Training renders against the *learned* `CubemapDecoderSky`, which
  the pilot trained from scratch. Inference builds an *observed* sky cubemap. A render comparison
  must say which sky it used.
- **Camera counts.** The `pa-multiview` CLI accepts 1/3/5 context cameras. The core does not
  enforce this (`research-goals.md`).
- **Dynamic layers** are dropped unless `--include-dynamic`. `learned` motion ignores cuboid tracks;
  `tracks` needs NCore cuboids, which PMT logs do not have.

## Render evaluation

These are two different questions. Do not mix their numbers:

| Question | Eval | Baseline |
|---|---|---|
| Does it reproduce its inputs? | re-render the newest context frame | the same eval on the comparison checkpoint |
| Does it extrapolate (simulation)? | next-frame render, +0.5 s (mapnn `inurec_render_eval.py --ckpt`) | **"hold"** (copy the previous frame) and the comparison checkpoint |

- Score on static pixels: PSNR / SSIM / LPIPS.
- The static mask comes from each model's own semantic head, so hold differs between models. Report
  hold for both.
- Current AV2 numbers (100 samples, `PMT_FINETUNE.md` §2): step 9000 scores 19.43 / 0.624 / 0.357,
  hold 17.14 / 0.457 / 0.335, zero-shot 15.52 / 0.469 / 0.571.
- A model that loses to hold on LPIPS has not won on perceptual quality, even with a PSNR lead.

## Rules

- A render claim names the checkpoint step, the sky source, the camera set, the resolution, the
  sample set and whether dynamic content was masked or rendered.
- Changing what the PLY stores (for example, adding semantic classes) is a format change for
  downstream NuRec. Make it additive (new property) and default-off, and say who consumes it.
- Check `README.md` and the CLI `--help` before quoting flags; this file is a summary.
