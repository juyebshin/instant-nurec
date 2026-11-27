# Renders for a fixer (ArtiFixer): RGB, opacity and cameras from an OpenScene log

How to get rendered RGB frames, their Gaussian opacity and the camera intrinsics and extrinsics out of this repo,
for a diffusion fixer such as NVIDIA ArtiFixer (<https://github.com/nv-tlabs/artifixer>). Two parts:
1. what already exists on this machine (the traffic-sim override of one OpenScene log);
2. how to do the same from scratch on another machine.

Written 2026-11-27. How the traffic-sim trajectory replaces a logged vehicle is described in
`world-x/docs/trafficsim-to-image.md` and `docs/world-model.md`.

```
OpenScene raw data ─► tools/index_export ─► index pickle ─► run_inference.py ─► 3DGS scene ─► renders
(NAVSIM log pkl,      (CPU, seconds)        (cameras, ego     (one forward pass   (PLY + dynamic   RGB + opacity
 camera jpgs)                                poses, boxes)     per 4 s chunk)      + sky + ISP)    + cameras
                                                                                       │
                                                         tools/track_edit.py (optional: replace a vehicle's path)
```

## 1. What exists on this machine

### Renders of the traffic-sim override

`outputs/world-model/2216-video-scene4_slot11/` holds two renders of the same frames, side by side:

| folder | scene |
|---|---|
| `override/` | the vehicle `5f64ff7fe8375dc0` re-posed onto the traffic sim's path (`lidar_box_override.csv`) |
| `log/` | the unedited reconstruction, as a reference |

Each folder contains:

```
images/          504x280 RGB PNG        (the model's input camera)
opacity/         504x280 8-bit gray PNG
images-1080p/    1920x1080 RGB PNG      (the index's undistorted camera)
opacity-1080p/   1920x1080 8-bit gray PNG
render.mp4, render-1080p.mp4            (H.264, 10 fps, the same frames)
```

- **Frames:** CAM_F0 (front camera), 10 Hz, from t0 − 1.0 s to t0 + 4.3 s (54 frames), where t0 =
  1631646461700169 µs. This window ends before the vehicle's first contact with the ego (t0 + 4.4 s). It covers the
  same frames as `compare_log_vs_override.mp4`.
- **File names:** the frame time in µs, e.g. `1631646460700169.png` … `1631646466000169.png`.
- **Camera path:** the ego's logged trajectory. The 2 Hz index poses are interpolated, linear in position and
  slerp in rotation.
- **Opacity:** accumulated Gaussian alpha, 255 = fully covered by Gaussians, 0 = only sky shows. This is the same
  format as `run_inference.py --save-video-opacity`.

**How they were rendered.** The renderer is the same one `run_inference.py --render-video` uses: gsplat 3DGUT, sky
composited behind the Gaussians, then the camera's ISP affine. Pixel values are rounded the same way.

**Check.** At frames where the times agree to within 0.13 ms, the `log/` renders match `run_inference`'s own
renders of the scene (`outputs/predict/openscene-2216-video/ply/<log>/images*`): about 0.01 % of pixels differ by
more than 3/255.

**Command.** They were made with `tools/render_frames.py` (section 3). It loads the exported scene with
`tools/viewer.py`'s `Scene` and renders each frame. Re-rendering with it gives the same PNGs up to gsplat's
run-to-run noise: at most 2/255, on 0.0003 % of values.

### Renders of the unedited scene from `run_inference`

`outputs/predict/openscene-2216-video/ply/<log>/` has `images/`, `opacity/`, `images-1080p/` and
`opacity-1080p/`. These are 94 CAM_F0 frames at the index's 2 Hz frames, over the first ~47 s of the log. It also
has `source.mp4` / `source-1080p.mp4`, the real camera images of those frames.

The log's camera images exist at 2 Hz only. The 10 Hz renders above therefore have no real camera image for most
frames.

## 2. Cameras: intrinsics and extrinsics

### Conventions

| what | convention |
|---|---|
| camera axes | OpenCV: x right, y down, z forward |
| extrinsics | camera → world 4×4 (`c2w`); world → camera is its inverse |
| world | global UTM (nuPlan's frame) in the index. Scene exports use UTM − `world_origin`, the first ego position, stored in `<log>.cameras.npz`, to keep float32 precise |
| intrinsics | pinhole `K`, no distortion: the images are undistorted on load. **fx ≠ fy**, e.g. CAM_F0 fx 1315.9, fy 1476.1 at 1920×1080 |
| time | µs, nuPlan / OpenScene clock |

### In the index pickle (`data/index/.../<log>_<a>-<b>.pkl`)

```python
import pickle
ix = pickle.load(open(pkl, "rb"))
cam = ix["cameras"]["CAM_F0"]
K, (w, h) = cam["K"], cam["wh"]          # undistorted pinhole, 1920x1080 (cv2.getOptimalNewCameraMatrix, alpha=0)
K_raw, D = cam["distortion"]["K"], cam["distortion"]["D"]   # the raw jpg's camera (OpenCV 5-coefficient model)
T_cam_ego = cam["T_cam_ego"]             # camera -> ego, despite the name (the loader uses T_ego_world @ T_cam_ego)
for t_us, T_ego_world in zip(ix["rig"]["timestamps_us"], ix["rig"]["T_ego_world"]):   # ego -> world (UTM), 2 Hz
    c2w = T_ego_world @ T_cam_ego        # camera -> world (UTM) of the frame with this t_us
```

- `ix["frames"][cam]` lists each frame's `t_us` and `img` path; at 2 Hz `t_us` equals `rig["timestamps_us"]`.
- A frame's time is NAVSIM's lidar timestamp. The cameras fire −21 to +23 ms from it (`docs/prepare_data.md`).

### For the renders

- **1080p renders:** `K` is the index `K` above.
- **504×280 renders:** `K` is the model's input camera, the index `K` cropped to the 504:280 aspect and resized
  (`CameraSubsampler`). It is written per context frame to the scene's `<log>.cameras.npz`. For CAM_F0 of this log:
  `fx 345.420, fy 387.470, cx 252.457, cy 145.136`.
- **Pose at frame time `t`** (scene world): `c2w = ego_pose(t) @ T_cam_ego`, minus `world_origin` in translation.
  - At the 2 Hz index frames, `ego_pose(t)` is the index pose.
  - At the 10 Hz in-between frames, it is interpolated: linear in position, slerp in rotation (`tools/viewer.py`
    `Scene.ego_pose`). Those poses are in no file yet.
- **Not written yet:** a per-frame camera file next to the renders (K, size and `c2w` per PNG, in UTM and in
  the scene world). The format depends on what the fixer reads (its own JSON, or COLMAP's `cameras.txt` /
  `images.txt`).

## 3. From scratch on another machine

Checked on 2026-11-27 from a fresh clone of `pmt-finetune` at `e05cb17`, with only the raw OpenScene data linked
in. Both the export and the inference below ran.

### Environment

The README's `uv sync --extra render` (or this machine's conda env `sjy-inurec`, `CLAUDE.md`). The exporter needs
only `cv2`, `numpy`, `torch` and `pandas`. Rendering JIT-compiles gsplat on first use, which needs the CUDA
toolchain in `.claude/skills/inurec-env/SKILL.md`, and `ffmpeg` with libx264 for videos.

```bash
export CUDA_HOME=$CONDA_PREFIX TORCH_EXTENSIONS_DIR=$PWD/.cache/torch_extensions \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Data

OpenScene (NAVSIM) from <https://github.com/OpenDriveLab/OpenScene>. Accept its terms yourself. You need the
trainval log pickles and the trainval camera sensor blobs. Link them under `data/`:

```bash
ln -s /path/to/openscene data/openscene
# the exporter reads sensor_blobs/all_sensor_blobs/<log>/<CAM>/<token>.jpg, a layout made on this machine.
# For a trainval download it is one symlink:
ln -s trainval data/openscene/sensor_blobs/all_sensor_blobs
```

Expected: `data/openscene/navsim_logs/trainval/<log>.pkl` and
`data/openscene/sensor_blobs/all_sensor_blobs/<log>/CAM_F0/<token>.jpg`.

**No labels are needed for inference and rendering.** MoGe-3 depth and Mask2Former masks (`data/moge3`,
`data/m2f`) are training labels. `docs/prepare_data.md` calls depth "required"; that holds for training only.
Without them the pickle has `depth = None` and `seg = None` per frame, and inference runs unchanged.

### Export the index (CPU, seconds per log)

```bash
python tools/index_export/export_index.py --dataset openscene \
  --scenes 2021.09.14.18.43.41_veh-45_00885_00952:0-133 --out-dir data/index/mine
```

- **`--scenes <log>:<first>-<last>`:** frame positions in the NAVSIM log pickle, inclusive; `0-133` is this whole
  log (134 frames at 2 Hz).
- **Output:** `data/index/mine/<log>_<first>-<last>.pkl` with 8 cameras, the ego poses and the annotated boxes
  (`tracks`, needed by `tools/track_edit.py`).
- **`scripts/index/export_openscene.sh`:** exports only our geo-split runs. It also hard-codes `~/miniconda3` and
  `sjy-inurec`; call `export_index.py` directly instead.

**Check (2026-11-27).** The fresh export of this log equals the pickle used for `openscene-2216-video`: same
cameras, rig poses, frame times and image paths, and box poses to 3e-16. The only difference is the label paths,
which are empty without labels.

### Reconstruct, and render the logged trajectory

The fine-tuned checkpoint is not in git. Copy
`checkpoints/training/pmt-openscene-motion-4ep/checkpoints/epoch=epoch=04-step=step=2216.ckpt` from this machine.

```bash
python run_inference.py --model pa-multiview \
  --index-path data/index/mine/2021.09.14.18.43.41_veh-45_00885_00952_0-133.pkl \
  --checkpoint <path to step=2216.ckpt> \
  --n-frames 8 --max-frame-gap-us 500000 --max-chunks 12 --output-dir outputs/predict \
  --merge --n-gaussians 20000000 --include-dynamic --motion-source learned \
  --render-video --save-video-frames --save-video-opacity        # add --video-source-resolution for 1920x1080
```

`outputs/predict/<run id>/ply/<log>/` then holds:
- the scene: `<log>.ply`, `.dynamic.npz`, `.sky.npz` and `.cameras.npz`;
- the 2 Hz CAM_F0 renders: `images/` and `opacity/`, named by frame time;
- `<log>.render.mp4`; add `--source-video` for the real images as video.

`INSTANT_NUREC_RUN_ID=<name>` sets the run id. `--max-chunks 12` covers the first ~47 s; leave it out for the whole
log.

**Check (2026-11-27).** From the fresh clone, a 1-chunk run (`--max-chunks 1`, no merge, `--render-preview`) wrote
the PLY (6.8 M static Gaussians), the dynamic layer (1.3 M), sky, cameras and a preview. The full 12-chunk command
above is the one behind `openscene-2216-video` on this machine; it was not rerun from the fresh clone.

### Edit and render the edited scene

`tools/track_edit.py` re-poses a vehicle onto a new path. `world-x/docs/trafficsim-to-image.md` §3 has the command;
the result is a complete scene folder.

`run_inference.py` writes RGB and opacity only for scenes it reconstructs itself, at the index frames.
`tools/render_frames.py` renders any export, edited or not, through one index camera along the ego's logged
trajectory, at any times:

```bash
L=openscene-2021.09.14.18.43.41_veh-45_00885_00952_0-133
IDX=data/index/geo/openscene/train/2021.09.14.18.43.41_veh-45_00885_00952_0-133.pkl
O=outputs/world-model/2216-video-scene4_slot11
for res in model source; do        # 504x280 -> images/, opacity/;  1920x1080 -> images-1080p/, opacity-1080p/
  python tools/render_frames.py $O/$L.ply --index $IDX --out $O/override --resolution $res \
    --start-us 1631646460700169 --end-us 1631646466000169 --hz 10 --video
  python tools/render_frames.py outputs/predict/openscene-2216-video/ply/$L/$L.ply --index $IDX --out $O/log \
    --resolution $res --start-us 1631646460700169 --end-us 1631646466000169 --hz 10 --video
done
```

These are the commands behind section 1's folders.

| option | meaning |
|---|---|
| (none of `--start-us` / `--end-us`) | renders the camera's index frames inside the scene's time span |
| `--camera` | any index camera (default `CAM_F0`) |
| `--video` | also writes `render<sfx>.mp4` |

The tool checks that the index's ego poses equal the export's (same log, same world) before rendering.

**Check (2026-11-27).** On the unedited scene with the defaults, `--resolution source` gives the 94 frames of
`run_inference`'s `images-1080p/` and `opacity-1080p/`, with the same names. Pixels differing by more than 3/255:
0.008 % on average, 0.04 % in the worst frame.

## 4. Known limits

- **Front camera only.** Only CAM_F0 has been rendered so far, though any camera and pose renders the same way.
- **Artefacts the fixer has to clean up.**
  - The re-posed vehicle trails a smear where its unseen sides show.
  - Holes are left where it used to be.
  - Lighting stays baked from where it actually was.

  Details in `world-x/docs/trafficsim-to-image.md` §5.
- **Not ArtiFixer's own pipeline.** ArtiFixer is built around COLMAP scenes and its own 3DGRUT renders.
  Its input is grayscale opacity PNGs read by index through its `split.json`; matching our files to that layout
  (names, folders, camera file) is still to do.
