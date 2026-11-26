# Editing dynamic objects by their cuboids

Notes of 2026-11-25, on whether a reconstructed and merged OpenScene log (`run_inference.py --index-path
… --merge --include-dynamic`, motion source `tracks`) lets one move a dynamic object by moving its box.
They record why the export alone cannot do it and what it would take. **Built since (2026-11-26):** the offline
tool of Level 1 (`tools/track_edit.py`, below) and its interactive control in `tools/viewer.py --index`. Not built:
Level 2.

## Why moving a box does not move the Gaussians today

- **Tracks are used once, at reconstruction.** `KelvinInferenceModel._compute_motion`
  (`instant_nurec/model/inference.py`) associates each MOVABLE Gaussian with a DYNAMIC track
  (`associate_points_with_cuboid_tracks`), and `warp_points_with_cuboid_tracks`
  (`instant_nurec/utils/motion.py`) places it at the previous and next frame times with that track's poses.
  The track is then discarded.
- **The export has no track id.** `KelvinDynamicLayer` and `.dynamic.npz` (`predict/export_dynamic.py`) hold, per
  Gaussian, 3 world-space keyframe positions and timestamps (prev / source / next), rotation, scale, rgb,
  max density and the track's time bounds. Nothing links a Gaussian back to its box.
- **Editing the boxes before reconstruction does not work either.** The middle keyframe is where the model saw
  the surface; only the outer two follow the box, so the object would jump between frames.
- **The dynamic layer is a stack of snapshots, not one object.** Every context frame of every chunk adds its own
  Gaussians for an object, and each is visible only about one frame gap (±0.5 s at 2 Hz) around its own frame;
  the falloff (`FALLOFF_GAUSSIAN_EXP = 10`) hides it outside. The log
  `2021.09.14.18.43.41_veh-45_00885_00952_0-133` (12 chunks, `track_dynamic_any_chunk` on) has 3.44 M dynamic
  Gaussians for 67 tracks.
- **Rotations do not follow the box.** `KelvinDynamicLayer.interpolate` moves positions and keeps each
  Gaussian's orientation, so a turning car's Gaussians do not turn with it.
- **Parked vehicles are in the static layer.** A track that is never DYNAMIC (travel ≤ 1.5 m in every chunk's
  window) keeps its Gaussians in the static layer, so neither level below covers it without also associating
  the static MOVABLE Gaussians.

## Level 1: edit the trajectory, keep the snapshots

- At export, store per dynamic Gaussian its track id and its position and orientation in the box frame at its
  source time: `x_local = T_track(t_source)^-1 x_source`.
- An edit replaces the track's poses; the 3 keyframes become `T_edited(t_k) x_local` and the rotations are
  turned by the same box rotation.
- Works for edits that keep each object's timing: a lane shift, a heading change, deleting an object.
  A speed change half works: at time t you still see the snapshot taken at t, at its new pose.
- Cannot put an object where it was never observed at that time, since no snapshot exists for that time.
- The ids need no re-inference. They can be recovered offline from the existing `.dynamic.npz`, the index
  pickle's boxes and `world_origin` in `.cameras.npz` (dataset world = PLY + `world_origin`), by running
  `associate_points_with_cuboid_tracks` on each Gaussian's source position at its source timestamp. Storing the
  id at export is exact and cleaner.
- Cheapest first step: an offline tool over the existing export, plus a box-editing control in `tools/viewer.py`,
  which also shows how clean the association is.

### Built: `tools/track_edit.py` (2026-11-26)

- `python tools/track_edit.py <scene.ply> --index <log.pkl>` associates every dynamic Gaussian with an index box at
  its source time (both passes of `associate_points_with_cuboid_tracks`, padding 1 m; boxes over the whole log, with
  the dataset's self-box skip and one-step extrapolation) and lists the tracks with their Gaussian counts.
- `--edits edits.json --out <dir>` writes an edited `.dynamic.npz` plus links to the PLY, sky and cameras, so
  `tools/viewer.py <dir>/<scene>.ply` shows it. Per track: `{"delete": true}` or `{"offset_m": [dx, dy, dz],
  "yaw_deg": a}` in the box frame. Only the edited tracks' Gaussians change: keyframes `T(t_k) E T(t_source)^-1 x`,
  rotation turned by the box's rotation change at the source time.
- Checked on `openscene-1500-video-anychunk` (12 chunks, motion source `tracks`, `track_dynamic_any_chunk`):
  - all 3,441,058 dynamic Gaussians associate, with the same 67 tracks;
  - re-posing every track with the identity edit reproduces the stored prev / next keyframes (p99 0.00 cm; < 0.005 %
    differ, up to 5 m, where a chunk's window had extrapolated the box);
  - with a 3.5 m lateral shift (`a35e139b`, a bus) and a 90° yaw (`f9268a9d`), the Gaussians visible at a time t
    (falloff > 0.5) sit exactly at the edited box-frame positions (p99 0.00 cm), the unedited ones are bit-identical,
    and a deleted pedestrian (`beb4d643`) loses all its 192,936 Gaussians. Front-camera render: the bus moves one lane
    left; the parked sedan behind it (static layer) stays.
- On a learned-motion export (`openscene-2216-video`) 97.0 % of the 10.7 M dynamic Gaussians associate (168 tracks,
  parked cars included, since learned motion sends every MOVABLE Gaussian to the dynamic layer); an edit replaces the
  learned motion of the edited tracks with the box's.
- Seen in the renders: where a moved object had hidden the road the whole time, the static layer has a hole (grey
  patch), as expected below.

### Built: interactive edits in `tools/viewer.py --index <log.pkl>` (2026-11-26)

- "edit objects" draws the associated tracks' boxes at the time slider (wireframe; orange = edited, red = deleted,
  yellow = selected). Clicking a box (or picking it from the list) puts a gizmo on it; dragging the gizmo, or setting
  offset / yaw / delete, re-poses that track's Gaussians in the rendered dynamic layer (roll and pitch of the gizmo
  are dropped). "save edits" writes `edits.json` and the edited scene, as `track_edit.py --edits` does.
- Each track's box-frame positions and keyframe box poses are cached on its first edit, so a drag only composes
  transforms: 6 ms per update for the 714 k Gaussians of the largest track of `openscene-1500-video-anychunk`, 8 ms
  per time step for its 67 boxes (H200).
- Checked without a browser, on that export, by driving the editor's handlers: the three edits above (the yaw through
  a simulated gizmo drag) render identically to the `track_edit.py` output (max 1/255), the saved `.dynamic.npz`
  equals it exactly, and a reset restores the dynamic layer exactly. Not yet tried in a browser.

### Built: a new path from a traffic simulator (2026-11-26)

- `track_edit.py` takes `{"<track id>": {"path_csv": "<file>", "until_contact": true}}`: a CSV of `track_token,
  timestamp, x, y, z, yaw` rows in the index's world (nuPlan `lidar_box` style) plus `after_contact`. The track keeps
  its logged boxes before the first row and follows the CSV after it; its Gaussians observed after the last row are
  dropped, since the logged object is elsewhere by then. Keyframes are `T_new(t_k) T_logged(t_source)^-1 x`.
- First use: `outputs/traffic/nuplan_adv_handover/scene4_slot11` (an adversarial handover; vehicle `5f64ff7f` of log
  `2021.09.14.18.43.41_veh-45_00885_00952`, new path from t0 = 1631646461700169 to first contact at t0 + 4.4 s) on
  `openscene-2216-video`. The override's logged columns equal the index boxes exactly (same UTM world, same yaw
  convention); its 2 Hz ego-frame table agrees with the 20 Hz one through the index rig poses (0.000 m).
  - 138,867 dynamic Gaussians of that track are re-posed, the 110,952 observed after contact dropped, all other
    Gaussians unchanged; at each frame time the track's visible Gaussians sit in the new box where they sat in the
    logged box when observed (p99 ≤ 0.03 cm). Between the 2 Hz keyframes a snapshot moves on the chord of its
    keyframe poses (22 cm behind the box at t0 + 4.2 s, next to the path's end).
  - `outputs/world-model/2216-video-scene4_slot11/compare_log_vs_override.mp4`: front camera and overhead, log vs
    override, t0 − 1 s to contact at 10 Hz. The car leaves its lane, turns about 90° and cuts in front of the ego,
    seen from its observed front; it trails a smear where the turn exposes sides that were never observed (its
    snapshots were taken while it drove straight), as Level 1 predicts.

## Level 2: one reusable asset per object (what simulation needs)

- Gather a track's Gaussians from every chunk into its box frame, de-duplicate them (as the static `voxelize`
  merge does), turn the falloff off, and render the asset at any pose and any time: insert, duplicate or move
  objects freely.
- Open problems:
  - snapshots of the same surface do not line up exactly, so a merge can leave doubled or blurred surfaces;
  - only the sides the cameras saw exist;
  - the wide second association pass (6× box padding) pulls road and neighbouring-object Gaussians into the box,
    so the asset needs cleaning;
  - shading is baked in from where the object actually was;
  - deleting or moving an object leaves a hole in the static road wherever it blocked the view the whole time.

This is the part of the simulation export (`.claude/rules/research-goals.md`) that does not exist yet.
