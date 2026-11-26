# Editing dynamic objects by their cuboids

Notes of 2026-11-25, on whether a reconstructed and merged OpenScene log (`run_inference.py --index-path
… --merge --include-dynamic`, motion source `tracks`) lets one move a dynamic object by moving its box.
**Nothing below is built.** It records why the current export cannot do it and what it would take.

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
