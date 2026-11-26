"""Move, turn or delete dynamic objects of an exported scene by editing their cuboid tracks (Level 1 of
``docs/world-model.md``), with no re-inference.

Each dynamic Gaussian is associated with an index box at its source (observation) time, as
``KelvinInferenceModel._compute_motion`` does (``associate_points_with_cuboid_tracks``, both passes), and expressed
in that box's frame. An edit right-multiplies the track's poses by a box-frame transform; the Gaussians of an edited
track get their 3 keyframes re-posed from the edited track (``T'(t_k) T(t_source)^-1 x``) and their rotation turned
with the box. Gaussians of unedited tracks, and unassociated ones, are written unchanged.

    python tools/track_edit.py <scene.ply> --index <log.pkl>                    # association report, track list
    python tools/track_edit.py <scene.ply> --index <log.pkl> --edits edits.json --out <dir>

``edits.json`` maps track ids (from the report) to ``{"delete": true}``, ``{"offset_m": [dx, dy, dz],
"yaw_deg": a}`` in the box frame (x along the length, z up; the yaw turns the object about its centre), or a new path
``{"path_csv": "<file>", "until_contact": true}``: a CSV with ``track_token, timestamp, x, y, z, yaw`` rows in the
index's world (nuPlan ``lidar_box`` style, e.g. a traffic simulator's override) and ``after_contact``, which
``until_contact`` uses to stop at the first contact. The track follows its logged boxes up to the CSV's first row,
then the CSV; its Gaussians observed after the last row are dropped (the logged object is elsewhere by then).
``<dir>`` gets the edited ``.dynamic.npz`` and links to the scene's PLY, sky and cameras, so
``tools/viewer.py <dir>/<scene>.ply`` shows the edit. Needs an ``--index-path`` export (``.cameras.npz`` holds
``world_origin``) and an index with ``tracks``. Parked objects in the static layer are not covered (see the doc).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle

from pathlib import Path

import numpy as np
import torch

from instant_nurec.datasets.tracks import CuboidTracks, TrackFlags
from instant_nurec.utils import se3 as lt
from instant_nurec.utils.geometry import quat_mult_xyzw, tquat_to_se3_matrix
from instant_nurec.utils.motion import associate_points_with_cuboid_tracks, cuboid_track_time_bounds_us

SELF_BOX_RIG_DIST_M = 3.0  # PMTCuboidTracksConfig.track_min_centroid_rig_dist_m: the ego's own box


def load_tracks(index: dict, world_origin: np.ndarray, device: torch.device) -> tuple[CuboidTracks, list[str]]:
    """Every index track over the whole log, in the export's world (index world - ``world_origin``), with the
    dataset's self-box skip and one-step extrapolation at each end (``PMTIndexDataset._track_window``)."""
    rig_t = np.asarray(index["rig"]["timestamps_us"])
    rig_T = np.asarray(index["rig"]["T_ego_world"], np.float64)
    ids, poses, stamps, dims, labels = [], [], [], [], []
    for track_id, track in index["tracks"].items():
        P = np.array(track["T_box_world"], np.float64)
        t = np.asarray(track["timestamps_us"], np.int64)
        rig = rig_T[np.abs(rig_t[None] - t[:, None]).argmin(1)]
        keep = np.linalg.norm(P[:, :3, 3] - rig[:, :3, 3], axis=1) >= SELF_BOX_RIG_DIST_M
        P, t = P[keep], t[keep]
        if len(t) <= 1:
            continue
        P[:, :3, 3] -= world_origin
        pre, post = (P[0] @ np.linalg.inv(P[1])) @ P[0], (P[-1] @ np.linalg.inv(P[-2])) @ P[-1]
        ids.append(str(track_id))
        poses.append(np.concatenate([pre[None], P, post[None]]).astype(np.float32))
        stamps.append(np.concatenate([[t[0] - (t[1] - t[0])], t, [t[-1] + (t[-1] - t[-2])]]).astype(np.int64))
        dims.append(np.asarray(track["dims"], np.float32))
        labels.append(str(track["label"]))
    tracks = CuboidTracks.Factory.from_numpy(ids, poses, stamps, [TrackFlags.NONE] * len(ids), cuboids_dims=dims,
                                             device=device)
    return tracks, labels


def track_poses(tracks: CuboidTracks, t_us: torch.Tensor, tracks_idx: torch.Tensor) -> lt.SE3:
    """Box poses at ``t_us``, held at the track's first / last pose outside its span (as the warp does)."""
    bounds = cuboid_track_time_bounds_us(tracks, tracks_idx)
    return tracks.interpolate_tracks_poses(t_us.clamp(min=bounds[:, 0], max=bounds[:, 1]), tracks_idx)


def box_edit(edit: dict, device: torch.device) -> lt.SE3:
    """``{"offset_m": [dx, dy, dz], "yaw_deg": a}`` -> the box-frame transform (tquat, xyzw)."""
    half = math.radians(float(edit.get("yaw_deg", 0.0))) / 2
    offset = [float(v) for v in edit.get("offset_m", (0.0, 0.0, 0.0))]
    return lt.SE3(torch.tensor([*offset, 0.0, 0.0, math.sin(half), math.cos(half)], device=device))


def associate(positions: torch.Tensor, source_us: torch.Tensor, tracks: CuboidTracks,
              padding: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> torch.Tensor:
    """Each dynamic Gaussian's track index (-1: none) from its source position and time, both passes of
    ``associate_points_with_cuboid_tracks`` with every Gaussian MOVABLE (the dynamic layer is)."""
    return associate_points_with_cuboid_tracks(
        points=positions, points_timestamps_us=source_us[:, None],
        points_dynamic_mask=torch.ones(len(positions), 1, dtype=torch.bool, device=positions.device),
        cuboid_tracks=tracks, cuboids_dims_padding=torch.tensor(padding, device=positions.device),
    ).squeeze(-1).long()


def repose_basis(positions: torch.Tensor, stamps: torch.Tensor, tracks: CuboidTracks,
                 track: int) -> tuple[torch.Tensor, lt.SE3, lt.SE3]:
    """What re-posing one track's unedited Gaussians (keyframe positions (n, T, 3), their times (n, T)) needs, for
    any edit: their box-frame positions at the source time, the box poses at every keyframe time (T, n) and at the
    source time."""
    idx = torch.full((len(positions),), track, dtype=tracks.tracks_packinfo.dtype, device=positions.device)
    source = track_poses(tracks, stamps[:, 1], idx)
    keyframes = lt.SE3(torch.stack([track_poses(tracks, stamps[:, k], idx).data for k in range(stamps.shape[1])]))
    return source.inv() * positions[:, 1], keyframes, source


def repose_apply(basis: tuple[torch.Tensor, lt.SE3, lt.SE3], rotations: torch.Tensor,
                 edit: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """``repose_basis`` and the Gaussians' wxyz rotations (n, 4) -> their keyframes on the edited track,
    ``T(t_k) E T(t_source)^-1 x`` (n, T, 3), and their rotations turned with the box."""
    local, keyframes, source = basis
    moved = lt.SE3(box_edit(edit, local.device).data.expand(len(local), 7))
    out = torch.stack([(keyframes[k] * moved) * local for k in range(keyframes.shape[0])], 1)
    turn = (source * moved * source.inv()).data[:, 3:]  # the box's rotation change at the source time (xyzw)
    return out, quat_mult_xyzw(turn, rotations[:, [1, 2, 3, 0]])[:, [3, 0, 1, 2]]


def repose(positions: torch.Tensor, stamps: torch.Tensor, rotations: torch.Tensor, tracks: CuboidTracks,
           track: int, edit: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """One track's unedited Gaussians (keyframe positions (n, T, 3), their times (n, T), wxyz rotations (n, 4))
    -> their keyframes on the edited track and their rotations turned with the box."""
    return repose_apply(repose_basis(positions, stamps, tracks, track), rotations, edit)


def load_path(csv_path: Path, track_id: str, tracks: CuboidTracks, world_origin: np.ndarray,
              until_contact: bool) -> dict:
    """A new path for track ``track_id``: its logged poses (``tracks``, export world) before the CSV's first row, then
    the CSV's rows (index world, yaw about z; with ``until_contact`` the rows with ``after_contact`` false) ->
    ``{"t_us", "T" (n, 4, 4) export world, "end_us"}``."""
    rows = [r for r in csv.DictReader(open(csv_path)) if r["track_token"] == track_id]
    if until_contact:
        rows = [r for r in rows if r["after_contact"].strip().lower() == "false"]
    if len(rows) < 2:
        raise SystemExit(f"{csv_path}: fewer than 2 rows for {track_id}")
    t_new = np.array([int(r["timestamp"]) for r in rows], np.int64)
    T_new = np.tile(np.eye(4), (len(rows), 1, 1))
    for i, r in enumerate(rows):
        c, s = math.cos(float(r["yaw"])), math.sin(float(r["yaw"]))
        T_new[i, :2, :2] = [[c, -s], [s, c]]
        T_new[i, :3, 3] = np.array([float(r["x"]), float(r["y"]), float(r["z"])]) - world_origin
    k = tracks.tracks_id.index(track_id)
    start, n = (int(v) for v in tracks.tracks_packinfo[k])
    t_log = tracks.tracks_timestamps_us[start:start + n].cpu().numpy()
    T_log = tquat_to_se3_matrix(tracks.tracks_poses.data[start:start + n].cpu(), unbatch=False).numpy()
    before = t_log < t_new[0]
    return {"t_us": np.concatenate([t_log[before], t_new]), "T": np.concatenate([T_log[before], T_new]),
            "end_us": int(t_new[-1])}


def repose_path(positions: torch.Tensor, stamps: torch.Tensor, rotations: torch.Tensor, tracks: CuboidTracks,
                track: int, path: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """One track's unedited Gaussians -> their keyframes on a new path, ``T_new(t_k) T(t_source)^-1 x``, and their
    rotations turned by the path's rotation against the logged box at the source time."""
    device = positions.device
    new = CuboidTracks.Factory.from_numpy(
        ["path"], [path["T"].astype(np.float32)], [path["t_us"]], [TrackFlags.NONE],
        cuboids_dims=[tracks.cuboids_dims[track].cpu().numpy()], device=device)
    idx = torch.full((len(positions),), track, dtype=tracks.tracks_packinfo.dtype, device=device)
    zero = torch.zeros_like(idx)
    source = track_poses(tracks, stamps[:, 1], idx)
    local = source.inv() * positions[:, 1]
    out = torch.stack([track_poses(new, stamps[:, k], zero) * local for k in range(stamps.shape[1])], 1)
    turn = (track_poses(new, stamps[:, 1], zero) * source.inv()).data[:, 3:]
    return out, quat_mult_xyzw(turn, rotations[:, [1, 2, 3, 0]])[:, [3, 0, 1, 2]]


def edit_dynamic(dynamic: dict, tracks: CuboidTracks, tracks_idx: torch.Tensor, edits: dict[int, dict]) -> dict:
    """The ``.dynamic.npz`` payload with every Gaussian of an edited track re-posed or deleted."""
    device = tracks_idx.device
    positions = torch.tensor(dynamic["layer_0_keyframe_positions"], device=device)
    stamps = torch.tensor(dynamic["layer_0_keyframe_timestamps_us"], device=device)
    rotations = torch.tensor(dynamic["layer_0_rotations"], device=device)  # wxyz
    keep = torch.ones(len(positions), dtype=torch.bool, device=device)
    for track, edit in edits.items():
        sel = torch.where(tracks_idx == track)[0]
        if edit.get("delete", False):
            keep[sel] = False
        elif "path" in edit:  # resolved by load_path
            positions[sel], rotations[sel] = repose_path(positions[sel], stamps[sel], rotations[sel], tracks, track,
                                                         edit["path"])
            keep[sel[stamps[sel, 1] > edit["path"]["end_us"]]] = False
        else:
            positions[sel], rotations[sel] = repose(positions[sel], stamps[sel], rotations[sel], tracks, track, edit)
    payload = dict(dynamic)
    mask = keep.cpu().numpy()
    payload["layer_0_keyframe_positions"] = positions.cpu().numpy()
    payload["layer_0_rotations"] = rotations.cpu().numpy()
    for name, value in dynamic.items():
        if name.startswith("layer_0_") and np.ndim(value) > 0 and len(value) == len(mask):
            payload[name] = payload[name][mask]
    return payload


def write_edited(ply: Path, payload: dict, out: Path) -> Path:
    """``out`` gets the edited ``.dynamic.npz`` and links to the scene's PLY, sky and cameras."""
    out.mkdir(parents=True, exist_ok=True)
    stem = ply.with_suffix("").name
    path = out / f"{stem}.dynamic.npz"
    np.savez_compressed(path, **payload)
    for suffix in (".ply", ".sky.npz", ".sky.png", ".cameras.npz"):  # what tools/viewer.py reads; not the renders
        source, target = ply.parent / f"{stem}{suffix}", out / f"{stem}{suffix}"
        if source.is_file() and not target.exists():
            target.symlink_to(source.resolve())
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply", type=Path, help="the export's .ply; its .dynamic.npz and .cameras.npz sit beside it")
    parser.add_argument("--index", type=Path, required=True, help="the log's index pickle (with 'tracks')")
    parser.add_argument("--padding", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("X", "Y", "Z"),
                        help="box padding of the first association pass, as run_inference.py --cuboid-padding")
    parser.add_argument("--edits", type=Path, help="JSON {track id: {delete | offset_m, yaw_deg | path_csv}}")
    parser.add_argument("--out", type=Path, help="directory for the edited scene (with --edits)")
    args = parser.parse_args()
    if (args.edits is None) != (args.out is None):
        parser.error("--edits and --out go together")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dynamic = dict(np.load(args.ply.with_suffix(".dynamic.npz")))
    if int(dynamic["n_layers"]) != 1:
        raise SystemExit(f"expected 1 dynamic layer, got {int(dynamic['n_layers'])}")
    world_origin = np.load(args.ply.with_suffix(".cameras.npz"))["world_origin"].astype(np.float64)
    with open(args.index, "rb") as f:
        index = pickle.load(f)
    tracks, labels = load_tracks(index, world_origin, device)

    positions = torch.as_tensor(dynamic["layer_0_keyframe_positions"][:, 1], device=device)
    source_us = torch.as_tensor(dynamic["layer_0_keyframe_timestamps_us"][:, 1], device=device)
    tracks_idx = associate(positions, source_us, tracks, tuple(args.padding))

    counts = torch.bincount(tracks_idx[tracks_idx >= 0], minlength=tracks.n_tracks).cpu().numpy()
    n = len(tracks_idx)
    print(f"{n:,} dynamic Gaussians, {tracks.n_tracks} tracks: {int(counts.sum()):,} ({counts.sum() / max(n, 1):.1%}) "
          f"associated with {int((counts > 0).sum())} tracks, {n - int(counts.sum()):,} with none")
    span = dynamic["layer_0_keyframe_timestamps_us"][:, 1]
    print(f"{'track':<18} {'label':<16} {'gaussians':>10}  {'seen [s]':>15}")
    t0 = int(span.min())
    for k in np.argsort(-counts):
        if counts[k] == 0:
            break
        seen = span[(tracks_idx == int(k)).cpu().numpy()]
        print(f"{tracks.tracks_id[k]:<18} {labels[k]:<16} {counts[k]:>10,}  {(seen.min() - t0) / 1e6:6.1f} – "
              f"{(seen.max() - t0) / 1e6:5.1f}")

    if args.edits is None:
        return
    requested = json.loads(args.edits.read_text())
    missing = sorted(set(requested) - set(tracks.tracks_id))
    if missing:
        raise SystemExit(f"unknown track ids: {missing}")
    edits = {tracks.tracks_id.index(k): v for k, v in requested.items()}
    for k, edit in requested.items():
        if "path_csv" in edit:
            edit = edits[tracks.tracks_id.index(k)] = dict(edit)
            edit["path"] = load_path(Path(edit["path_csv"]), k, tracks, world_origin, edit.get("until_contact", False))
            print(f"{k}: new path from {edit['path_csv']}, {len(edit['path']['t_us'])} poses, ends "
                  f"{(edit['path']['end_us'] - int(span.min())) / 1e6:.2f} s into the scene")
    payload = edit_dynamic(dynamic, tracks, tracks_idx, edits)
    path = write_edited(args.ply, payload, args.out)
    for k, edit in requested.items():
        print(f"edited {k}: {edit} ({counts[tracks.tracks_id.index(k)]:,} Gaussians)")
    (args.out / "edits.json").write_text(json.dumps(requested, indent=1))
    print(f"Wrote {path} ({len(payload['layer_0_rotations']):,} dynamic Gaussians)")


if __name__ == "__main__":
    main()
