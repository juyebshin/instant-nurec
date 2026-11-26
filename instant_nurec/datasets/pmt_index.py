# SPDX-License-Identifier: Apache-2.0
"""Training dataset for non-NCore logs (AV2 / nuScenes / Waymo) from per-log index pickles.

The index is written by ``tools/index_export/export_index.py`` into ``data/index/`` (until 2026-11-25 by
PseudoMapTrainer ``RoGS/scripts/inurec_finetune_index.py``; same output)::

    sequence_id: str
    cameras: {name: {"K": (3, 3), "wh": (W, H), "T_cam_ego": (4, 4)}}   # OpenCV camera frame
    rig: {"timestamps_us": (N,) int64, "T_ego_world": (N, 4, 4) float64}  # one pose per image timestamp
    frames: {name: [{"t_us": int, "img": str, "seg": str | None, "depth": str | None}, ...]}  # sorted by t_us
    semantic_lut: (256,) uint8   # segmentation id -> KelvinSemanticClass, 255 = ignore

The exporter's LUT is 8-class; ``semantic_remap`` in the config folds it for a smaller head.

A camera may also carry (Waymo does):

    distortion: {"K", "D" (OpenCV k1 k2 p1 p2 k3), "wh"}  # image / seg files are raw; undistorted onto K / wh
    depth_grid: {"K", "wh"}  # the depth files' own pinhole grid of the same camera; resampled onto K / wh
    seg_grid: {"K", "wh"}    # masks on such a grid (OpenScene: DVGT's MoGe image), resampled, not undistorted

Depth files are DVGT/MoGe log-encoded uint16 PNGs of camera z-depth (``near`` / ``far`` in
the PNG text; 0 = unknown, 65535 = inf); they are converted to ray distance here.

An index may also carry annotated boxes (OpenScene does)::

    tracks: {id: {"label": str, "dims": (3,) l/w/h along box x/y/z, "timestamps_us": (n,) int64,
                  "T_box_world": (n, 4, 4) float64}}   # box centre, x along the length

The batch mirrors ``NCoreInstantNuRecDataset``: rig = ego, world = the first camera's
first context frame, cameras global-shutter pinhole. With ``config.frame_batch_sampler`` set (predict,
``run_inference.py --index-path``) the context chunks cover each log in order, as the NCore predict loader's, and
the dataset world is moved to the log's first rig pose (``meta["world_origin"]`` = that pose's translation). Cuboid tracks only with ``config.cuboid_tracks``
set (built from ``tracks`` as ``NCoreInstantNuRecDataset._compute_cuboid_tracks`` does); otherwise none, and
moving objects go to the learned-motion dynamic layer.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import pickle
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import ncore.data

from instant_nurec.config_schema.dataset import PMTIndexDatasetConfig
from instant_nurec.datasets.instantnurec_base import BaseInstantNuRecIndexableDataset, CameraSubsampler
from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
from instant_nurec.datasets.samplers import AdaptiveSequentialFrameBatchSampler, count_sequential_chunks
from instant_nurec.datasets.tracks import CuboidTracks, CuboidTracksDataPack, TrackFlags
from instant_nurec.primitives.kelvin_primitive import KelvinSemanticClass as C
from instant_nurec.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    InstantNuRecDataBatch,
)
from instant_nurec.utils.geometry import se3_matrix_inverse
from instant_nurec.utils.types import FrameConversion, HalfClosedInterval, RayFlags, RigTrajectories

logger = logging.getLogger(__name__)

ROAD_SURFACE = (int(C.ROAD), int(C.LANE), int(C.CROSSWALK))
IGNORE = 255


def read_log_depth(path: str) -> np.ndarray:
    """DVGT ``tools/data_process/utils/io.py`` decode; invalid -> 0."""
    image = Image.open(path)
    near, far = float(image.info["near"]), float(image.info["far"])
    raw = np.asarray(image)
    depth = near ** (1 - (raw.astype(np.float32) - 1) / 65533) * far ** ((raw.astype(np.float32) - 1) / 65533)
    depth[(raw == 0) | (raw == 65535)] = 0.0
    depth[depth < 1.0] = 0.0  # DVGT's own lower bound
    return depth


def z_to_ray_distance(depth: np.ndarray, K: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    """Camera z-depth -> distance along the unit ray (K given at resolution ``wh``)."""
    h, w = depth.shape
    sx, sy = w / wh[0], h / wh[1]
    fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
    x = (np.arange(w, dtype=np.float32) - cx) / fx
    y = (np.arange(h, dtype=np.float32) - cy) / fy
    return depth * np.sqrt(x[None, :] ** 2 + y[:, None] ** 2 + 1.0)


def pinhole_parameters(K: np.ndarray, wh: tuple[int, int]) -> ncore.data.OpenCVPinholeCameraModelParameters:
    # NCore stores the OpenCV principal point (integer pixel centres) and adds 0.5 itself.
    return ncore.data.OpenCVPinholeCameraModelParameters(
        resolution=np.array(wh, dtype=np.uint64),
        shutter_type=ncore.data.ShutterType.GLOBAL,
        principal_point=np.array([K[0, 2], K[1, 2]], dtype=np.float32),
        focal_length=np.array([K[0, 0], K[1, 1]], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )


@functools.lru_cache(maxsize=10)  # one sample's cameras; ~15 MB per 1920x1280 map pair
def _undistort_maps(K_raw: tuple, D: tuple, K: tuple, wh: tuple[int, int]):
    as_K = lambda k: np.array(k, np.float64).reshape(3, 3)  # noqa: E731
    return cv2.initUndistortRectifyMap(as_K(K_raw), np.array(D, np.float64), None, as_K(K), wh, cv2.CV_16SC2)


def undistort_image(image: np.ndarray, distortion: dict, K: np.ndarray, wh: tuple[int, int],
                    interpolation: int = cv2.INTER_LINEAR, border_value: int = 0) -> np.ndarray:
    """Raw image of the distorted camera ``distortion`` -> the pinhole grid ``K`` / ``wh``.

    Use ``cv2.INTER_NEAREST`` for label maps. An image at another size than the raw ``wh`` (e.g. a
    downscaled mask) is first resized to it with the same interpolation."""
    raw_wh = tuple(distortion["wh"])
    if image.shape[1::-1] != raw_wh:
        image = cv2.resize(image, raw_wh, interpolation=interpolation)
    maps = _undistort_maps(*(tuple(np.ravel(x).tolist()) for x in (distortion["K"], distortion["D"], K)), tuple(wh))
    return cv2.remap(image, *maps, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def resample_depth_grid(depth: np.ndarray, depth_K: np.ndarray, K: np.ndarray, wh: tuple[int, int],
                        border_value: int = 0) -> np.ndarray:
    """z-depth (or a label map) on one pinhole grid -> another grid of the same camera (nearest; ``border_value``
    outside)."""
    ax, ay = depth_K[0, 0] / K[0, 0], depth_K[1, 1] / K[1, 1]
    target_to_source = np.array([[ax, 0, depth_K[0, 2] - ax * K[0, 2]], [0, ay, depth_K[1, 2] - ay * K[1, 2]]])
    return cv2.warpAffine(depth, target_to_source, tuple(wh), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


class PMTIndexDataset(BaseInstantNuRecIndexableDataset[InstantNuRecDataBatch]):
    def __init__(self, config: PMTIndexDatasetConfig, global_seed: int, retry_on_error: bool = False):
        self.config = config
        self._global_seed = global_seed
        self._retry_on_error = retry_on_error
        self._rng_epoch = -1
        self._epoch = 0
        self._remap = np.arange(256, dtype=np.uint8)  # composed onto each index's semantic_lut
        for old, new in config.semantic_remap.items():
            self._remap[old] = new
        if config.track_dynamic_any_chunk and (config.frame_batch_sampler is None or config.cuboid_tracks is None):
            raise ValueError("track_dynamic_any_chunk needs cuboid_tracks and the predict chunks (frame_batch_sampler)")
        self._any_chunk_dynamic: dict[str, set[str]] = {}  # index path -> track ids DYNAMIC in any chunk
        self.indexes = []
        paths = [str(q) for p in config.index_paths
                 for q in (sorted(Path(p).glob("*.pkl")) if Path(p).is_dir() else [Path(p)])]
        for path in paths:
            with open(path, "rb") as f:
                index = pickle.load(f)
            missing = set(config.camera_ids) - set(index["cameras"])
            if missing:
                raise ValueError(f"{path}: cameras {sorted(missing)} not in index ({sorted(index['cameras'])})")
            index["_path"] = path
            if config.frame_batch_sampler is not None:  # predict: log-local world (OpenScene's is UTM, ~4e6 m)
                origin = np.asarray(index["rig"]["T_ego_world"][0][:3, 3], np.float64).copy()
                index["rig"]["T_ego_world"] = np.array(index["rig"]["T_ego_world"], np.float64)
                index["rig"]["T_ego_world"][:, :3, 3] -= origin
                for track in index.get("tracks", {}).values():
                    track["T_box_world"] = np.array(track["T_box_world"], np.float64)
                    track["T_box_world"][:, :3, 3] -= origin
                index["_world_origin"] = tuple(float(x) for x in origin)
            self.indexes.append(index)

    @property
    def num_samples_per_sequence(self) -> int:
        sampler = self.config.frame_batch_sampler
        return self.config.n_samples_per_sequence if sampler is None else sampler.n_samples_per_sequence

    def __len__(self) -> int:
        return len(self.indexes) * self.num_samples_per_sequence

    def set_rng_epoch(self, rng_epoch: int) -> None:
        self._rng_epoch = rng_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _get_rng(self, batch_idx: int) -> np.random.Generator:
        digest = hashlib.sha256(f"{self._rng_epoch}_{batch_idx}_{self._global_seed}".encode()).digest()
        return np.random.default_rng(seed=int.from_bytes(digest[:8], "big"))

    # ------------------------------------------------------------------ sampling
    def _context_candidates(self, index, camera: str) -> list[int]:
        frames = index["frames"][camera]
        if self.config.context_requires_depth and self.config.depth:
            return [i for i, f in enumerate(frames) if f["depth"] is not None]
        return list(range(len(frames)))

    def _sample_context(self, index, rng) -> dict[str, list[int]]:
        cfg, cams = self.config, self.config.camera_ids
        ref = self._context_candidates(index, cams[0])
        times = np.array([index["frames"][cams[0]][i]["t_us"] for i in ref], dtype=np.int64)
        chains = []
        for start in range(len(ref)):
            chain = [start]
            while len(chain) < cfg.n_frames_per_sample:
                nxt = np.searchsorted(times, times[chain[-1]] + int(0.9 * cfg.frame_gap_us))
                if nxt >= len(ref):
                    break
                chain.append(int(nxt))
            if len(chain) == cfg.n_frames_per_sample:
                chains.append(chain)
        if not chains:
            raise ValueError(f"{index['_path']}: no {cfg.n_frames_per_sample}-frame context chain")
        chain = chains[int(rng.integers(len(chains)))]
        ref_times = times[chain]
        out = {cams[0]: [ref[i] for i in chain]}
        for cam in cams[1:]:
            cand = self._context_candidates(index, cam)
            t = np.array([index["frames"][cam][i]["t_us"] for i in cand], dtype=np.int64)
            out[cam] = [cand[int(np.abs(t - rt).argmin())] for rt in ref_times]
        return out

    def _sequence_interval(self, index, times: dict[str, np.ndarray]) -> HalfClosedInterval:
        """As the NCore predict loader: the rig-pose span, cut to each context camera's span +- 0.1 s."""
        rig_t = index["rig"]["timestamps_us"]
        start = max([int(rig_t.min())] + [int(t.min()) - 100_000 for t in times.values()])
        end = min([int(rig_t.max())] + [int(t.max()) + 100_000 for t in times.values()])
        return HalfClosedInterval(start, end)

    def _sequential_context(self, index, sample_idx: int) -> dict[str, list[int]]:
        """Chunk ``sample_idx`` of the log (AdaptiveSequentialFrameBatchSampler); {} past the last chunk."""
        cands = {cam: self._context_candidates(index, cam) for cam in self.config.camera_ids}
        times = {cam: np.array([index["frames"][cam][i]["t_us"] for i in c], np.int64) for cam, c in cands.items()}
        sampler = AdaptiveSequentialFrameBatchSampler(self.config.frame_batch_sampler, self.config.n_frames_per_sample)
        picked = sampler.sample_frame_batch(sample_idx, times, [self._sequence_interval(index, times)])
        return {cam: [cands[cam][j] for j in idxs] for cam, idxs in picked.items()}

    def _sample_supervision(self, index, context: dict[str, list[int]], rng) -> dict[str, list[int]]:
        cfg = self.config.supervision_frame_batch
        span = [index["frames"][c][i]["t_us"] for c, idxs in context.items() for i in idxs]
        lo, hi = min(span) - cfg.prepend_timestamps_us, max(span) + cfg.append_timestamps_us
        out = {}
        for cam in self.config.camera_ids:
            cand = np.array([i for i, f in enumerate(index["frames"][cam]) if lo <= f["t_us"] <= hi])
            if cfg.sample_strategy == "random":
                idxs = np.sort(rng.choice(cand, size=cfg.n_frames_per_camera, replace=True)).tolist()
            else:
                bins = np.linspace(0, len(cand), cfg.n_frames_per_camera + 1, dtype=int)
                idxs = [int(cand[a]) if a == b else int(rng.choice(cand[a:b])) for a, b in zip(bins[:-1], bins[1:])]
            if cfg.include_context_frames:
                idxs = sorted(set(idxs) | set(context[cam]))
            out[cam] = [int(i) for i in idxs]
        return out

    # ------------------------------------------------------------------ loading
    def _pose(self, index, t_us: int) -> np.ndarray:
        ts = index["rig"]["timestamps_us"]
        return index["rig"]["T_ego_world"][int(np.abs(ts - t_us).argmin())]

    def _load_labels(self, index, cam: str, frame: dict, subsampler: CameraSubsampler) -> CameraFrameLabels:
        h, w = subsampler.frame_height, subsampler.frame_width
        meta = index["cameras"][cam]
        K, wh = np.asarray(meta["K"], np.float64), tuple(meta["wh"])
        image = cv2.imread(frame["img"])
        if "distortion" in meta:
            image = undistort_image(image, meta["distortion"], K, wh)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        labels = CameraFrameLabels(rgb=torch.from_numpy(subsampler.apply_frame_data(image))[None])
        flags = torch.full((h, w), int(RayFlags.RGB_LABEL), dtype=torch.int32)
        sky_full = None
        if self.config.semantic_segmentation:
            if frame["seg"] is not None:
                classes_full = self._remap[index["semantic_lut"]][cv2.imread(frame["seg"], cv2.IMREAD_UNCHANGED)]
                if "seg_grid" in meta:  # masks predicted on another pinhole grid of the camera
                    classes_full = resample_depth_grid(classes_full, np.asarray(meta["seg_grid"]["K"], np.float64), K,
                                                       wh, IGNORE)
                elif "distortion" in meta:  # masks predicted on the raw, distorted image
                    classes_full = undistort_image(classes_full, meta["distortion"], K, wh, cv2.INTER_NEAREST, IGNORE)
                sky_full = classes_full == int(C.SKY)
                classes = torch.from_numpy(subsampler.apply_frame_data(classes_full))
                flags |= int(RayFlags.VALID_SEMANTIC)
                flags[classes == int(C.SKY)] |= int(RayFlags.SKY_SEMANTIC)
                flags[torch.isin(classes, torch.tensor(ROAD_SURFACE, dtype=classes.dtype))] |= int(
                    RayFlags.ROAD_SEMANTIC
                )
                flags[classes == int(C.MOVABLE)] |= int(RayFlags.VEHICLE_SEMANTIC)
                flags[classes == int(C.EGO)] |= int(RayFlags.INVALID | RayFlags.EGO_SEMANTIC)
            else:
                classes = torch.full((h, w), IGNORE, dtype=torch.uint8)
            labels.semantic_class = classes[None, ..., None]
        if self.config.depth and frame["depth"] is not None:
            depth = read_log_depth(frame["depth"])
            if "depth_grid" in meta:
                depth = resample_depth_grid(depth, np.asarray(meta["depth_grid"]["K"], np.float64), K, wh)
            if sky_full is not None:  # as the NCore loader: no depth on sky
                if sky_full.shape != depth.shape:  # e.g. nuScenes MoGe depth is a downscaled image
                    sky_full = cv2.resize(sky_full.astype(np.uint8), depth.shape[::-1],
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                depth[sky_full] = 0.0
            distance = z_to_ray_distance(depth, K, wh)
            labels.metric_distance = torch.from_numpy(subsampler.apply_depth_data(distance))[None, ..., None]
        labels.flags = flags[None, ..., None]
        return labels

    def _build(self, index, frame_batch: dict[str, list[int]], T_world_ref: np.ndarray, prefix: str,
               subsampler: CameraSubsampler) -> tuple[DataAndRenderingBatch, RigTrajectories]:
        cams = self.config.camera_ids
        seq_id = prefix + index["sequence_id"]
        camera_batches, frame_ts, calibrations, used_t = [], {}, OrderedDict(), []
        unique_frame_idx = 0
        for sensor_idx, cam in enumerate(cams):
            meta = index["cameras"][cam]
            calibrations[cam] = RigTrajectories.CameraCalibration(
                sequence_id=seq_id,
                unique_sensor_idx=sensor_idx,
                T_sensor_rig=torch.as_tensor(np.asarray(meta["T_cam_ego"], np.float32)),
                camera_model_parameters=subsampler.apply_camera_parameters(
                    pinhole_parameters(np.asarray(meta["K"], np.float64), tuple(meta["wh"]))
                ),
            )
            stamps = []
            for i in frame_batch[cam]:
                frame = index["frames"][cam][i]
                stamps.append((frame["t_us"], frame["t_us"]))
                used_t.append(frame["t_us"])
                camera_batches.append(
                    DataBatch.Camera(
                        meta=[FrameMeta(unique_sensor_idx=sensor_idx, unique_frame_idx=unique_frame_idx)],
                        labels=self._load_labels(index, cam, frame, subsampler),
                    )
                )
                unique_frame_idx += 1
            frame_ts[cam] = torch.tensor(stamps, dtype=torch.int64)

        # rig poses over the used span (+1 s), padded at both ends like the NCore loader
        ts, poses = index["rig"]["timestamps_us"], index["rig"]["T_ego_world"]
        keep = (ts >= min(used_t) - 1_000_000) & (ts <= max(used_t) + 1_000_000)
        ts, poses = ts[keep], poses[keep]
        ts = np.concatenate([[ts[0] - 1], ts, [ts[-1] + 1]])
        poses = np.concatenate([poses[:1], poses, poses[-1:]])
        rig = RigTrajectories(
            T_world_base=se3_matrix_inverse(torch.as_tensor(T_world_ref, dtype=torch.float64)),
            world_to_scene=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            rig_trajectories=[
                RigTrajectories.RigTrajectory(
                    sequence_id=seq_id,
                    cameras_frame_timestamps_us=frame_ts,
                    T_rig_worlds=torch.as_tensor(T_world_ref @ poses, dtype=torch.float64),
                    T_rig_world_timestamps_us=torch.as_tensor(ts, dtype=torch.int64),
                )
            ],
            camera_calibrations=calibrations,
        )
        data = DataAndRenderingBatch(data=DataBatch(camera=DataBatch.Camera.collate_fn(camera_batches)))
        return data, rig

    def _track_window(self, index, track, t_min: int, t_max: int,
                      T_world_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        """One track around the context span [t_min, t_max] (window, self-box skip, one-step extrapolation at each
        end) -> box poses in ``T_world_ref``'s frame, their timestamps and the travel that decides DYNAMIC; None if it
        has <= 1 box."""
        params = self.config.cuboid_tracks
        lo, hi = t_min - params.track_extrapolate_timestamps_us, t_max + params.track_extrapolate_timestamps_us
        ts = track["timestamps_us"]
        keep = np.nonzero((ts >= lo) & (ts < hi))[0]
        keep = [k for k in keep if np.linalg.norm(  # skip self-classifications (box centre near the rig)
            (np.linalg.inv(self._pose(index, int(ts[k]))) @ track["T_box_world"][k])[:3, 3]
        ) >= params.track_min_centroid_rig_dist_m]
        if len(keep) <= 1:
            return None
        P, t = T_world_ref @ track["T_box_world"][keep], ts[keep]
        pre, post = (P[0] @ np.linalg.inv(P[1])) @ P[0], (P[-1] @ np.linalg.inv(P[-2])) @ P[-1]
        P = np.concatenate([pre[None], P, post[None]]).astype(np.float32)
        t = np.concatenate([[t[0] - (t[1] - t[0])], t, [t[-1] + (t[-1] - t[-2])]]).astype(np.int64)
        travel = float(np.linalg.norm(P[-1, :3, 3] - P[0, :3, 3])) * (t_max - t_min) / (t[-1] - t[0])
        return P, t, travel

    def _dynamic_in_any_chunk(self, index) -> set[str]:
        """Ids of the tracks that are DYNAMIC (travel rule) in at least one of the log's predict chunks."""
        if index["_path"] not in self._any_chunk_dynamic:
            ids = set()
            for k in range(self.num_samples_per_sequence):
                chunk = self._sequential_context(index, k)
                if not chunk:
                    break
                span = [index["frames"][c][i]["t_us"] for c, idxs in chunk.items() for i in idxs]
                for track_id, track in index["tracks"].items():
                    window = self._track_window(index, track, min(span), max(span), np.eye(4))
                    if window is not None and window[2] > self.config.cuboid_tracks.track_min_travel_distance_m:
                        ids.add(str(track_id))
            self._any_chunk_dynamic[index["_path"]] = ids
        return self._any_chunk_dynamic[index["_path"]]

    def _cuboid_tracks(self, index, context_batch: dict[str, list[int]], T_world_ref: np.ndarray) -> CuboidTracksDataPack:
        """The index's boxes -> cuboid tracks around the context span, following
        ``NCoreInstantNuRecDataset._compute_cuboid_tracks`` (window, self-box skip, one-step extrapolation at each
        end, DYNAMIC = always-moving class or travel over the threshold; with ``track_dynamic_any_chunk``, travel over
        the threshold in any chunk of the log)."""
        params = self.config.cuboid_tracks
        if "tracks" not in index:
            raise ValueError(f"{index['_path']}: cuboid_tracks is set but the index has no 'tracks'")
        span = [index["frames"][c][i]["t_us"] for c, idxs in context_batch.items() for i in idxs]
        t_min, t_max = min(span), max(span)
        any_chunk = self._dynamic_in_any_chunk(index) if self.config.track_dynamic_any_chunk else set()
        ids, poses, stamps, flags, dims = [], [], [], [], []
        for track_id, track in index["tracks"].items():
            window = self._track_window(index, track, t_min, t_max, T_world_ref)
            if window is None:
                continue
            P, t, travel = window
            dynamic = (track["label"] in NCoreInstantNuRecDataset.UNCONDITIONALLY_DYNAMIC_LABELS
                       or travel > params.track_min_travel_distance_m or str(track_id) in any_chunk)
            ids.append(str(track_id))
            poses.append(P)
            stamps.append(t)
            flags.append(TrackFlags.DYNAMIC if dynamic else TrackFlags.NONE)
            dims.append(np.asarray(track["dims"], np.float32))
        tracks = CuboidTracks.Factory.from_numpy(ids, poses, stamps, flags, cuboids_dims=dims, device=torch.device("cpu"))
        return CuboidTracksDataPack(tracks_data=tracks.tracks_data, cuboidtracks_data=tracks.cuboidtracks_data)

    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> InstantNuRecDataBatch:
        index = self.indexes[batch_idx // self.num_samples_per_sequence]
        cams = self.config.camera_ids
        if self.config.frame_batch_sampler is not None:
            context_batch = self._sequential_context(index, batch_idx % self.num_samples_per_sequence)
            if not context_batch:  # past the log's last chunk: 0-sized, concatenated away as in the NCore loader
                return InstantNuRecDataBatch(context=[], cuboid_tracks=[], context_rig=[], meta=[])
        else:
            context_batch = self._sample_context(index, rng)

        # world = the first camera at its first context frame (NCore: T_world_ref is world -> ref sensor)
        first = index["frames"][cams[0]][context_batch[cams[0]][0]]
        T_cam_world = self._pose(index, first["t_us"]) @ np.asarray(index["cameras"][cams[0]]["T_cam_ego"])
        T_world_ref = np.linalg.inv(T_cam_world)

        subsampler = CameraSubsampler(
            frame_width=self.config.camera_subsampler.frame_width,
            frame_height=self.config.camera_subsampler.frame_height,
        )
        context, context_rig = self._build(index, context_batch, T_world_ref, "context-", subsampler)

        supervision = supervision_rig = None
        sup_cfg = self.config.supervision_frame_batch
        if sup_cfg.n_frames_per_camera > 0:
            sup_subsampler = CameraSubsampler(
                frame_width=sup_cfg.camera_subsampler.frame_width,
                frame_height=sup_cfg.camera_subsampler.frame_height,
            )
            supervision, supervision_rig = self._build(
                index, self._sample_supervision(index, context_batch, rng), T_world_ref, "supervision-",
                sup_subsampler,
            )
        return InstantNuRecDataBatch(
            context=[context],
            supervision=[supervision] if supervision is not None else None,
            context_rig=[context_rig],
            supervision_rig=[supervision_rig] if supervision_rig is not None else None,
            cuboid_tracks=(
                [self._cuboid_tracks(index, context_batch, T_world_ref)] if self.config.cuboid_tracks is not None else None
            ),
            meta=[{"sequence_id": index["sequence_id"], "index_path": index["_path"],
                   **({"world_origin": index["_world_origin"]} if "_world_origin" in index else {})}],
        )

    def load_full_camera_rig(self, index_path: str, reference_rig: RigTrajectories, camera_id: str,
                             source_resolution: bool = False) -> RigTrajectories:
        """Every frame of ``camera_id`` in the log as a lightweight rig (no images), in ``reference_rig``'s frame;
        the PMT counterpart of ``NCoreInstantNuRecDataset.load_full_camera_rig`` (calibrated video). If --max-chunks
        truncated the log, only the frames inside the reconstructed chunks (the NCore loader raises instead).
        ``source_resolution``: the index's own pinhole camera (``K`` / ``wh``, e.g. OpenScene's 1920x1080, which
        the images are undistorted onto) instead of the model's cropped and resized input camera."""
        index = next(x for x in self.indexes if x["_path"] == str(index_path))
        times = {cam: np.array([index["frames"][cam][i]["t_us"] for i in self._context_candidates(index, cam)],
                               np.int64) for cam in self.config.camera_ids}
        interval = self._sequence_interval(index, times)
        sampler = self.config.frame_batch_sampler
        n, k = self.config.n_frames_per_sample, sampler.chunk_overlap_frames
        required = count_sequential_chunks(interval.end - interval.start, n, sampler.max_frame_gap_timestamp_us, k)
        calibration = reference_rig.camera_calibrations[camera_id]
        if source_resolution:
            meta = index["cameras"][camera_id]
            calibration = dataclasses.replace(calibration, camera_model_parameters=pinhole_parameters(
                np.asarray(meta["K"], np.float64), tuple(meta["wh"])))
        stamps = np.array([f["t_us"] for f in index["frames"][camera_id]], np.int64)
        if required > self.num_samples_per_sequence:
            # --max-chunks truncated the log: render only the span of the reconstructed chunks. The sampler cuts
            # the interval into ``required`` chunks of equal slots and keeps the first ``num_samples_per_sequence``.
            m = self.num_samples_per_sequence
            covered_end = interval.start + (interval.end - interval.start) * (m * (n - k) + k) / (required * (n - k) + k)
            n_all = len(stamps)
            stamps = stamps[stamps < covered_end]
            if len(stamps) == 0:
                raise ValueError(f"{index_path}: no {camera_id} frame inside the {self.num_samples_per_sequence} "
                                 f"reconstructed chunk(s)")
            logger.warning(
                "--max-chunks=%d covers %d of %d chunks of this log: the video renders %d of %d %s frames "
                "(%.1f s of %.1f s). Rerun with --max-chunks %d for the whole log.",
                self.num_samples_per_sequence, self.num_samples_per_sequence, required, len(stamps), n_all,
                camera_id, (covered_end - interval.start) / 1e6, (interval.end - interval.start) / 1e6, required,
            )
        ts, poses = index["rig"]["timestamps_us"], index["rig"]["T_ego_world"]
        ts = np.concatenate([[min(ts[0], stamps[0]) - 1], ts, [max(ts[-1], stamps[-1]) + 1]])
        poses = np.concatenate([poses[:1], poses, poses[-1:]])
        T_world_ref = se3_matrix_inverse(reference_rig.T_world_base.to(torch.float64)).cpu().numpy()
        trajectory = RigTrajectories.RigTrajectory(
            sequence_id=calibration.sequence_id,
            cameras_frame_timestamps_us={camera_id: torch.as_tensor(np.stack([stamps, stamps], 1))},
            T_rig_worlds=torch.as_tensor(T_world_ref @ poses, dtype=torch.float64),
            T_rig_world_timestamps_us=torch.as_tensor(ts, dtype=torch.int64),
        )
        return RigTrajectories(
            T_world_base=reference_rig.T_world_base,
            world_to_scene=reference_rig.world_to_scene,
            rig_trajectories=[trajectory],
            camera_calibrations=OrderedDict([(camera_id, calibration)]),
        )
