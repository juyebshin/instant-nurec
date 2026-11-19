# SPDX-License-Identifier: Apache-2.0
"""Training dataset for non-NCore logs (AV2 / nuScenes) from per-log index pickles.

The index is written by PseudoMapTrainer ``RoGS/scripts/inurec_finetune_index.py``
(that env has the AV2 / nuScenes devkits; this one does not)::

    sequence_id: str
    cameras: {name: {"K": (3, 3), "wh": (W, H), "T_cam_ego": (4, 4)}}   # OpenCV camera frame
    rig: {"timestamps_us": (N,) int64, "T_ego_world": (N, 4, 4) float64}  # one pose per image timestamp
    frames: {name: [{"t_us": int, "img": str, "seg": str | None, "depth": str | None}, ...]}  # sorted by t_us
    semantic_lut: (256,) uint8   # segmentation id -> KelvinSemanticClass, 255 = ignore

Depth files are DVGT/MoGe log-encoded uint16 PNGs of camera z-depth (``near`` / ``far`` in
the PNG text; 0 = unknown, 65535 = inf); they are converted to ray distance here.

The batch mirrors ``NCoreInstantNuRecDataset``: rig = ego, world = the first camera's
first context frame, cameras global-shutter pinhole, no cuboid tracks (moving objects go
to the learned-motion dynamic layer).
"""

from __future__ import annotations

import hashlib
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
from instant_nurec.primitives.kelvin_primitive import KelvinSemanticClass as C
from instant_nurec.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    InstantNuRecDataBatch,
)
from instant_nurec.utils.geometry import se3_matrix_inverse
from instant_nurec.utils.types import FrameConversion, RayFlags, RigTrajectories

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


class PMTIndexDataset(BaseInstantNuRecIndexableDataset[InstantNuRecDataBatch]):
    def __init__(self, config: PMTIndexDatasetConfig, global_seed: int, retry_on_error: bool = False):
        self.config = config
        self._global_seed = global_seed
        self._retry_on_error = retry_on_error
        self._rng_epoch = -1
        self._epoch = 0
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
            self.indexes.append(index)

    def __len__(self) -> int:
        return len(self.indexes) * self.config.n_samples_per_sequence

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
        image = cv2.cvtColor(cv2.imread(frame["img"]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        labels = CameraFrameLabels(rgb=torch.from_numpy(subsampler.apply_frame_data(image))[None])
        flags = torch.full((h, w), int(RayFlags.RGB_LABEL), dtype=torch.int32)
        sky_full = None
        if self.config.semantic_segmentation:
            if frame["seg"] is not None:
                classes_full = index["semantic_lut"][cv2.imread(frame["seg"], cv2.IMREAD_UNCHANGED)]
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
            meta = index["cameras"][cam]
            depth = read_log_depth(frame["depth"])
            if sky_full is not None:  # as the NCore loader: no depth on sky
                if sky_full.shape != depth.shape:  # e.g. nuScenes MoGe depth is a downscaled image
                    sky_full = cv2.resize(sky_full.astype(np.uint8), depth.shape[::-1],
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                depth[sky_full] = 0.0
            distance = z_to_ray_distance(depth, np.asarray(meta["K"], np.float64), tuple(meta["wh"]))
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

    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> InstantNuRecDataBatch:
        index = self.indexes[batch_idx // self.config.n_samples_per_sequence]
        cams = self.config.camera_ids
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
            cuboid_tracks=None,
            meta=[{"sequence_id": index["sequence_id"], "index_path": index["_path"]}],
        )
