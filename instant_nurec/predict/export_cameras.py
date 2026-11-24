# SPDX-License-Identifier: Apache-2.0
"""Context-camera sidecar (``.cameras.npz``) next to a PMT-index PLY, for the viewer (``tools/viewer.py``).

Written only for PMT index logs (``run_inference.py --index-path``); the NCore outputs are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.sensor import to_simple_pinhole_model_parameters
from instant_nurec.utils.types import RigTrajectories


def cameras_sidecar_path(ply_path: Path) -> Path:
    return ply_path.with_suffix(".cameras.npz")


def export_cameras(context: DataAndRenderingBatch, rig: RigTrajectories, world_origin, ply_path: Path) -> Path:
    """Every context frame in the PLY's (world) frame: OpenCV camera-to-world, pinhole K at the model resolution;
    and the ego (rig) trajectory the chunks were built on (``ego_T_world``: rig -> PLY world, at
    ``ego_timestamps_us``; the context chunks' spans +- 1 s, so it covers every context frame).

    ``world_origin`` is the dataset-world translation of the PLY frame's origin (PMT predict moves the world to
    the log's first rig pose), so ``dataset world = PLY + world_origin``.
    """
    rendering = unpack_optional(unpack_optional(context.rendering).camera)
    data = unpack_optional(context.data.camera)
    c2w = tquat_to_se3_matrix(rendering.poses_tquat_startend[:, 1, :], unbatch=False).double()
    c2w = rig.T_world_base.to(device=c2w.device, dtype=torch.float64) @ c2w
    names = {calibration.unique_sensor_idx: name for name, calibration in rig.camera_calibrations.items()}
    Ks, whs = [], []
    for parameters in rendering.sensor_model_parameters:
        # PMT cameras are undistorted pinholes: their own K; anything else, the encoder's simple pinhole
        pinhole = parameters if hasattr(parameters, "focal_length") else to_simple_pinhole_model_parameters(parameters)
        (fx, fy), (cx, cy) = pinhole.focal_length, pinhole.principal_point
        Ks.append([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        whs.append([int(v) for v in pinhole.resolution])
    T_world_base = rig.T_world_base.to(dtype=torch.float64)
    ego_t, ego_T = [], []
    for trajectory in rig.rig_trajectories:
        ego_t.append(trajectory.T_rig_world_timestamps_us.cpu().long())
        ego_T.append(T_world_base.to(trajectory.T_rig_worlds.device) @ trajectory.T_rig_worlds.double())
    ego_t, ego_T = torch.cat(ego_t), torch.cat([T.cpu() for T in ego_T])
    order = torch.argsort(ego_t)
    ego_t, ego_T = ego_t[order], ego_T[order]
    first = torch.ones_like(ego_t, dtype=torch.bool)  # one pose per timestamp
    first[1:] = ego_t[1:] != ego_t[:-1]
    path = cameras_sidecar_path(ply_path)
    np.savez(
        path,
        c2w=c2w.cpu().numpy(),
        K=np.asarray(Ks, np.float64),
        wh=np.asarray(whs, np.int64),
        camera=np.asarray([names[meta.unique_sensor_idx] for meta in data.meta]),
        timestamps_us=rendering.timestamps_startend_us_cpu[:, 1].numpy().astype(np.int64),
        world_origin=np.asarray(world_origin if world_origin is not None else (0.0, 0.0, 0.0), np.float64),
        ego_T_world=ego_T[first].numpy(),
        ego_timestamps_us=ego_t[first].numpy(),
    )
    return path
