# SPDX-License-Identifier: Apache-2.0
"""Per-log index pickles for fine-tuning Instant-NuRec on AV2 / nuScenes / Waymo / OpenScene.

Read by ``instant_nurec/datasets/pmt_index.py`` (``PMTIndexDataset``). Moved here on 2026-11-25 from
PseudoMapTrainer ``RoGS/scripts/inurec_finetune_index.py`` (untracked there), logic unchanged; runs in
``sjy-inurec``, which has the av2 / nuscenes-devkit packages since 2026-11-25. One pickle per log / scene:

    cameras: {name: {K, wh, T_cam_ego}}                 # OpenCV camera frame
    rig: {timestamps_us, T_ego_world}                     # ego pose at every image timestamp
    frames: {name: [{t_us, img, seg, depth}, ...]}        # every image; seg / depth or None
    semantic_lut: Mapillary Vistas v2 id -> 8-class Instant-NuRec head (tools/index_export/semantic_lut.py)
    tracks: {id: {label, dims, timestamps_us, T_box_world}}   # OpenScene only: annotated 3D boxes

Labels: M2F Mapillary-v2 masks (RoGS label dirs; Waymo: ``Mask2Former/demo/inference_waymo.py``,
predicted on the raw distorted images; ``seg`` is None where no mask exists) and LiDAR-aligned
MoGe-3 depth made by DVGT's pipeline (AV2: 2 Hz keyframes of all 1000 logs; nuScenes: trainval keyframes; Waymo: 2 Hz
frames of all 1000 segments).

Waymo (``waymo_ns``, DVGT's nerfstudio-layout conversion) has lens distortion, and the depth
sits on its own undistorted, cropped and anisotropically resized grid. So a Waymo camera also
carries ``distortion`` {K, D, wh} (raw jpgs and masks; the loader undistorts them onto {K, wh}) and
``depth_grid`` {K, wh} (the depth PNGs' pinhole grid, reproduced with the labeling code itself).

OpenScene (NAVSIM log pickles, 2 Hz, 8 distorted cameras) is exported like Waymo, one pickle per geo-split run
``<log>:<first>-<last>`` (positions in the log pickle; instant-nurec ``configs/splits/geo_near/openscene``). Its
MoGe-3 depth is DVGT's (``open_scene_loader.py``, num_tokens 2500); its masks come from PMT
``Mask2Former/demo/inference_openscene.py`` at ``data/m2f/openscene/<log>/seg_<CAM>/<token>.png`` (Vistas v2 ids).
A camera's mask grid is read off its first mask: raw size (1920x1080) = predicted on the raw image, undistorted on
load like Waymo; depth-grid size = predicted on DVGT's MoGe image, so the camera carries ``seg_grid`` = its
``depth_grid`` and the loader resamples masks like depth. It also writes the
annotated boxes as ``tracks``: per track the class name, (l, w, h) along the box's x / y / z, and a box -> world
pose (centre, yaw about z) at every annotated frame; instant-nurec turns them into cuboid tracks, the motion
head's supervision.

Every input is read through this repo's ``data/`` (symlinked datasets, Mask2Former outputs and the copied
MoGe-3 labels; ``tools/moge_labels/README.md``), so the pickles reference no other repo's tree. The batch
scripts are ``scripts/index/export_*.sh``; one log by hand, from the repo root:

    python tools/index_export/export_index.py --dataset av2 --split train \
        --scenes 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a --out-dir data/index/av2/train
    python tools/index_export/export_index.py --dataset waymo \
        --scenes 10017090168044687777_6380_000_6400_000 --out-dir /tmp/index_test
"""
import argparse
import json
import os
import os.path as osp
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

INUREC = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # repo root; pickles store paths under it
sys.path.insert(0, INUREC)
from tools.index_export.semantic_lut import semantic_lut  # noqa: E402

DATA = f"{INUREC}/data"
MOGE = f"{DATA}/moge3/align_depth"
WAYMO = f"{DATA}/waymo_ns"
AV2_CAMS = ["ring_front_center", "ring_front_right", "ring_front_left", "ring_rear_right",
            "ring_rear_left", "ring_side_right", "ring_side_left"]
NUSC_CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
             "CAM_BACK_RIGHT"]
WAYMO_CAMS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
OPENSCENE = f"{DATA}/openscene"
OPENSCENE_CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]
OPENSCENE_EXCLUDED = {  # DVGT open_scene_loader.py EXCLUDED_SCENES (bad poses; not depth-labelled)
    "log-0333-scene-0001", "log-0558-scene-0002", "log-0761-scene-0002", "log-0767-scene-0002",
    "log-0781-scene-0001", "log-0801-scene-0002", "log-0973-scene-0003", "log-1035-scene-0003",
    "log-1035-scene-0004", "log-1214-scene-0002", "log-0046-scene-0003", "log-0132-scene-0013"}
RUB_TO_RDF = np.diag([1.0, -1.0, -1.0, 1.0])  # nerfstudio (OpenGL) camera axes -> OpenCV


def maybe(path):
    return path if osp.exists(path) else None


def index_av2(log, split):
    from av2.geometry.camera.pinhole_camera import PinholeCamera
    from av2.utils.io import read_city_SE3_ego
    if split == "auto":  # geo splits cut across the official train / val / test directories
        split = next(s for s in ("train", "val", "test") if osp.isdir(f"{DATA}/av2/{s}/{log}"))
    root = Path(f"{DATA}/av2/{split}")
    city_SE3_ego = read_city_SE3_ego(root / log)  # one feather read (AV2SensorDataLoader re-reads per call)
    cameras, frames, poses = {}, {}, {}
    for cam in AV2_CAMS:
        pin = PinholeCamera.from_feather(log_dir=root / log, cam_name=cam)
        cameras[cam] = dict(K=pin.intrinsics.K.astype(np.float64), wh=(pin.width_px, pin.height_px),
                            T_cam_ego=pin.ego_SE3_cam.transform_matrix.astype(np.float64))
        img_dir = root / log / "sensors" / "cameras" / cam
        frames[cam] = []
        for p in sorted(img_dir.glob("*.jpg")):
            t_ns = int(p.stem)
            if t_ns not in poses:
                poses[t_ns] = city_SE3_ego[t_ns].transform_matrix
            frames[cam].append(dict(
                t_us=t_ns // 1000, img=str(p),
                seg=maybe(f"{DATA}/m2f/av2/{split}/{log}/sensors/cameras/seg_{cam}/{t_ns}.png"),
                depth=maybe(f"{MOGE}/argoverse/{log}/{cam}/{t_ns}.png")))
    ts = np.array(sorted(poses), dtype=np.int64)
    rig = dict(timestamps_us=ts // 1000, T_ego_world=np.stack([poses[t] for t in ts]).astype(np.float64))
    return cameras, frames, rig


def index_nusc(scene, nusc):
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    rec = next(s for s in nusc.scene if s["name"] == scene)
    first = nusc.get("sample", rec["first_sample_token"])
    cameras, frames, poses = {}, {}, {}
    for cam in NUSC_CAMS:
        sd = nusc.get("sample_data", first["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        cameras[cam] = dict(K=np.array(cs["camera_intrinsic"], np.float64), wh=(sd["width"], sd["height"]),
                            T_cam_ego=transform_matrix(cs["translation"], Quaternion(cs["rotation"])))
        frames[cam] = []
        while True:  # keyframes + sweeps; the sample_data chain stays inside the scene
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            poses[sd["timestamp"]] = transform_matrix(ep["translation"], Quaternion(ep["rotation"]))
            kind, _, name = sd["filename"].split("/")
            stem = osp.splitext(name)[0]
            frames[cam].append(dict(
                t_us=sd["timestamp"], img=osp.join(nusc.dataroot, sd["filename"]),
                seg=maybe(f"{DATA}/m2f/nuscenes/{kind}/seg_{cam}/{stem}.png"),
                depth=maybe(f"{MOGE}/nuscene/samples/{cam}/{stem}.png") if sd["is_key_frame"] else None))
            if not sd["next"]:
                break
            sd = nusc.get("sample_data", sd["next"])
    ts = np.array(sorted(poses), dtype=np.int64)
    rig = dict(timestamps_us=ts, T_ego_world=np.stack([poses[t] for t in ts]).astype(np.float64))
    return cameras, frames, rig


def dvgt_waymo_depth_grid(K, D, wh, num_tokens=3700):
    """The pinhole grid of the Waymo depth / image files (``waymo_loader.py``: undistort, MoGe token
    resize and principal-point crop, then resize back to the raw aspect), from the labeling code.
    OpenScene's loader does the same with its default num_tokens=2500."""
    sys.path.insert(0, INUREC)
    from tools.moge_labels.utils.image import process_image_sequentially, resize_to_aspect_ratio
    blank = np.zeros((wh[1], wh[0], 3), np.uint8)
    image, K_in, _ = process_image_sequentially(blank, K, D, num_tokens=num_tokens)  # WaymoDataset default 3700
    image, K_save = resize_to_aspect_ratio(image, K_in, wh[0] / wh[1])
    return K_save, (image.shape[1], image.shape[0])


def index_waymo(segment):
    root = Path(f"{WAYMO}/{segment}")
    meta = json.loads((root / "transforms.json").read_text())
    cameras, frames = {}, {}
    for cam in WAYMO_CAMS:
        p = meta["sensor_params"][cam]
        K_raw, D, wh = np.array(p["camera_intrinsic"], np.float64), np.array(p["camera_D"], np.float64), \
            (int(p["width"]), int(p["height"]))
        K, _ = cv2.getOptimalNewCameraMatrix(K_raw, D, wh, alpha=0)  # undistorted, no black border
        K_depth, wh_depth = dvgt_waymo_depth_grid(K_raw, D, wh)
        cameras[cam] = dict(K=K, wh=wh, T_cam_ego=np.array(p["extrinsic"], np.float64) @ RUB_TO_RDF,
                            distortion=dict(K=K_raw, D=D, wh=wh),
                            depth_grid=dict(K=K_depth.astype(np.float64), wh=wh_depth))
        frames[cam] = []
    for f in meta["frames"]:
        img = root / f["file_path"]
        t_us = int(img.stem)
        depth = maybe(f"{MOGE}/waymo/{segment}/images/{f['camera']}/{t_us}.png")
        seg = maybe(f"{DATA}/m2f/waymo/{segment}/seg_{f['camera']}/{t_us}.png")
        frames[f["camera"]].append(dict(t_us=t_us, img=str(img), seg=seg, depth=depth))
    for cam in WAYMO_CAMS:
        frames[cam].sort(key=lambda x: x["t_us"])
        first = next((x["depth"] for x in frames[cam] if x["depth"]), None)
        if first is not None:  # the reproduced grid must match the label files
            assert cv2.imread(first, cv2.IMREAD_UNCHANGED).shape[::-1] == cameras[cam]["depth_grid"]["wh"], first
    # Waymo ego (vehicle) pose == lidar pose; camera frames share the lidar timestamps in waymo_ns.
    ts = np.array([int(Path(x["file_path"]).stem) for x in meta["lidar_frames"]], np.int64)
    poses = np.stack([np.array(x["transform_matrix"], np.float64) for x in meta["lidar_frames"]])
    order = np.argsort(ts)
    return cameras, frames, dict(timestamps_us=ts[order], T_ego_world=poses[order])


def index_openscene(run):
    """run = '<log>:<first>-<last>', a geo-split run of positions in the NAVSIM log pickle."""
    log, span = run.rsplit(":", 1)
    first, last = map(int, span.split("-"))
    with open(f"{OPENSCENE}/navsim_logs/trainval/{log}.pkl", "rb") as f:
        metas = pickle.load(f)[first:last + 1]
    assert len(metas) == last - first + 1, (run, len(metas))
    assert not {m["scene_name"] for m in metas} & OPENSCENE_EXCLUDED, run  # none in the geo split (2026-11-23)
    t = np.array([m["timestamp"] for m in metas], np.int64)
    # 2 Hz; 32 of the 750 geo runs of >= 8 frames drop a frame or two (1.0 / 1.5 s gaps, 2026-11-23)
    assert (np.diff(t) > 0).all() and (np.diff(t) < 1_600_000).all(), run
    cameras, frames = {}, {}
    for cam in OPENSCENE_CAMS:
        c = metas[0]["cams"][cam]
        K_raw, D = np.array(c["cam_intrinsic"], np.float64), np.array(c["distortion"], np.float64)
        for m in metas[1:]:  # calibration is per log; the index stores it once
            assert np.allclose(m["cams"][cam]["cam_intrinsic"], K_raw) and np.allclose(
                m["cams"][cam]["sensor2lidar_translation"], c["sensor2lidar_translation"]), (run, cam)
        h, w = cv2.imread(f"{OPENSCENE}/sensor_blobs/all_sensor_blobs/{c['data_path']}").shape[:2]
        K, _ = cv2.getOptimalNewCameraMatrix(K_raw, D, (w, h), alpha=0)  # undistorted, no black border
        K_depth, wh_depth = dvgt_waymo_depth_grid(K_raw, D, (w, h), num_tokens=2500)
        T_lidar_cam = np.eye(4)
        T_lidar_cam[:3, :3], T_lidar_cam[:3, 3] = c["sensor2lidar_rotation"], c["sensor2lidar_translation"]
        grid = dict(K=K_depth.astype(np.float64), wh=wh_depth)
        cameras[cam] = dict(K=K, wh=(w, h), T_cam_ego=np.asarray(metas[0]["lidar2ego"], np.float64) @ T_lidar_cam,
                            distortion=dict(K=K_raw, D=D, wh=(w, h)), depth_grid=grid)
        frames[cam] = []
        for m in metas:
            path = m["cams"][cam]["data_path"]  # <log>/<CAM>/<token>.jpg
            stem = osp.splitext(path)[0]
            frames[cam].append(dict(
                t_us=int(m["timestamp"]), img=f"{OPENSCENE}/sensor_blobs/all_sensor_blobs/{path}",
                seg=maybe(f"{DATA}/m2f/openscene/{log}/seg_{cam}/{osp.basename(stem)}.png"),
                depth=maybe(f"{MOGE}/openscene/{stem}.png")))
        depth0 = next((x["depth"] for x in frames[cam] if x["depth"]), None)
        if depth0 is not None:  # the reproduced grid must match the label files
            assert cv2.imread(depth0, cv2.IMREAD_UNCHANGED).shape[::-1] == wh_depth, depth0
        seg0 = next((x["seg"] for x in frames[cam] if x["seg"]), None)
        if seg0 is not None:  # raw-image masks (undistorted on load) or masks on the MoGe grid (resampled)
            seg_wh = cv2.imread(seg0, cv2.IMREAD_UNCHANGED).shape[::-1]
            assert seg_wh in ((w, h), wh_depth), (seg0, seg_wh)
            if seg_wh == wh_depth:
                cameras[cam]["seg_grid"] = grid
    rig = dict(timestamps_us=t, T_ego_world=np.stack([np.asarray(m["ego2global"], np.float64) for m in metas]))
    tracks = {}
    for m in metas:  # gt_boxes: x, y, z, l, w, h, yaw in the lidar frame
        a = m["anns"]
        T_world_lidar = np.asarray(m["ego2global"], np.float64) @ np.asarray(m["lidar2ego"], np.float64)
        for tok, name, box in zip(a["track_tokens"], a["gt_names"], np.asarray(a["gt_boxes"], np.float64)):
            tr = tracks.setdefault(str(tok), dict(label=str(name), dims=box[3:6].copy(), timestamps_us=[],
                                                  T_box_world=[]))
            if tr["timestamps_us"] and tr["timestamps_us"][-1] == int(m["timestamp"]):
                continue  # duplicate box of a track in one frame
            T = np.eye(4)
            cy, sy = np.cos(box[6]), np.sin(box[6])
            T[:3, :3], T[:3, 3] = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], box[:3]
            tr["timestamps_us"].append(int(m["timestamp"]))
            tr["T_box_world"].append(T_world_lidar @ T)
    for tr in tracks.values():
        tr["timestamps_us"] = np.array(tr["timestamps_us"], np.int64)
        tr["T_box_world"] = np.stack(tr["T_box_world"])
    return cameras, frames, rig, tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("av2", "nusc", "waymo", "openscene"), required=True)
    ap.add_argument("--split", default="train", help="AV2 split dir, or 'auto' to find the log's")
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="AV2 log ids / nuScenes scene names / Waymo segment names / OpenScene runs <log>:<a>-<b>")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    nusc = None
    if args.dataset == "nusc":
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version="v1.0-trainval", dataroot=f"{DATA}/nuscenes", verbose=False)
    lut = semantic_lut()
    for scene in args.scenes:
        extra = {}
        if args.dataset == "openscene":
            cameras, frames, rig, extra["tracks"] = index_openscene(scene)
            scene = scene.replace(":", "_")  # file / sequence name <log>_<first>-<last>
        elif args.dataset == "waymo":
            cameras, frames, rig = index_waymo(scene)
        elif args.dataset == "av2":
            cameras, frames, rig = index_av2(scene, args.split)
        else:
            cameras, frames, rig = index_nusc(scene, nusc)
        out = osp.join(args.out_dir, f"{scene}.pkl")
        with open(out, "wb") as f:
            pickle.dump(dict(sequence_id=f"{args.dataset}-{scene}", cameras=cameras, frames=frames, rig=rig,
                             semantic_lut=lut, **extra), f)
        n = {c: (len(v), sum(x["seg"] is not None for x in v), sum(x["depth"] is not None for x in v))
             for c, v in frames.items()}
        print(f"{out}: rig {len(rig['timestamps_us'])} poses; frames (all, seg, depth) per camera {n}"
              + (f"; {len(extra['tracks'])} tracks" if "tracks" in extra else ""))


if __name__ == "__main__":
    main()
