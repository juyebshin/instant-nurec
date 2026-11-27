"""Render an exported scene through one logged camera, along the ego's logged trajectory: RGB and Gaussian-opacity
PNGs per frame, and optionally an MP4, at any times (e.g. 10 Hz between the 2 Hz index frames). Works on any
export folder, including an edited one from ``tools/track_edit.py``.

    python tools/render_frames.py <scene.ply> --index <log.pkl> --out <dir>                  # the index's frames
    python tools/render_frames.py <scene.ply> --index <log.pkl> --out <dir> \\
        --start-us 1631646460700169 --end-us 1631646466000169 --hz 10 --resolution source --video

Writes ``<dir>/images<sfx>/<t_us>.png`` (RGB), ``<dir>/opacity<sfx>/<t_us>.png`` (8-bit gray, 255 = fully covered by
Gaussians) and with ``--video`` ``<dir>/render<sfx>.mp4``. ``<sfx>`` is empty at ``--resolution model`` (the model's
input camera, e.g. 504x280, from the export's ``.cameras.npz``) and ``-<H>p`` at ``--resolution source`` (the
index's undistorted camera, e.g. 1920x1080 for OpenScene).

Pixels are those of ``run_inference.py --render-video`` (``render_composited_frame``): gsplat 3DGUT with calibrated
pinhole rays, near plane 0.2, sky composited behind the Gaussians, the camera's ISP affine, rounded to uint8. The
camera at time ``t`` is ``ego(t) @ T_cam_ego``: the ego pose of the export's ``.cameras.npz`` (the index's rig
poses), linear in position and slerp in rotation between them (``viewer.Scene.ego_pose``), and the index camera's
extrinsics. Needs an ``--index-path`` export (``.cameras.npz`` with ``ego_T_world`` and ``world_origin``).
"""

from __future__ import annotations

import argparse
import pickle
import sys

from pathlib import Path

import numpy as np
import torch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer as V

from instant_nurec.predict.render_preview import composite_sky_and_affine
from instant_nurec.predict.render_video import encode_video


def frame_times(args, index: dict, scene: V.Scene) -> list[int]:
    if args.start_us is None and args.end_us is None:  # the camera's index frames inside the scene's time span
        stamps = [int(f["t_us"]) for f in index["frames"][args.camera]]
        return [t for t in stamps if scene.t0 <= t <= scene.t1]
    if args.start_us is None or args.end_us is None:
        raise SystemExit("--start-us and --end-us go together")
    step = round(1e6 / args.hz)
    return list(range(args.start_us, args.end_us + 1, step))


def camera(args, index: dict, scene: V.Scene) -> tuple[np.ndarray, tuple[int, int]]:
    if args.resolution == "source":
        meta = index["cameras"][args.camera]
        return np.asarray(meta["K"], np.float64), tuple(int(x) for x in meta["wh"])
    rows = np.flatnonzero(scene.cameras["camera"] == args.camera)
    if len(rows) == 0:
        raise SystemExit(f"{args.camera} is not a context camera of this export; use --resolution source")
    return scene.cameras["K"][rows[0]].astype(np.float64), tuple(int(x) for x in scene.cameras["wh"][rows[0]])


@torch.inference_mode()
def render(renderer: V.Renderer, c2w: np.ndarray, K: np.ndarray, wh: tuple[int, int], t_us: int,
           affine_index: int) -> tuple[np.ndarray, np.ndarray]:
    """(uint8 RGB (H, W, 3), uint8 opacity (H, W)) of the scene at ``t_us`` from OpenCV camera-to-world ``c2w``."""
    scene, (w, h) = renderer.scene, wh
    device = scene.positions.device
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
    c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    v, u = torch.meshgrid(torch.arange(h, device=device) + 0.5, torch.arange(w, device=device) + 0.5, indexing="ij")
    dirs_cam = torch.stack([(u - K_t[0, 2]) / K_t[0, 0], (v - K_t[1, 2]) / K_t[1, 1], torch.ones_like(u)], -1)
    dirs = torch.nn.functional.normalize(dirs_cam @ c2w_t[:3, :3].T, dim=-1)
    rays = torch.cat([c2w_t[:3, 3].expand(h, w, 3), dirs], -1).contiguous()
    means, quats, scales, opacities, colors = renderer._gaussians(False, t_us, True)  # static + dynamic at t_us
    def raster(config):
        return renderer.rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=torch.linalg.inv(c2w_t)[None], Ks=K_t[None], width=w, height=h, near_plane=0.2,
            far_plane=torch.finfo(torch.float32).max, sh_degree=None, render_mode="RGB", packed=False, with_ut=True,
            ut_params=V._reference_ut_parameters(), renderer_config=config, with_eval3d=True, global_z_order=False,
            rays=rays[None], camera_model="pinhole",
        )

    try:
        out, alpha, _ = raster(renderer.parallel_batch)
    except RuntimeError as e:  # a tile with > ~2 M Gaussians: see viewer.Renderer.render
        if "max batches per tile" not in str(e):
            raise
        out, alpha, _ = raster(renderer.mixed_batch)
    opacity = alpha[0, ..., 0]
    image = composite_sky_and_affine(out[0, ..., :3], opacity, V.sample_sky_cubemap(scene.sky, dirs),
                                     scene.affine[affine_index])
    to_u8 = lambda x: x.mul(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8).cpu().numpy()
    return to_u8(image), to_u8(opacity)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply", type=Path, help="the export's <log>.ply (sidecars next to it)")
    p.add_argument("--index", type=Path, required=True, help="the log's index pickle (camera extrinsics, frames)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera", default="CAM_F0")
    p.add_argument("--resolution", choices=("model", "source"), default="model")
    p.add_argument("--start-us", type=int, default=None, help="first frame time (default: the index's frames)")
    p.add_argument("--end-us", type=int, default=None, help="last frame time, inclusive")
    p.add_argument("--hz", type=float, default=10.0, help="frame rate between --start-us and --end-us")
    p.add_argument("--video", action="store_true", help="also write render<sfx>.mp4")
    args = p.parse_args()

    device = torch.device("cuda")
    with open(args.index, "rb") as f:
        index = pickle.load(f)
    scene = V.Scene(args.ply, device)
    if scene.cameras is None or "ego_T_world" not in scene.cameras or scene.sky is None:
        raise SystemExit(f"{args.ply}: needs an --index-path export with .cameras.npz (ego_T_world) and .sky.npz")
    renderer = V.Renderer(scene)
    origin = scene.cameras["world_origin"].astype(np.float64)
    T_cam_ego = np.asarray(index["cameras"][args.camera]["T_cam_ego"], np.float64)  # camera -> ego
    for t, T in zip(index["rig"]["timestamps_us"], index["rig"]["T_ego_world"]):  # same world as the export
        if scene.t0 <= t <= scene.t1:
            ref = np.asarray(T, np.float64).copy()
            ref[:3, 3] -= origin
            if np.abs(scene.ego_pose(int(t)) - ref).max() > 1e-3:
                raise SystemExit(f"{args.index} does not match the export's ego poses at {t}")
    times = frame_times(args, index, scene)
    K, (w, h) = camera(args, index, scene)
    affine_index = scene.affine_names.index(args.camera)
    sfx = "" if args.resolution == "model" else f"-{h}p"
    for name in ("images", "opacity"):
        (args.out / f"{name}{sfx}").mkdir(parents=True, exist_ok=True)
    print(f"{len(times)} frames of {args.camera} at {w}x{h}, t {times[0]} .. {times[-1]} us", flush=True)

    def frames():
        for i, t in enumerate(times):
            c2w = scene.ego_pose(t) @ T_cam_ego  # the scene's world (UTM - world_origin)
            rgb, opacity = render(renderer, c2w, K, (w, h), t, affine_index)
            Image.fromarray(rgb).save(args.out / f"images{sfx}" / f"{t}.png")
            Image.fromarray(opacity, mode="L").save(args.out / f"opacity{sfx}" / f"{t}.png")
            if i % 20 == 0:
                print(f"{i + 1} / {len(times)}", flush=True)
            yield rgb

    if args.video:
        fps = args.hz if args.start_us is not None else 1e6 / float(np.median(np.diff(times)))
        n = encode_video(frames(), width=w, height=h, fps=fps, path=args.out / f"render{sfx}.mp4")
        print(f"wrote {n} frames and {args.out / f'render{sfx}.mp4'}", flush=True)
    else:
        n = sum(1 for _ in frames())
        print(f"wrote {n} frames", flush=True)
    print(f"{args.out / f'images{sfx}'}, {args.out / f'opacity{sfx}'}", flush=True)


if __name__ == "__main__":
    main()
