#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interactive viewer for ``run_inference.py`` outputs, nerfstudio-style: a viser web page whose view is rendered
on the GPU here (gsplat 3DGUT, as the model's own renderer) and streamed to the browser.

Reads ``<scene>.ply`` and, when they sit next to it, ``.sky.npz`` (sky cubemap + per-camera ISP affine),
``.dynamic.npz`` (``--include-dynamic``) and ``.cameras.npz`` (context cameras; written for ``--index-path`` runs).

    CUDA_VISIBLE_DEVICES=0 python tools/viewer.py outputs/predict/<run>/ply/<seq>/<seq>.ply --port 8080

then open http://localhost:8080 (through ``ssh -L 8080:localhost:8080 <host>`` from another machine).

The ego trajectory (``.cameras.npz``) is drawn with the ego at the time slider's value as a box + axes; the
traversed part of the trajectory is highlighted. With "follow ego" on, the view looks through the ego's front camera
(the context camera facing the ego's +x, at its pose on the rig and its vertical FoV) at the slider time, so play
drives the view along the trajectory as that camera saw it.

Render modes: rgb (Gaussians over the sky, through the chosen camera's ISP affine, as the model renders), rgb
without sky, semantic class (the PLY's ``semantic_class`` from ``--index-path`` runs; otherwise only its road / sky
masks), depth (ray distance) and opacity. Click a context-camera frustum to look through it: the view, time and
ISP camera snap to that frame. Moving the camera renders at a reduced size and refines once the view rests.
Frames go as JPEG only; "frustums behind geometry" adds the depth PNG (~20x the JPEG) to refined frames.

With ``--index <log.pkl>`` (an ``--index-path`` export with ``--include-dynamic``), "edit objects" moves, turns or
deletes dynamic objects by their index boxes (``tools/track_edit.py``, ``docs/world-model.md``): the boxes are drawn at
the slider time; click one (or pick it) for a gizmo, drag it or set offset / yaw in the box frame, and its Gaussians
are re-posed in the render. "save edits" writes ``edits.json`` and the edited scene (``track_edit.py --edits``).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _gsplat_toolchain_defaults() -> None:
    """The repo's gsplat JIT build with this env's own toolchain (CLAUDE.md), as ``conda activate sjy-inurec`` sets
    it. Without CC / CXX, torch writes ``cxx = c++`` and nvcc falls back to a ``gcc`` this machine lacks; ninja
    then sees changed compile commands and rebuilds, which fails. Paths keep the activated spelling
    (CONDA_PREFIX), since ninja compares command text and the cache was built with it."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    same_env = conda_prefix is not None and os.path.exists(conda_prefix) and os.path.samefile(conda_prefix, sys.prefix)
    prefix = conda_prefix if same_env else sys.prefix
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(REPO / ".cache" / "torch_extensions"))
    os.environ.setdefault("CUDA_HOME", prefix)
    bin_dir = os.path.join(prefix, "bin")  # ninja, nvcc
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    cc, cxx = (os.path.join(prefix, "bin", f"x86_64-conda-linux-gnu-{name}") for name in ("cc", "c++"))
    if "CXX" not in os.environ and os.path.exists(cxx):
        os.environ.setdefault("CC", cc)
        os.environ["CXX"] = cxx
        if "-ccbin" not in os.environ.get("NVCC_PREPEND_FLAGS", ""):
            os.environ["NVCC_PREPEND_FLAGS"] = f"{os.environ.get('NVCC_PREPEND_FLAGS', '')} -ccbin={cxx}"


_gsplat_toolchain_defaults()
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import point_cloud_utils as pcu  # noqa: E402
import torch  # noqa: E402
import viser  # noqa: E402
import viser.transforms as vtf  # noqa: E402

from instant_nurec.predict.render_preview import (  # noqa: E402
    _reference_renderer_config,
    _reference_ut_parameters,
    require_gsplat,
)
from instant_nurec.primitives.kelvin_primitive import KelvinDynamicLayer, KelvinSemanticClass  # noqa: E402
from instant_nurec.utils.cubemap import sample_sky_cubemap  # noqa: E402
from instant_nurec.utils.gaussians.utils import C0  # noqa: E402

PALETTE = {  # class -> RGB (Cityscapes-like)
    KelvinSemanticClass.OTHERS: (150, 150, 150),
    KelvinSemanticClass.EGO: (20, 20, 20),
    KelvinSemanticClass.SKY: (70, 130, 180),
    KelvinSemanticClass.ROAD: (128, 64, 128),
    KelvinSemanticClass.MOVABLE: (0, 0, 230),
}
for _name, _rgb in (("LANE", (250, 230, 30)), ("CROSSWALK", (255, 255, 255)), ("BOUNDARY", (255, 120, 0))):
    if hasattr(KelvinSemanticClass, _name):
        PALETTE[KelvinSemanticClass[_name]] = _rgb
CAMERA_COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                 (70, 240, 240), (240, 50, 230), (210, 245, 60)]


def _sidecar(ply: Path, suffix: str) -> Path | None:
    path = ply.with_suffix(suffix)
    return path if path.exists() else None


class Scene:
    """The exported Gaussians and sidecars on the GPU."""

    def __init__(self, ply: Path, device: torch.device):
        mesh = pcu.load_triangle_mesh(str(ply))
        attrs = mesh.vertex_data.custom_attributes
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)  # noqa: E731
        self.positions = t(mesh.vertex_data.positions).float()
        self.rotations = torch.nn.functional.normalize(
            t(np.stack([attrs[f"rot_{i}"] for i in range(4)], -1)).float(), dim=-1)  # wxyz
        self.scales = t(np.stack([attrs[f"scale_{i}"] for i in range(3)], -1)).float().exp()
        self.densities = torch.sigmoid(t(attrs["opacity"]).float())
        self.rgb = t(np.stack([attrs[f"f_dc_{i}"] for i in range(3)], -1)).float() * C0 + 0.5
        if "semantic_class" in attrs:
            self.semantic = t(attrs["semantic_class"]).long()
            self.semantic_source = "semantic_class"
        else:  # NCore / older outputs: the PLY keeps road and sky masks only
            self.semantic = torch.full_like(self.densities, int(KelvinSemanticClass.OTHERS), dtype=torch.long)
            if "road_mask" in attrs:
                self.semantic[t(attrs["road_mask"]) > 0] = int(KelvinSemanticClass.ROAD)
            if "sky_mask" in attrs:
                self.semantic[t(attrs["sky_mask"]) > 0.5] = int(KelvinSemanticClass.SKY)
            self.semantic_source = "road_mask / sky_mask"
        self.classes = sorted(int(c) for c in self.semantic.unique().tolist())

        self.sky = self.affine = None
        self.affine_names = ["(none)"]
        if (path := _sidecar(ply, ".sky.npz")) is not None:
            sky = np.load(path)
            self.sky = t(sky["sky_cubemap"]).float()
            self.affine = t(sky["affine_matrix"]).float()
            self.affine_names = [str(n) for n in sky["affine_sensor_ids"]]

        self.dynamic = None
        if (path := _sidecar(ply, ".dynamic.npz")) is not None:
            dyn = np.load(path)
            if int(dyn["n_layers"]) > 0 and len(dyn["layer_0_max_densities"]) > 0:
                bounds = dyn["layer_0_motion_time_bounds_us"] if "layer_0_motion_time_bounds_us" in dyn else None
                self.dynamic = KelvinDynamicLayer(
                    rotations=t(dyn["layer_0_rotations"]).float(),
                    scales=t(dyn["layer_0_scales"]).float(),
                    rgb=t(dyn["layer_0_rgb"]).float(),
                    max_densities=t(dyn["layer_0_max_densities"]).float(),
                    keyframe_positions=t(dyn["layer_0_keyframe_positions"]).float(),
                    keyframe_timestamps_us=t(dyn["layer_0_keyframe_timestamps_us"]).long(),
                    falloff=bool(dyn["layer_0_falloff"]),
                    motion_time_bounds_us=None if bounds is None else t(bounds).long(),
                )

        self.cameras = None
        if (path := _sidecar(ply, ".cameras.npz")) is not None:
            self.cameras = {k: v for k, v in np.load(path).items()}
        if self.cameras is not None:
            stamps = self.cameras["timestamps_us"]
        elif self.dynamic is not None:
            stamps = self.dynamic.keyframe_timestamps_us[:, 1].cpu().numpy()
        else:
            stamps = np.zeros(1, np.int64)
        self.t0, self.t1 = int(stamps.min()), int(stamps.max())

        # ego trajectory: the rig poses of the export (.cameras.npz ego_T_world, 2026-11-24 on); older files fall
        # back to the first camera's path, turned to ego axes (x forward, z up)
        self.ego_t = self.ego_T = None
        self.ego_source = None
        if self.cameras is not None and "ego_T_world" in self.cameras:
            self.ego_t, self.ego_T = self.cameras["ego_timestamps_us"], self.cameras["ego_T_world"]
            self.ego_source = "ego (rig) poses"
        elif self.cameras is not None:
            first = self.cameras["camera"] == self.cameras["camera"][0]
            order = np.argsort(self.cameras["timestamps_us"][first])
            self.ego_t = self.cameras["timestamps_us"][first][order]
            c2w = self.cameras["c2w"][first][order]
            self.ego_T = c2w.copy()
            self.ego_T[:, :3, :3] = np.stack([c2w[:, :3, 2], -c2w[:, :3, 0], -c2w[:, :3, 1]], -1)
            self.ego_source = f"{self.cameras['camera'][0]} path (re-export for ego poses)"

        # "follow ego" looks through the front camera: the context camera whose optical axis points most along the
        # ego's +x, at its (rigid) pose in the ego frame and its vertical FoV
        self.front = None  # (camera name, T_ego_camera, vertical fov [rad])
        if self.ego_t is not None:
            cams = self.cameras
            first = {str(n): i for i, n in reversed(list(enumerate(cams["camera"])))}  # first frame per camera
            rel = {n: np.linalg.inv(self.ego_pose(int(cams["timestamps_us"][i]))) @ cams["c2w"][i]
                   for n, i in first.items()}
            name = max(rel, key=lambda n: rel[n][0, 2])  # camera z (optical axis) . ego x
            i = first[name]
            self.front = (name, rel[name], 2 * math.atan(0.5 * float(cams["wh"][i][1]) / float(cams["K"][i][1, 1])))

    def ego_pose(self, t_us: int) -> np.ndarray:
        """Ego -> world at ``t_us``: linear in position, slerp in rotation, clamped to the trajectory."""
        t, T = self.ego_t, self.ego_T
        j = int(np.clip(np.searchsorted(t, t_us), 1, len(t) - 1))
        a = float(np.clip((t_us - t[j - 1]) / max(t[j] - t[j - 1], 1), 0.0, 1.0))
        R0, R1 = vtf.SO3.from_matrix(T[j - 1, :3, :3]), vtf.SO3.from_matrix(T[j, :3, :3])
        out = np.eye(4)
        out[:3, :3] = (R0 @ vtf.SO3.exp(a * (R0.inverse() @ R1).log())).as_matrix()
        out[:3, 3] = (1 - a) * T[j - 1, :3, 3] + a * T[j, :3, 3]
        return out


class Renderer:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.rasterization = require_gsplat()
        from gsplat.rendering import RendererConfig_MixedBatch
        self.parallel_batch, self.mixed_batch = _reference_renderer_config(), RendererConfig_MixedBatch()
        self.lock = threading.Lock()  # one render at a time on the GPU
        self.static_keep = torch.ones_like(scene.densities, dtype=torch.bool)
        self.show_movable = True  # the dynamic layer is MOVABLE; the class filter hides it with that class
        self._cache_key, self._cache = None, None  # the Gaussian set of the last render (rebuilt on change only)
        self.palette = torch.zeros(256, 3, device=scene.positions.device)
        for c, rgb in PALETTE.items():
            self.palette[int(c)] = torch.tensor(rgb, dtype=torch.float32, device=self.palette.device) / 255.0

    def set_classes(self, classes: set[int]) -> None:
        keep = torch.zeros(256, dtype=torch.bool, device=self.scene.semantic.device)
        keep[list(classes)] = True
        self.static_keep = keep[self.scene.semantic]
        self._cache_key = None

    def _gaussians(self, semantic: bool, t_us: int, dynamic: bool) -> tuple[torch.Tensor, ...]:
        """The class-filtered static Gaussians plus the dynamic layer at ``t_us``; cached, since gathering
        them (~10 M with OpenScene's dynamic layer) costs more than rasterising them."""
        s = self.scene
        dynamic = dynamic and s.dynamic is not None and self.show_movable
        key = (semantic, t_us if dynamic else None, dynamic)
        if key == self._cache_key:
            return self._cache
        keep = self.static_keep
        means, quats, scales = s.positions[keep], s.rotations[keep], s.scales[keep]
        opacities = s.densities[keep]
        colors = self.palette[s.semantic[keep]] if semantic else s.rgb[keep]
        if dynamic:
            layer = s.dynamic.interpolate(t_us)
            visible = torch.isfinite(layer.positions).all(-1) & (layer.densities[:, 0] > 1e-5)
            means = torch.cat([means, layer.positions[visible]])
            quats = torch.cat([quats, torch.nn.functional.normalize(layer.rotations[visible], dim=-1)])
            scales = torch.cat([scales, layer.scales[visible]])
            opacities = torch.cat([opacities, layer.densities[visible, 0]])
            dyn_colors = (self.palette[int(KelvinSemanticClass.MOVABLE)].expand(int(visible.sum()), 3)
                          if semantic else layer.rgb[visible])
            colors = torch.cat([colors, dyn_colors])
        self._cache_key, self._cache = key, (means, quats, scales, opacities, colors)
        return self._cache

    @torch.inference_mode()
    def render(self, c2w: np.ndarray, K: np.ndarray, wh: tuple[int, int], *, mode: str, t_us: int,
               affine_index: int, dynamic: bool, near: float, with_depth: bool = True,
               ) -> tuple[np.ndarray, np.ndarray | None]:
        """OpenCV camera-to-world ``c2w`` and pinhole ``K`` at ``wh`` -> (uint8 RGB image, z-depth or None)."""
        s, (w, h) = self.scene, wh
        device = s.positions.device
        K = torch.as_tensor(K, dtype=torch.float32, device=device)
        c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=device)
        # calibrated world rays (origin, direction), as the model's renderer takes them
        v, u = torch.meshgrid(torch.arange(h, device=device) + 0.5, torch.arange(w, device=device) + 0.5,
                              indexing="ij")
        dirs_cam = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)
        dirs = torch.nn.functional.normalize(dirs_cam @ c2w_t[:3, :3].T, dim=-1)
        rays = torch.cat([c2w_t[:3, 3].expand(h, w, 3), dirs], -1).contiguous()

        with self.lock:
            means, quats, scales, opacities, colors = self._gaussians(mode == "semantic", t_us, dynamic)
            raster = lambda config: self.rasterization(  # noqa: E731
                means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
                viewmats=torch.linalg.inv(c2w_t)[None], Ks=K[None], width=w, height=h,
                near_plane=near, far_plane=torch.finfo(torch.float32).max, sh_degree=None, render_mode="RGB-d",
                packed=False, with_ut=True, ut_params=_reference_ut_parameters(),
                renderer_config=config, with_eval3d=True, global_z_order=False,
                rays=rays[None], camera_model="pinhole",
            )
            try:
                out, alpha, _ = raster(self.parallel_batch)
            except RuntimeError as e:
                # ParallelBatch (the preview renderer's) counts a tile's Gaussian batches in uint16: a view where
                # one tile meets > ~2 M Gaussians fails. MixedBatch composites serially per tile, with no cap.
                if "max batches per tile" not in str(e):
                    raise
                out, alpha, _ = raster(self.mixed_batch)
        fg, alpha = out[0, ..., :3], alpha[0]
        distance = out[0, ..., 3:4] / alpha.clamp_min(1e-6)
        if mode == "rgb" or mode == "rgb (no sky)":
            image = fg
            if mode == "rgb" and s.sky is not None:
                image = image + (1.0 - alpha) * sample_sky_cubemap(s.sky, dirs)
            if s.affine is not None and affine_index >= 0:
                affine = s.affine[affine_index]
                image = image @ affine[:, :3].T + affine[:, 3]
        elif mode == "semantic":
            image = fg + (1.0 - alpha) * 0.08
        elif mode == "depth":
            d = distance[..., 0].clamp(0.5, 200.0).log()
            d = (d - math.log(0.5)) / (math.log(200.0) - math.log(0.5))
            image = _turbo(1.0 - d) * (alpha > 0.5)
        else:  # opacity
            image = alpha.expand(h, w, 3)
        rgb = (image.clamp(0, 1) * 255).byte().cpu().numpy()
        if not with_depth:
            return rgb, None
        # z-depth for viser to composite the frustums; background far away
        z = distance[..., 0] * dirs_cam[..., 2] / dirs_cam.norm(dim=-1)
        z = torch.where(alpha[..., 0] > 0.5, z, torch.full_like(z, 1e4))
        return rgb, z.cpu().numpy()


class TrackEditor:
    """Move, turn or delete dynamic objects through their index boxes, live (``tools/track_edit.py``, Level 1 of
    ``docs/world-model.md``): the boxes at the slider time, click one (or pick it) to get a gizmo; dragging it or
    setting offset / yaw (box frame) re-poses that track's Gaussians in the rendered dynamic layer."""

    COLORS = {"box": (0, 200, 255), "edited": (255, 140, 0), "deleted": (255, 40, 40), "selected": (255, 230, 0)}

    def __init__(self, viewer: Viewer, ply: Path, index_path: Path, padding: tuple[float, float, float]):
        import pickle

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import track_edit as te
        from instant_nurec.utils.motion import cuboid_track_time_bounds_us

        s = viewer.scene
        if s.dynamic is None or s.cameras is None or "world_origin" not in s.cameras:
            raise SystemExit("--index needs the export's dynamic layer and .cameras.npz (an --index-path run)")
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        if "tracks" not in index:
            raise SystemExit(f"{index_path} has no 'tracks'")
        self.te, self.viewer, self.ply = te, viewer, ply
        d = s.dynamic
        origin = s.cameras["world_origin"].astype(np.float64)
        self.tracks, self.labels = te.load_tracks(index, origin, d.rotations.device)
        self.base = (d.keyframe_positions.clone(), d.rotations.clone(), d.max_densities.clone())
        print(f"Associating {len(d):,} dynamic Gaussians with {self.tracks.n_tracks} boxes ...", flush=True)
        self.idx = te.associate(d.keyframe_positions[:, 1], d.keyframe_timestamps_us[:, 1], self.tracks, padding)
        counts = torch.bincount(self.idx[self.idx >= 0], minlength=self.tracks.n_tracks).cpu().numpy()
        self.active = [int(k) for k in np.argsort(-counts) if counts[k] > 0]
        self.members = {k: torch.where(self.idx == k)[0] for k in self.active}
        all_idx = torch.arange(self.tracks.n_tracks, device=self.idx.device, dtype=self.tracks.tracks_packinfo.dtype)
        self.bounds = cuboid_track_time_bounds_us(self.tracks, all_idx).cpu().numpy()
        self.dims = self.tracks.cuboids_dims.cpu().numpy()
        self.names = {f"{self.tracks.tracks_id[k]} {self.labels[k]} ({counts[k]:,})": k for k in self.active}
        self.edits: dict[int, dict] = {}
        self.current: int | None = None
        self.basis: dict[int, tuple] = {}  # per track: repose_basis, built on its first edit

        gui, scene = viewer.server.gui, viewer.server.scene
        with gui.add_folder("edit objects"):
            gui.add_markdown(f"{counts.sum() / max(len(self.idx), 1):.1%} of the dynamic Gaussians in "
                             f"{len(self.active)} boxes. Click a box; offset and yaw are in its frame (x along the "
                             "length).")
            self.show = gui.add_checkbox("boxes", initial_value=True)
            self.pick = gui.add_dropdown("object", ("(none)", *self.names), initial_value="(none)")
            self.offset = gui.add_vector3("offset [m]", (0.0, 0.0, 0.0), step=0.1, disabled=True)
            self.yaw = gui.add_number("yaw [deg]", 0.0, step=1.0, disabled=True)
            self.delete = gui.add_checkbox("delete", initial_value=False, disabled=True)
            self.reset = gui.add_button("reset object", disabled=True)
            self.out = gui.add_text("save to", initial_value=str(ply.parent / "edited"))
            save = gui.add_button("save edits")
            self.status = gui.add_markdown("")
        self.boxes = {k: self._add_box(k) for k in self.active}
        self.gizmo = scene.add_transform_controls("/objects/gizmo", scale=2.5, disable_sliders=True, visible=False)
        self.gizmo.on_update(self._on_gizmo)
        self.show.on_update(lambda _: self.update())
        self.pick.on_update(lambda _: self.select(self.names.get(self.pick.value)))
        for handle in (self.offset, self.yaw, self.delete):
            handle.on_update(self._on_field)
        self.reset.on_click(lambda _: self._set_edit(self.current, {}))
        save.on_click(lambda _: self.save())
        viewer.time.on_update(lambda _: self.update())
        self.update()

    def _add_box(self, k: int):
        state = ("selected" if k == self.current else "deleted" if self.edits.get(k, {}).get("delete")
                 else "edited" if k in self.edits else "box")
        handle = self.viewer.server.scene.add_box(
            f"/objects/{self.tracks.tracks_id[k]}", color=self.COLORS[state], dimensions=tuple(self.dims[k]),
            wireframe=True, visible=False)
        handle.on_click(lambda _, k=k: setattr(self.pick, "value", self._name(k)))
        return handle

    def _name(self, k: int) -> str:
        return next(name for name, j in self.names.items() if j == k)

    def _t_us(self) -> int:
        return self.viewer.scene.t0 + int(self.viewer.time.value * 1e6)

    def _poses(self, ks: list[int], edited: bool = True) -> np.ndarray:
        """The boxes of tracks ``ks`` -> world at the slider time (n, 4, 4), with their edits unless ``edited`` is
        off; one batched interpolation."""
        dev = self.idx.device
        idx = torch.tensor(ks, dtype=self.tracks.tracks_packinfo.dtype, device=dev)
        T = self.te.track_poses(self.tracks, torch.full((len(ks),), self._t_us(), device=dev), idx)
        data = T.data.cpu().numpy()
        out = np.tile(np.eye(4), (len(ks), 1, 1))
        out[:, :3, :3] = vtf.SO3(data[:, [6, 3, 4, 5]]).as_matrix()
        out[:, :3, 3] = data[:, :3]
        for i, k in enumerate(ks):
            if edited and k in self.edits and not self.edits[k].get("delete"):
                E = np.eye(4)
                E[:3, :3] = vtf.SO3.from_z_radians(math.radians(self.edits[k]["yaw_deg"])).as_matrix()
                E[:3, 3] = self.edits[k]["offset_m"]
                out[i] = out[i] @ E
        return out

    def _pose(self, k: int, edited: bool = True) -> np.ndarray:
        return self._poses([k], edited)[0]

    def update(self) -> None:
        """Boxes of the tracks annotated around the slider time at their (edited) poses; the gizmo on the selected."""
        t = self._t_us()
        shown = [k for k in self.boxes if self.show.value and self.bounds[k, 0] <= t <= self.bounds[k, 1]]
        poses = dict(zip(shown, self._poses(shown))) if shown else {}
        with self.viewer.server.atomic():
            for k, handle in self.boxes.items():
                if k in poses:
                    handle.wxyz, handle.position = vtf.SO3.from_matrix(poses[k][:3, :3]).wxyz, poses[k][:3, 3]
                handle.visible = k in poses
            if self.current is not None:
                T = self._pose(self.current)
                self.gizmo.wxyz, self.gizmo.position = vtf.SO3.from_matrix(T[:3, :3]).wxyz, T[:3, 3]
            self.gizmo.visible = self.current is not None

    def select(self, k: int | None) -> None:
        previous, self.current = self.current, k
        for j in {previous, k} - {None}:
            self.boxes[j].remove()
            self.boxes[j] = self._add_box(j)
        edit = self.edits.get(k, {}) if k is not None else {}
        self.offset.value = tuple(edit.get("offset_m", (0.0, 0.0, 0.0)))  # client=None: no re-apply
        self.yaw.value = float(edit.get("yaw_deg", 0.0))
        self.delete.value = bool(edit.get("delete", False))
        for handle in (self.offset, self.yaw, self.delete, self.reset):
            handle.disabled = k is None
        if k is not None:
            start, end = (self.bounds[k] - self.viewer.scene.t0) / 1e6
            self.status.content = f"{len(self.members[k]):,} Gaussians, annotated {start:.1f} – {end:.1f} s"
        self.update()

    def _on_field(self, event) -> None:
        if event.client is None or self.current is None:  # set by the server (select / gizmo), already applied
            return
        edit = {"delete": True} if self.delete.value else {"offset_m": [float(v) for v in self.offset.value],
                                                             "yaw_deg": float(self.yaw.value)}
        self._set_edit(self.current, edit)

    def _on_gizmo(self, event) -> None:
        if self.current is None or self.delete.value:
            return
        T_box = self._pose(self.current, edited=False)
        T_gizmo = np.eye(4)
        T_gizmo[:3, :3], T_gizmo[:3, 3] = vtf.SO3(np.asarray(event.target.wxyz)).as_matrix(), event.target.position
        E = np.linalg.inv(T_box) @ T_gizmo  # the edit in the box frame; roll and pitch are dropped
        edit = {"offset_m": [round(float(v), 3) for v in E[:3, 3]],
                "yaw_deg": round(math.degrees(math.atan2(E[1, 0], E[0, 0])), 2)}
        self.offset.value, self.yaw.value = tuple(edit["offset_m"]), edit["yaw_deg"]
        self._set_edit(self.current, edit, move_gizmo=event.phase == "end")

    def _set_edit(self, k: int | None, edit: dict, move_gizmo: bool = True) -> None:
        """Re-pose (or hide) track ``k``'s Gaussians in the rendered dynamic layer from the unedited ones."""
        if k is None:
            return
        identity = not edit.get("delete") and not any(edit.get("offset_m", ())) and not edit.get("yaw_deg")
        if identity:
            self.edits.pop(k, None)
        else:
            self.edits[k] = edit
        if not edit and self.current == k:  # reset
            self.select(k)
        sel, d = self.members[k], self.viewer.scene.dynamic
        positions, rotations, densities = self.base
        with self.viewer.renderer.lock:
            d.max_densities[sel] = 0.0 if edit.get("delete") else densities[sel]
            if identity or edit.get("delete"):
                d.keyframe_positions[sel], d.rotations[sel] = positions[sel], rotations[sel]
            else:
                if k not in self.basis:
                    self.basis[k] = self.te.repose_basis(positions[sel], d.keyframe_timestamps_us[sel], self.tracks, k)
                d.keyframe_positions[sel], d.rotations[sel] = self.te.repose_apply(self.basis[k], rotations[sel], edit)
            self.viewer.renderer._cache_key = None
        self.viewer.mark_all()
        if move_gizmo:
            self.update()
        else:  # mid-drag: move the box only, the gizmo is the user's
            T = self._pose(k)
            self.boxes[k].wxyz, self.boxes[k].position = vtf.SO3.from_matrix(T[:3, :3]).wxyz, T[:3, 3]

    def save(self) -> None:
        """The edits as ``edits.json`` and the edited scene (``track_edit.py --edits``) in the "save to" folder."""
        import json

        out = Path(self.out.value).expanduser()
        dynamic = dict(np.load(self.ply.with_suffix(".dynamic.npz")))
        path = self.te.write_edited(self.ply, self.te.edit_dynamic(dynamic, self.tracks, self.idx, self.edits), out)
        edits = {self.tracks.tracks_id[k]: v for k, v in self.edits.items()}
        (out / "edits.json").write_text(json.dumps(edits, indent=1))
        self.status.content = f"saved {len(edits)} edit(s): `{path}`"


def _turbo(x: torch.Tensor) -> torch.Tensor:
    """Google turbo colormap (polynomial fit), x in [0, 1] -> RGB."""
    x = x.clamp(0, 1)[..., None]
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return torch.cat([r, g, b], -1).clamp(0, 1)


class Viewer:
    def __init__(self, scene: Scene, renderer: Renderer, port: int, host: str):
        self.scene, self.renderer = scene, renderer
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self.clients: dict[int, dict] = {}
        gui = self.server.gui
        n_static = len(scene.positions)
        n_dynamic = 0 if scene.dynamic is None else len(scene.dynamic)
        gui.add_markdown(f"**{n_static:,}** static, **{n_dynamic:,}** dynamic Gaussians  \n"
                         f"classes from {scene.semantic_source}")
        self.mode = gui.add_dropdown("render", ("rgb", "rgb (no sky)", "semantic", "depth", "opacity"), initial_value="rgb")
        self.max_res = gui.add_slider("max resolution", min=256, max=2560, step=64, initial_value=1280)
        self.preview_res = gui.add_slider("moving resolution", min=128, max=1280, step=64, initial_value=480)
        self.jpeg = gui.add_slider("jpeg quality", min=50, max=100, step=5, initial_value=85)
        self.near = gui.add_slider("near plane [m]", min=0.05, max=5.0, step=0.05, initial_value=0.2)
        self.affine = gui.add_dropdown("ISP camera", tuple(scene.affine_names), initial_value=scene.affine_names[0])
        self.dynamic = gui.add_checkbox("dynamic layer", initial_value=n_dynamic > 0, disabled=n_dynamic == 0)
        span = max((scene.t1 - scene.t0) / 1e6, 0.05)
        timed = n_dynamic > 0 or scene.ego_t is not None
        self.time = gui.add_slider("time [s]", min=0.0, max=span, step=0.05, initial_value=0.0, disabled=not timed)
        self.play = gui.add_checkbox("play", initial_value=False, disabled=not timed)
        self.follow = gui.add_checkbox(
            "follow ego", initial_value=False, disabled=scene.front is None,
            hint=None if scene.front is None else f"look through the ego's front camera ({scene.front[0]}) at the "
            "slider time, as play or the slider moves it",
        )
        # the dynamic layer is MOVABLE, so MOVABLE gets a box whenever there is one, even with no static MOVABLE
        # Gaussians (AV2 / nuScenes / Waymo send them all to the dynamic layer)
        movable = int(KelvinSemanticClass.MOVABLE)
        classes = sorted(set(scene.classes) | ({movable} if n_dynamic > 0 else set()))
        with gui.add_folder("classes"):
            self.class_boxes = {
                c: gui.add_checkbox(
                    (f"{c} {KelvinSemanticClass(c).name}" if c in KelvinSemanticClass._value2member_map_ else str(c))
                    + (" (+ dynamic layer)" if c == movable and n_dynamic > 0 else ""),
                    initial_value=True,
                )
                for c in classes
            }
        for handle in (self.mode, self.affine, self.dynamic, self.time, self.near):
            handle.on_update(lambda _: self.mark_all())
        for box in self.class_boxes.values():
            box.on_update(lambda _: self.update_classes())
        self.show_cameras = self.occlude = None
        if scene.cameras is not None:
            self.show_cameras = gui.add_checkbox("context cameras", initial_value=True)
            self.show_cameras.on_update(lambda _: self._set_frustums_visible(self.show_cameras.value))
            # The occlusion depth goes as a lossless PNG, ~20x the JPEG (2 MB at 1280x720), so it is off by
            # default and, when on, sent with the refined frame only.
            self.occlude = gui.add_checkbox("frustums behind geometry", initial_value=False)
            self.occlude.on_update(lambda _: self.mark_all())
            self._add_frustums()
        self.show_ego = None
        if scene.ego_t is not None:
            self.show_ego = gui.add_checkbox("ego trajectory", initial_value=True, hint=scene.ego_source)
            self.show_ego.on_update(lambda _: self._set_ego_visible(self.show_ego.value))
            self._add_ego()
            self.time.on_update(lambda _: self.update_ego())
        if scene.front is not None:
            self.follow.on_update(lambda _: self.follow_ego(set_isp=True))
            self.time.on_update(lambda _: self.follow_ego())
        self.stats = gui.add_text("last frame", initial_value="", disabled=True)
        self.server.on_client_connect(self.on_connect)
        self.server.on_client_disconnect(lambda client: self.clients.pop(client.client_id, None))

    # ------------------------------------------------------------------ scene objects
    def _add_frustums(self) -> None:
        cams = self.scene.cameras
        names = list(dict.fromkeys(str(n) for n in cams["camera"]))
        self.frustums = []
        for i, (c2w, K, wh, name) in enumerate(zip(cams["c2w"], cams["K"], cams["wh"], cams["camera"])):
            w, h = (int(x) for x in wh)
            handle = self.server.scene.add_camera_frustum(
                f"/cameras/{name}/{i:04d}", fov=2 * math.atan(0.5 * h / K[1, 1]), aspect=w / h, scale=0.4,
                color=CAMERA_COLORS[names.index(str(name)) % len(CAMERA_COLORS)],
                wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz, position=c2w[:3, 3],
            )
            handle.on_click(lambda event, i=i: self.snap(event.client, i))
            self.frustums.append(handle)

    EGO_BOX = (4.6, 1.9, 1.6)  # nominal car, l / w / h [m]
    EGO_BOX_CENTRE = (1.4, 0.0, 0.8)  # in the ego frame (origin ~ rear axle on the ground, x forward, z up)
    PAST, FUTURE = (255, 140, 0), (150, 150, 150)

    def _add_ego(self) -> None:
        """The trajectory as a line (traversed part highlighted) and the ego at the current time as a box + axes."""
        s = self.scene
        keep = np.unique(np.linspace(0, len(s.ego_t) - 1, min(len(s.ego_t), 1500)).round().astype(int))
        self.ego_line_t = s.ego_t[keep]
        p = s.ego_T[keep, :3, 3] + np.array([0.0, 0.0, 0.05])  # just above the road
        self.ego_line = self.server.scene.add_line_segments(
            "/ego/trajectory", points=np.stack([p[:-1], p[1:]], 1),
            colors=np.tile(np.array(self.FUTURE, np.uint8), (len(p) - 1, 2, 1)), thickness=0.15,
        )
        self.ego_frame = self.server.scene.add_frame("/ego/current", axes_length=2.5, axes_radius=0.08)
        self.server.scene.add_box("/ego/current/box", color=(255, 60, 60), dimensions=self.EGO_BOX,
                                  position=self.EGO_BOX_CENTRE, opacity=0.6)
        self.update_ego()

    def update_ego(self) -> None:
        """Move the ego marker to the time slider and colour the traversed part of the trajectory."""
        if self.show_ego is None:
            return
        t_us = self.scene.t0 + int(self.time.value * 1e6)
        T = self.scene.ego_pose(t_us)
        self.ego_frame.wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
        self.ego_frame.position = T[:3, 3]
        past = self.ego_line_t[1:] <= t_us
        colors = np.where(past[:, None, None], np.array(self.PAST, np.uint8), np.array(self.FUTURE, np.uint8))
        self.ego_line.colors = np.ascontiguousarray(np.broadcast_to(colors, (len(past), 2, 3)))

    def follow_ego(self, set_isp: bool = False) -> None:
        """With "follow ego" on, put every client's view at the ego's front camera at the slider time (its pose
        and vertical FoV); ``set_isp`` (on switching it on) also selects that camera's ISP affine."""
        if not self.follow.value:
            return
        name, T_ego_cam, fov = self.scene.front
        if set_isp and name in self.scene.affine_names:
            self.affine.value = name
        c2w = self.scene.ego_pose(self.scene.t0 + int(self.time.value * 1e6)) @ T_ego_cam
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.fov = fov
                client.camera.wxyz = vtf.SO3.from_matrix(c2w[:3, :3]).wxyz
                client.camera.position = c2w[:3, 3]

    def _set_ego_visible(self, visible: bool) -> None:
        self.ego_line.visible = visible
        self.ego_frame.visible = visible

    def _set_frustums_visible(self, visible: bool) -> None:
        for handle in self.frustums:
            handle.visible = visible

    def snap(self, client: viser.ClientHandle, i: int) -> None:
        """Look through context frame ``i``: its pose, vertical FoV, time and ISP camera."""
        cams = self.scene.cameras
        c2w, K, wh = cams["c2w"][i], cams["K"][i], cams["wh"][i]
        # time first: with "follow ego" on, the time change moves the views to the front camera, and this one then
        # lands on the clicked frame (until the next time change)
        self.time.value = min(max((int(cams["timestamps_us"][i]) - self.scene.t0) / 1e6, 0.0), self.time.max)
        if str(cams["camera"][i]) in self.scene.affine_names:
            self.affine.value = str(cams["camera"][i])
        with client.atomic():
            client.camera.fov = 2 * math.atan(0.5 * float(wh[1]) / float(K[1, 1]))
            client.camera.wxyz = vtf.SO3.from_matrix(c2w[:3, :3]).wxyz
            client.camera.position = c2w[:3, 3]

    def update_classes(self) -> None:
        on = {c for c, box in self.class_boxes.items() if box.value}
        self.renderer.set_classes(on)
        movable = self.class_boxes.get(int(KelvinSemanticClass.MOVABLE))
        self.renderer.show_movable = movable is None or movable.value
        self.mark_all()

    # ------------------------------------------------------------------ clients
    def mark_all(self) -> None:
        for state in self.clients.values():
            state["dirty"] = True

    def on_connect(self, client: viser.ClientHandle) -> None:
        state = {"dirty": True, "moved": time.time(), "refine": False}
        self.clients[client.client_id] = state

        @client.camera.on_update
        def _(_camera) -> None:
            state["dirty"], state["moved"] = True, time.time()

        if self.scene.cameras is not None:
            self.snap(client, 0)
        else:  # above the scene centre, looking down the road
            centre = self.scene.positions.median(0).values.cpu().numpy()
            client.camera.position = centre + np.array([-30.0, 0.0, 20.0])
            client.camera.look_at = centre
        threading.Thread(target=self._loop, args=(client, state), daemon=True).start()

    def _loop(self, client: viser.ClientHandle, state: dict) -> None:
        while client.client_id in self.clients:
            if state["dirty"]:
                state["dirty"], state["refine"] = False, True
                self._render(client, self.preview_res.value, refined=False)
            elif state["refine"] and time.time() - state["moved"] > 0.2:
                state["refine"] = False
                self._render(client, self.max_res.value, refined=True)
            else:
                time.sleep(0.01)

    def _render(self, client: viser.ClientHandle, res: int, refined: bool) -> None:
        camera = client.camera
        aspect = camera.aspect
        w, h = (res, max(2, round(res / aspect))) if aspect >= 1 else (max(2, round(res * aspect)), res)
        c2w = np.eye(4)
        c2w[:3, :3] = vtf.SO3(camera.wxyz).as_matrix()
        c2w[:3, 3] = camera.position
        f = 0.5 * h / math.tan(0.5 * camera.fov)
        K = np.array([[f, 0.0, 0.5 * w], [0.0, f, 0.5 * h], [0.0, 0.0, 1.0]])
        start = time.perf_counter()
        image, z = self.renderer.render(
            c2w, K, (w, h), mode=self.mode.value, t_us=self.scene.t0 + int(self.time.value * 1e6),
            affine_index=self.scene.affine_names.index(self.affine.value) if self.scene.affine is not None else -1,
            dynamic=self.dynamic.value, near=self.near.value,
            with_depth=refined and self.occlude is not None and self.occlude.value,
        )
        rendered = time.perf_counter()
        client.scene.set_background_image(image, format="jpeg", jpeg_quality=int(self.jpeg.value), depth=z)
        if refined:
            self.stats.value = (f"{w}x{h}: render {1e3 * (rendered - start):.0f} ms, encode "
                                f"{1e3 * (time.perf_counter() - rendered):.0f} ms{' + depth' if z is not None else ''}")

    def run(self) -> None:
        while True:
            if self.play.value and not self.play.disabled:
                self.time.value = (self.time.value + 0.1) % (self.time.max + 1e-6)  # 0.1 s per tick
                time.sleep(0.1)
            else:
                time.sleep(0.2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", type=Path, help="a PLY written by run_inference.py (merged or one chunk)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (shared machine): reach it through ssh -L; "
                    "0.0.0.0 serves every network interface")
    ap.add_argument("--index", type=Path, help="the log's index pickle (with 'tracks'): adds 'edit objects', moving, "
                    "turning or deleting dynamic objects by their boxes (tools/track_edit.py)")
    ap.add_argument("--padding", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("X", "Y", "Z"),
                    help="with --index: box padding of the first association pass (run_inference.py --cuboid-padding)")
    args = ap.parse_args()
    device = torch.device("cuda")
    t = time.time()
    scene = Scene(args.ply, device)
    n_dynamic = 0 if scene.dynamic is None else len(scene.dynamic)
    n_cameras = 0 if scene.cameras is None else len(scene.cameras["c2w"])
    print(f"Loaded {len(scene.positions):,} static + {n_dynamic:,} dynamic Gaussians, {n_cameras} context cameras "
          f"in {time.time() - t:.1f} s", flush=True)
    viewer = Viewer(scene, Renderer(scene), args.port, args.host)
    if args.index is not None:
        TrackEditor(viewer, args.ply, args.index, tuple(args.padding))
    viewer.run()


if __name__ == "__main__":
    main()
