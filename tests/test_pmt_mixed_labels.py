# SPDX-License-Identifier: Apache-2.0
"""PMT index cameras with lens distortion / a separate depth grid (Waymo), and batches without masks."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import torch

from instant_nurec.config_schema.train import TrainingLossConfig
from instant_nurec.datasets.pmt_index import resample_depth_grid, undistort_image
from instant_nurec.model.supervision import SupervisionPack
from instant_nurec.training.losses import TrainingLosses
from instant_nurec.training.renderer import RenderOutput
from instant_nurec.utils.batch import CameraFrameLabels, DataAndRenderingBatch, DataBatch, FrameMeta
from instant_nurec.utils.types import RayFlags

K_RAW = np.array([[400.0, 0, 322.0], [0, 400.0, 238.0], [0, 0, 1]])
D = np.array([0.03, -0.33, 0.001, 0.0007, 0.0])  # Waymo-like barrel distortion
WH_RAW = (640, 480)


def test_undistort_puts_each_pinhole_pixel_where_the_distorted_camera_sees_it() -> None:
    K, _ = cv2.getOptimalNewCameraMatrix(K_RAW, D, WH_RAW, alpha=0)
    u, v = np.meshgrid(np.arange(WH_RAW[0], dtype=np.float32), np.arange(WH_RAW[1], dtype=np.float32))
    raw = np.stack([u, v, np.zeros_like(u)], axis=-1)  # each raw pixel holds its own coordinates

    out = undistort_image(raw, {"K": K_RAW, "D": D, "wh": WH_RAW}, K, WH_RAW)

    for x, y in [(30, 40), (320, 240), (600, 450), (100, 400)]:
        ray = np.array([[(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], 1.0]])
        expected, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), K_RAW, D)
        np.testing.assert_allclose(out[y, x, :2], expected.ravel(), atol=0.1)


def test_masks_undistort_by_nearest_onto_the_image_grid() -> None:
    K, _ = cv2.getOptimalNewCameraMatrix(K_RAW, D, WH_RAW, alpha=0)
    distortion = {"K": K_RAW, "D": D, "wh": WH_RAW}
    u, v = np.meshgrid(np.arange(WH_RAW[0]), np.arange(WH_RAW[1]))
    classes = ((u // 40 + v // 40) % 5).astype(np.uint8) * 50  # blocks of labels 0, 50, ..., 200

    out = undistort_image(classes, distortion, K, WH_RAW, cv2.INTER_NEAREST, 255)
    # a mask saved at half resolution lands on the same grid
    half = cv2.resize(classes, (WH_RAW[0] // 2, WH_RAW[1] // 2), interpolation=cv2.INTER_NEAREST)
    out_half = undistort_image(half, distortion, K, WH_RAW, cv2.INTER_NEAREST, 255)
    # the image carrying the same blocks, undistorted bilinearly
    image = undistort_image(classes.astype(np.float32), distortion, K, WH_RAW)

    assert set(np.unique(out)) <= {0, 50, 100, 150, 200}  # no blended labels
    kernel = np.ones((3, 3), np.uint8)
    interior = cv2.erode(image, kernel) == cv2.dilate(image, kernel)  # away from block edges
    assert interior.mean() > 0.8 and (out[interior] == np.rint(image[interior])).all()
    assert (out_half == out).mean() > 0.95


def test_depth_grid_resample_matches_the_target_grid_and_zeros_outside() -> None:
    # DVGT-style grid: a centred crop of the upper image, squeezed anisotropically.
    depth_K = np.array([[120.0, 0, 60.0], [0, 300.0, 40.0], [0, 0, 1]])
    depth_wh = (121, 81)
    K, wh = np.array([[400.0, 0, 320.0], [0, 400.0, 100.0], [0, 0, 1]]), (640, 480)
    xs = (np.arange(depth_wh[0]) - depth_K[0, 2]) / depth_K[0, 0]
    depth = np.broadcast_to(5.0 + xs[None, :], depth_wh[::-1]).astype(np.float32)  # encodes x / z

    out = resample_depth_grid(depth, depth_K, K, wh)

    x_target = (np.arange(wh[0]) - K[0, 2]) / K[0, 0]
    row = out[100]
    inside = row > 0
    assert inside[320] and not inside[0] and not inside[-1]
    np.testing.assert_allclose(row[inside], 5.0 + x_target[inside], atol=0.5 / depth_K[0, 0])
    assert (out[-1] == 0).all()  # below the depth crop: no label


def _batch(labels: CameraFrameLabels) -> DataAndRenderingBatch:
    return DataAndRenderingBatch(
        data=DataBatch(camera=DataBatch.Camera(meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)],
                                               labels=labels)),
        rendering=SimpleNamespace(camera=SimpleNamespace(distance_to_depth_scale=torch.ones_like(labels.flags))),
    )


def _config(**weights) -> TrainingLossConfig:
    zero = dict.fromkeys(("primitive_sky_cubemap", "primitive_rgb", "primitive_distance",
                          "primitive_distance_gradient", "primitive_semantics", "primitive_normal",
                          "primitive_velocity"), 0)
    return TrainingLossConfig(**{**zero, **weights})


def test_semantics_without_any_mask_is_skipped_not_nan() -> None:
    labels = CameraFrameLabels(flags=torch.full((1, 2, 2, 1), int(RayFlags.RGB_LABEL), dtype=torch.int32),
                               semantic_class=torch.full((1, 2, 2, 1), 255, dtype=torch.uint8))
    logits = torch.zeros((1, 2, 2, 8), requires_grad=True)

    values, skipped = TrainingLosses(_config(primitive_semantics=1.0))._context_losses(
        SupervisionPack(context_semantic_logits=logits), _batch(labels))

    assert "primitive_semantics" not in values and skipped == ["primitive_semantics"]


def test_background_can_be_limited_to_masked_pixels() -> None:
    masked = int(RayFlags.VALID_SEMANTIC)
    flags = torch.tensor([[[[masked | int(RayFlags.SKY_SEMANTIC)], [masked], [0]]]], dtype=torch.int32)
    output = RenderOutput(rgb=torch.zeros((1, 1, 3, 3)), opacity=torch.tensor([[[[0.0], [1.0], [0.0]]]]),
                          distance=torch.zeros((1, 1, 3, 1)), sky_rgb=torch.zeros((1, 1, 3, 3)))

    def background(**options) -> torch.Tensor:
        losses = TrainingLosses(_config(background=1.0, **options))
        return losses._render_losses(output, _batch(CameraFrameLabels(flags=flags)))[0]["background"]

    # default (upstream): the unmasked pixel counts as foreground with opacity 0 -> error 1 of 3 pixels
    torch.testing.assert_close(background(), torch.tensor(1 / 3))
    torch.testing.assert_close(background(background_requires_semantics=True), torch.tensor(0.0))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
