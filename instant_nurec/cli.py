# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone argparse CLI.

``--ncore-path`` accepts either a single ``.json`` ncorev4 sequence
metadata file (NuRec-aligned) or a ``.lst`` manifest listing one JSON
path per line (each absolute or relative-to-the-LST-file's directory).

``--index-path`` (this branch) instead takes PseudoMapTrainer index pickles
(AV2 / nuScenes / Waymo / OpenScene logs, ``PMTIndexDataset``), each log
chunked and exported as an NCore sequence is. ``--checkpoint`` runs a
fine-tune's weights instead of the released ones.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from instant_nurec.pretrained import (
    DEFAULT_MODEL_VARIANT,
    MODEL_VARIANTS,
    get_model_profile,
)

_DEFAULT_MAX_CHUNKS = 8
_DEFAULT_N_GAUSSIANS = 2_000_000


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instant_nurec",
        description="Standalone Kelvin predict-mode CLI.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_VARIANTS,
        default=DEFAULT_MODEL_VARIANT,
        help=(
            "Released model profile. pa-front uses one 784x448 camera with 18 frames; "
            "pa-multiview uses three 504x280 cameras with 18 frames each by default; "
            "pq-front uses the selective point-query decoder with 18 front-camera "
            "frames. "
            f"Default: {DEFAULT_MODEL_VARIANT}."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--ncore-path",
        type=Path,
        help=(
            "ncorev4 input. Either a single sequence ``.json`` (NuRec-aligned) "
            "or a ``.lst`` manifest with one JSON path per line "
            "(absolute or relative-to-the-LST-file's directory; "
            "``#``-prefixed and blank lines skipped)."
        ),
    )
    source.add_argument(
        "--index-path",
        type=Path,
        action="append",
        help=(
            "PseudoMapTrainer index input instead of NCore (AV2 / nuScenes / Waymo / OpenScene): a log's "
            "``.pkl`` or a directory of them; repeatable. Context cameras default to all of the index's "
            "cameras in its order (the fine-tune's order; any count). OpenScene boxes become cuboid tracks."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Weights to run instead of the released ones: a weights-only .pth or a fine-tune's Lightning "
            ".ckpt (e.g. checkpoints/training/<run>/checkpoints/last.ckpt). The semantic head width "
            "(5 or 8) is read from the weights."
        ),
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help=(
            "--index-path only: frames per camera per chunk (default: the profile's, 18). The PMT "
            "fine-tunes trained with 8."
        ),
    )
    parser.add_argument(
        "--max-frame-gap-us",
        type=int,
        default=750_000,
        help=(
            "--index-path only: max gap between a chunk's frames (default 750000, as NCore predict). "
            "The PMT fine-tunes trained with 500000."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for PLY and sky-output bundles.",
    )
    parser.add_argument(
        "--include-dynamic",
        action="store_true",
        help="Retain dynamic Gaussians and write .dynamic.npz motion keyframes (PA profiles only).",
    )
    parser.add_argument(
        "--motion-source",
        choices=("tracks", "learned"),
        default="tracks",
        help="Dynamic motion source: NCore cuboid tracks (default) or learned motion. Learned requires --include-dynamic.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "If set, merge per-chunk primitives into a single "
            "frustum-ownership PLY per sequence and run kl-optimal "
            "voxelization (target count controlled by --n-gaussians). "
            "Default (flag absent) writes per-chunk PLYs and skips "
            "voxelization entirely."
        ),
    )
    parser.add_argument(
        "--n-gaussians",
        type=int,
        default=_DEFAULT_N_GAUSSIANS,
        help=(
            f"Target number of static Gaussians after kl-optimal voxelization "
            f"(default: {_DEFAULT_N_GAUSSIANS}). The voxel size is searched "
            f"iteratively via bracketed binary search (starting at 0.1) until "
            f"the count lands in [0.9 * target, target]. Only consulted when "
            f"--merge is set (voxelization is bundled with merge); must be > 0."
        ),
    )
    parser.add_argument(
        "--camera-id",
        dest="camera_ids",
        metavar="CAMERA_ID",
        type=str,
        action="append",
        default=None,
        help=(
            "Override a profile's ncorev4 context camera. Repeat the flag to "
            "provide multiple cameras in canonical order. The selected profile "
            "validates its camera contract. pq-front is fixed to "
            "camera_front_wide_120fov. If omitted, the profile's default camera "
            "set is used."
        ),
    )
    parser.add_argument(
        "--track-label-source",
        choices=["AUTOLABEL", "EXTERNAL", "GT_SYNTHETIC", "GT_ANNOTATION"],
        default="AUTOLABEL",
        help=(
            "NCore cuboid-track label source used for dynamic-object matching. "
            "Default: AUTOLABEL."
        ),
    )
    parser.add_argument(
        "--cuboid-padding",
        metavar=("X", "Y", "Z"),
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help=(
            "Padding in meters added to cuboid-track dimensions along x, y, z "
            "during dynamic-object matching. Default: 1 1 1."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=_DEFAULT_MAX_CHUNKS,
        help=(
            f"Maximum number of time-chunks processed per clip "
            f"(default: {_DEFAULT_MAX_CHUNKS}). One chunk spans up to 13.5 s. "
            f"Longer clips are "
            f"truncated and a WARNING is logged naming the dropped chunk "
            f"count and the --max-chunks value needed to cover the full clip."
        ),
    )
    parser.add_argument(
        "--render-preview",
        action="store_true",
        help=(
            "Render the first context frame of each output chunk as a PNG, "
            "using its calibrated NCore F-theta model and rolling-shutter poses and "
            "compositing the observation-derived sky behind the Gaussians. "
            "Install with `uv sync --extra render` first."
        ),
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help=(
            "Render every original frame from the first context camera as an H.264 MP4, "
            "using the source F-theta calibration, exposure trajectory, and sky. Requires "
            "one resolved sequence, --merge, enough --max-chunks for full coverage, "
            "`uv sync --extra render`, and ffmpeg with libx264. With --index-path, a log that "
            "--max-chunks truncates renders only the frames of the reconstructed chunks."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level forwarded to logging.basicConfig.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.include_dynamic and args.model == "pq-front":
        parser.error("--include-dynamic currently requires pa-front or pa-multiview")
    if args.motion_source == "learned" and not args.include_dynamic:
        parser.error("--motion-source learned requires --include-dynamic")

    if args.render_video and not args.merge:
        parser.error("--render-video requires --merge so one complete scene is rendered")
    if args.max_chunks <= 0:
        parser.error("--max-chunks must be greater than zero")
    if args.index_path is None and (args.n_frames is not None or args.max_frame_gap_us != 750_000):
        parser.error("--n-frames / --max-frame-gap-us apply to --index-path input only")

    if args.render_preview or args.render_video:
        from instant_nurec.predict.render_preview import require_gsplat

        try:
            require_gsplat()
        except RuntimeError as exc:
            parser.error(str(exc))

    if args.render_video:
        from instant_nurec.predict.render_video import require_ffmpeg

        try:
            require_ffmpeg()
        except RuntimeError as exc:
            parser.error(str(exc))

    # Lazy imports keep argparse-only invocations (e.g. --help) cheap.
    from instant_nurec.config_schema.dataset import (
        AdaptiveSequentialFrameBatchSamplerConfig,
        NCoreInstantNuRecCuboidTracksParamsConfig,
        NCoreInstantNuRecDatasetConfig,
        InstantNuRecSplitsConfig,
    )
    from instant_nurec.config_schema.instantnurec import (
        GaussiansInstantNuRecSystemConfig,
        InstantNuRecConfig,
    )
    from instant_nurec.config_schema.models import (
        KelvinDPTDecoderConfig,
        KelvinModelConfig,
        KelvinPointQueryCADecoderConfig,
    )
    from instant_nurec.config_schema.predict import PredictConfig, PrimitiveMergeConfig
    from instant_nurec.ncore_input import resolve_ncore_paths
    from instant_nurec.predict.run import run_predict

    profile = get_model_profile(args.model)
    if args.index_path is not None:
        return _main_index(args, parser, profile)

    json_paths = resolve_ncore_paths(args.ncore_path)
    if args.render_video and len(json_paths) != 1:
        parser.error(
            "--render-video currently requires exactly one resolved NCore sequence "
            "so all requested chunks are merged into one complete scene"
        )
    camera_ids = list(args.camera_ids or profile.context_camera_ids)
    allowed_camera_counts = profile.supported_camera_counts
    if len(camera_ids) not in allowed_camera_counts:
        counts = ", ".join(str(count) for count in allowed_camera_counts)
        raise ValueError(
            f"{args.model} expects {counts} context camera(s); got {len(camera_ids)}. "
            "Repeat --camera-id once per camera."
        )

    decoder_config = (
        KelvinPointQueryCADecoderConfig(motion_depth=profile.motion_depth)
        if profile.decoder_kind == "point-query"
        else KelvinDPTDecoderConfig(motion_depth=profile.motion_depth)
    )

    config = InstantNuRecConfig(
        out_dir=str(args.output_dir),
        release_profile=args.model,
        system=(
            GaussiansInstantNuRecSystemConfig(predict_batch_size=args.max_chunks)
            if args.render_video
            else GaussiansInstantNuRecSystemConfig()
        ),
        model=KelvinModelConfig(
            decoder=decoder_config,
            track_padding_m=args.cuboid_padding,
        ),
        dataset=InstantNuRecSplitsConfig(
            predict=NCoreInstantNuRecDatasetConfig(
                ncore_json_paths=[str(p) for p in json_paths],
                camera_subsampler={
                    "frame_width": profile.frame_width,
                    "frame_height": profile.frame_height,
                },
                context_camera_ids=camera_ids,
                supervision_camera_ids=camera_ids,
                cuboid_tracks_params=NCoreInstantNuRecCuboidTracksParamsConfig(
                    track_label_source=args.track_label_source,
                ),
                frame_batch_sampler=AdaptiveSequentialFrameBatchSamplerConfig(
                    n_frames_per_sample=profile.n_frames_per_sample,
                    n_samples_per_sequence=args.max_chunks,
                ),
            ),
        ),
        predict=PredictConfig(
            include_dynamic=args.include_dynamic,
            motion_source=args.motion_source,
            primitive_merge=PrimitiveMergeConfig(
                enabled=args.merge,
                enable_voxelization=args.merge,
                target_n_gaussians=args.n_gaussians,
            ),
            render_preview=args.render_preview,
            render_video=args.render_video,
            checkpoint_path=str(args.checkpoint) if args.checkpoint is not None else None,
        ),
    )
    run_predict(config)
    _print_next(args)
    return 0


def _main_index(args: argparse.Namespace, parser: argparse.ArgumentParser, profile) -> int:
    """``--index-path``: PMT index logs through the same predict, merge and export path as NCore sequences."""
    import pickle

    from instant_nurec.config_schema.dataset import (
        AdaptiveSequentialFrameBatchSamplerConfig,
        InstantNuRecSplitsConfig,
        NCoreInstantNuRecCuboidTracksParamsConfig,
        PMTIndexDatasetConfig,
    )
    from instant_nurec.config_schema.instantnurec import GaussiansInstantNuRecSystemConfig, InstantNuRecConfig
    from instant_nurec.config_schema.models import KelvinDPTDecoderConfig, KelvinModelConfig
    from instant_nurec.config_schema.predict import PredictConfig, PrimitiveMergeConfig
    from instant_nurec.predict.run import run_predict

    if profile.decoder_kind != "pixel-aligned":
        parser.error("--index-path requires a pixel-aligned profile (pa-front or pa-multiview)")
    index_paths: list[Path] = []
    for path in args.index_path:
        found = sorted(path.glob("*.pkl")) if path.is_dir() else [path]
        if not found or not all(p.is_file() for p in found):
            parser.error(f"--index-path {path}: no index pickle")
        index_paths.extend(found)
    if args.render_video and len(index_paths) != 1:
        parser.error("--render-video currently requires exactly one index log")
    with open(index_paths[0], "rb") as f:
        first = pickle.load(f)
    camera_ids = list(args.camera_ids or first["cameras"])
    n_frames = args.n_frames or profile.n_frames_per_sample
    logging.getLogger(__name__).info(
        "%d index log(s), cameras %s, %d frames per camera per chunk, gap <= %d us",
        len(index_paths), camera_ids, n_frames, args.max_frame_gap_us,
    )
    config = InstantNuRecConfig(
        out_dir=str(args.output_dir),
        release_profile=args.model,
        # one batch = every chunk of one log, so --merge always merges a whole log
        system=GaussiansInstantNuRecSystemConfig(predict_batch_size=args.max_chunks),
        model=KelvinModelConfig(
            decoder=KelvinDPTDecoderConfig(motion_depth=profile.motion_depth),
            track_padding_m=args.cuboid_padding,
        ),
        dataset=InstantNuRecSplitsConfig(
            predict=PMTIndexDatasetConfig(
                index_paths=[str(p) for p in index_paths],
                camera_ids=camera_ids,
                camera_subsampler={"frame_width": profile.frame_width, "frame_height": profile.frame_height},
                n_frames_per_sample=n_frames,
                context_requires_depth=False,  # images only: every frame is a context candidate
                depth=False,
                semantic_segmentation=False,
                cuboid_tracks=NCoreInstantNuRecCuboidTracksParamsConfig() if "tracks" in first else None,
                track_dynamic_any_chunk="tracks" in first,  # a box that moves in any chunk is dynamic in all
                frame_batch_sampler=AdaptiveSequentialFrameBatchSamplerConfig(
                    n_frames_per_sample=n_frames,
                    n_samples_per_sequence=args.max_chunks,
                    max_frame_gap_timestamp_us=args.max_frame_gap_us,
                ),
            ),
        ),
        predict=PredictConfig(
            include_dynamic=args.include_dynamic,
            motion_source=args.motion_source,
            primitive_merge=PrimitiveMergeConfig(
                enabled=args.merge,
                enable_voxelization=args.merge,
                target_n_gaussians=args.n_gaussians,
            ),
            render_preview=args.render_preview,
            render_video=args.render_video,
            checkpoint_path=str(args.checkpoint) if args.checkpoint is not None else None,
        ),
    )
    run_predict(config)
    _print_next(args)
    return 0


def _print_next(args: argparse.Namespace) -> None:
    if args.merge:
        print(
            "Next: refine into USDZ with NuRec — "
            "https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html",
            flush=True,
        )
    else:
        print(
            "Next: view your 3DGS PLY with SuperSplat "
            "(https://playcanvas.com/supersplat/editor) "
            "or ply_viewer (NuRec container).",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
