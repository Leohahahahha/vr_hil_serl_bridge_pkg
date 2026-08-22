#!/usr/bin/env python3
"""Convert the raw DM-Tac W recording into N0-VTLA canonical LeRobot v3.

The raw recorder stores each DM-Tac W sample as one lossless packed uint8 row
in zarr.  This exporter decodes ``shear_x``, ``shear_y`` and ``depth``, applies
one dataset-level linear mapping shared by both fingers, and writes the result
as two three-channel tactile videos.  N0-VTLA then loads frame 0 as the
zero-contact baseline and computes ``current - baseline`` itself.

The output follows the canonical N0-VTLA contract:

* ``observation.state``: 32 floats.  A single FR3 arm occupies dimensions 0:10
  as xyz + rot6d + gripper; dimensions 10:32 are zero.
* ``action``: 32 floats containing the absolute commanded target pose in the
  same xyz + rot6d + gripper layout.
* ``action_mask``: dimensions 0:10 are true and 10:32 are false.
* RGB videos use ``observation.image.third_view`` and
  ``observation.image.left_wrist_view``.
* The two physical finger sensors use
  ``observation.image.left_wrist_left_tactile`` and
  ``observation.image.left_wrist_right_tactile``.  N0-VTLA supplies false-mask
  placeholders for the two absent right-arm tactile slots.

By default, timestamp/candidate gaps are rejected.  ``--timing-policy compact``
exists only for format smoke tests: it compacts retained rows onto a fixed-FPS
grid and therefore changes the physical time scale of an irregular recording.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


CANONICAL_DIM = 32
FR3_DIM = 10

STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_MASK_KEY = "action_mask"
FRONT_KEY = "observation.image.third_view"
WRIST_KEY = "observation.image.left_wrist_view"
TACTILE_LEFT_KEY = "observation.image.left_wrist_left_tactile"
TACTILE_RIGHT_KEY = "observation.image.left_wrist_right_tactile"
VIDEO_KEYS = (FRONT_KEY, WRIST_KEY, TACTILE_LEFT_KEY, TACTILE_RIGHT_KEY)

TACTILE_CHANNELS = ("shear_x", "shear_y", "depth")
EXPECTED_TACTILE_PACKAGE = "dmtac_tactile"
EXPECTED_DMTAC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FrameRef:
    raw_episode: str
    episode_index: int
    frame_index: int
    front_path: Path
    wrist_path: Path
    tactile_left_index: int
    tactile_right_index: int


class DMTacSource:
    """Random-access decoder for one side of the packed DM-Tac W stream."""

    def __init__(self, raw_root: Path, side: str):
        import zarr

        self.side = side
        self.zarr_path = raw_root / "tactile" / f"tactile_{side}" / "data.zarr"
        self.meta_path = raw_root / "tactile" / f"tactile_{side}" / "meta.jsonl"
        if not self.zarr_path.exists():
            raise FileNotFoundError(self.zarr_path)
        if not self.meta_path.exists():
            raise FileNotFoundError(self.meta_path)
        self.array = zarr.open(str(self.zarr_path), mode="r")
        if len(self.array.shape) != 2 or np.dtype(self.array.dtype) != np.dtype(np.uint8):
            raise ValueError(
                f"{self.zarr_path} must be a 2-D uint8 array, got "
                f"shape={self.array.shape}, dtype={self.array.dtype}"
            )
        self.meta: dict[int, dict[str, Any]] = {}
        for line_number, line in enumerate(self.meta_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["zarr_index"])
            if index in self.meta:
                raise ValueError(f"{self.meta_path}:{line_number}: duplicate zarr_index={index}")
            self.meta[index] = row

    def metadata(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= int(self.array.shape[0]):
            raise IndexError(
                f"{self.side} tactile zarr index {index} is outside 0..{int(self.array.shape[0]) - 1}"
            )
        try:
            return self.meta[index]
        except KeyError as exc:
            raise KeyError(f"{self.meta_path} has no metadata for zarr_index={index}") from exc

    def field(self, index: int) -> np.ndarray:
        meta = self.metadata(index)
        if str(meta.get("msg_package")) != EXPECTED_TACTILE_PACKAGE:
            raise ValueError(
                f"{self.side} zarr_index={index}: expected msg_package="
                f"{EXPECTED_TACTILE_PACKAGE!r}, got {meta.get('msg_package')!r}"
            )
        packed = np.asarray(self.array[index], dtype=np.uint8).reshape(-1)
        layout = meta.get("layout")
        if not isinstance(layout, dict):
            raise ValueError(f"{self.side} zarr_index={index}: missing layout metadata")
        if int(layout.get("schema_version", -1)) != EXPECTED_DMTAC_SCHEMA_VERSION:
            raise ValueError(
                f"{self.side} zarr_index={index}: unsupported schema_version="
                f"{layout.get('schema_version')!r}"
            )
        if int(layout.get("byte_order_little_endian", 0)) != 1:
            raise ValueError(f"{self.side} zarr_index={index}: only little-endian float32 is supported")
        expected_bytes = int(layout.get("packed_frame_bytes", -1))
        if packed.size != expected_bytes:
            raise ValueError(
                f"{self.side} zarr_index={index}: packed bytes {packed.size} != metadata {expected_bytes}"
            )
        shear = _decode_float_modality(packed, layout, "shear", expected_channels=2)
        depth = _decode_float_modality(packed, layout, "depth", expected_channels=1)
        field = np.concatenate([shear, depth], axis=-1).astype(np.float32, copy=False)
        if field.shape[-1] != 3 or not np.all(np.isfinite(field)):
            raise ValueError(
                f"{self.side} zarr_index={index}: invalid shear/depth field "
                f"shape={field.shape}, finite={bool(np.isfinite(field).all())}"
            )
        return field


def _decode_float_modality(
    packed: np.ndarray,
    layout: dict[str, Any],
    name: str,
    *,
    expected_channels: int,
) -> np.ndarray:
    start = int(layout[f"{name}_start"])
    length = int(layout[f"{name}_len"])
    height = int(layout[f"{name}_height"])
    width = int(layout[f"{name}_width"])
    channels = int(layout[f"{name}_channels"])
    itemsize = int(layout[f"{name}_itemsize"])
    if channels != expected_channels or itemsize != 4:
        raise ValueError(
            f"{name}: expected float32 x {expected_channels} channels, "
            f"got itemsize={itemsize}, channels={channels}"
        )
    expected_length = height * width * channels * itemsize
    if length != expected_length or start < 0 or start + length > packed.size:
        raise ValueError(
            f"{name}: invalid byte range start={start}, length={length}, "
            f"expected_length={expected_length}, packed={packed.size}"
        )
    values = packed[start : start + length].view("<f4")
    return values.reshape(height, width, channels)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True, type=Path, help="Raw DM-Tac W dataset root")
    parser.add_argument("--out-root", required=True, type=Path, help="Output canonical LeRobot root")
    parser.add_argument("--fps", type=float, default=None, help="Override source meta/info.json fps")
    parser.add_argument("--task", default=None, help="Override source task_description")
    parser.add_argument(
        "--timing-policy",
        choices=("strict", "compact"),
        default="strict",
        help="strict rejects missing fixed-FPS candidates; compact is only for format smoke tests",
    )
    parser.add_argument(
        "--timing-tolerance-sec",
        type=float,
        default=None,
        help="Allowed error around 1/fps in strict mode (default: 0.25/fps)",
    )
    parser.add_argument(
        "--max-tactile-sync-sec",
        type=float,
        default=0.15,
        help="Reject a referenced tactile frame whose absolute sync_dt exceeds this value",
    )
    parser.add_argument(
        "--tactile-scale",
        type=float,
        nargs=3,
        metavar=("SHEAR_X", "SHEAR_Y", "DEPTH"),
        default=None,
        help="Manual symmetric channel scales in source units; otherwise estimate from this dataset",
    )
    parser.add_argument(
        "--tactile-percentile",
        type=float,
        default=99.9,
        help="Percentile of abs(field-baseline) used for automatic channel scales",
    )
    parser.add_argument(
        "--scale-sample-stride",
        type=int,
        default=8,
        help="Spatial stride while estimating tactile scales",
    )
    parser.add_argument(
        "--state-gripper-unit",
        choices=("normalized", "meter"),
        default="normalized",
        help="Unit of raw robot.gripper_pos / observation.state[-1]",
    )
    parser.add_argument(
        "--action-gripper-unit",
        choices=("normalized", "meter"),
        default="meter",
        help="Unit of raw action.target_gripper_width",
    )
    parser.add_argument(
        "--gripper-output-unit",
        choices=("normalized", "meter"),
        default="normalized",
        help="Common gripper unit written into state and action",
    )
    parser.add_argument(
        "--gripper-open-width-m",
        type=float,
        default=0.085,
        help="Fully open gripper width used for normalized<->meter conversion",
    )
    parser.add_argument("--video-codec", default="mp4v", help="FourCC codec used for all MP4 files")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output root")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = export_raw_dmtac_w_to_n0vtla_lerobot(
        raw_root=args.raw_root,
        out_root=args.out_root,
        fps=args.fps,
        task=args.task,
        timing_policy=args.timing_policy,
        timing_tolerance_sec=args.timing_tolerance_sec,
        max_tactile_sync_sec=args.max_tactile_sync_sec,
        tactile_scale=args.tactile_scale,
        tactile_percentile=args.tactile_percentile,
        scale_sample_stride=args.scale_sample_stride,
        state_gripper_unit=args.state_gripper_unit,
        action_gripper_unit=args.action_gripper_unit,
        gripper_output_unit=args.gripper_output_unit,
        gripper_open_width_m=args.gripper_open_width_m,
        video_codec=args.video_codec,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def export_raw_dmtac_w_to_n0vtla_lerobot(
    *,
    raw_root: Path,
    out_root: Path,
    fps: float | None = None,
    task: str | None = None,
    timing_policy: str = "strict",
    timing_tolerance_sec: float | None = None,
    max_tactile_sync_sec: float = 0.15,
    tactile_scale: Iterable[float] | None = None,
    tactile_percentile: float = 99.9,
    scale_sample_stride: int = 8,
    state_gripper_unit: str = "normalized",
    action_gripper_unit: str = "meter",
    gripper_output_unit: str = "normalized",
    gripper_open_width_m: float = 0.085,
    video_codec: str = "mp4v",
    overwrite: bool = False,
) -> dict[str, Any]:
    raw_root = raw_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    raw_info = _load_json(raw_root / "meta" / "info.json")
    if str(raw_info.get("tactile_msg_package")) != EXPECTED_TACTILE_PACKAGE:
        raise ValueError(
            f"DM-Tac W exporter requires tactile_msg_package={EXPECTED_TACTILE_PACKAGE!r}; "
            f"got {raw_info.get('tactile_msg_package')!r}"
        )
    fps_value = float(fps if fps is not None else raw_info.get("fps", 10.0))
    if not np.isfinite(fps_value) or fps_value <= 0:
        raise ValueError(f"invalid fps: {fps_value}")
    task_text = str(task if task is not None else raw_info.get("task_description", "teleoperation"))
    if timing_policy not in {"strict", "compact"}:
        raise ValueError(f"unsupported timing_policy={timing_policy!r}")
    timing_tolerance = float(
        timing_tolerance_sec if timing_tolerance_sec is not None else 0.25 / fps_value
    )
    if timing_tolerance < 0:
        raise ValueError("timing_tolerance_sec must be non-negative")
    if max_tactile_sync_sec <= 0:
        raise ValueError("max_tactile_sync_sec must be positive")
    if not 0 < tactile_percentile <= 100:
        raise ValueError("tactile_percentile must be in (0, 100]")
    if scale_sample_stride <= 0:
        raise ValueError("scale_sample_stride must be positive")
    if gripper_open_width_m <= 0:
        raise ValueError("gripper_open_width_m must be positive")
    if len(video_codec) != 4:
        raise ValueError("video_codec must be a four-character FourCC value")

    left_source = DMTacSource(raw_root, "left")
    right_source = DMTacSource(raw_root, "right")
    source_episode_metadata = _source_episode_metadata_report(raw_root)
    episode_paths = _find_episode_parquets(raw_root)
    if not episode_paths:
        raise FileNotFoundError(f"no episode parquet files found below {raw_root / 'data'}")

    rows: list[dict[str, Any]] = []
    frames: list[FrameRef] = []
    episode_rows: list[dict[str, Any]] = []
    timing_reports: list[dict[str, Any]] = []
    global_index = 0

    for raw_episode_path in episode_paths:
        df = pd.read_parquet(raw_episode_path)
        _require_columns(
            df,
            (
                "timestamp",
                "frame_index",
                STATE_KEY,
                ACTION_KEY,
                "action.pose7",
                "action.target_gripper_width",
                "image.path",
                "wrist_image.path",
                "tactile.left.zarr_index",
                "tactile.right.zarr_index",
            ),
            raw_episode_path,
        )
        if df.empty:
            continue
        timing = _timing_report(df, fps_value, timing_tolerance)
        timing["raw_episode"] = str(raw_episode_path.relative_to(raw_root))
        timing_reports.append(timing)
        if timing_policy == "strict" and not timing["strict_ok"]:
            duplicate_note = ""
            if source_episode_metadata["duplicate_episode_indices"]:
                duplicate_note = (
                    " Source meta/episodes.jsonl also contains duplicate episode_index values: "
                    f"{source_episode_metadata['duplicate_episode_indices']}."
                )
            raise ValueError(
                f"{raw_episode_path} is not a continuous {fps_value:g} Hz episode: "
                f"missing_candidate_steps={timing['missing_candidate_steps']}, "
                f"max_timestamp_step_sec={timing['max_timestamp_step_sec']:.6f}. "
                f"{duplicate_note} "
                "Re-record with continuous sensor updates. Use --timing-policy compact only "
                "for a format smoke test because it changes the physical time scale."
            )

        episode_index = len(episode_rows)
        episode_start = global_index
        left_frame_ids: list[int] = []
        right_frame_ids: list[int] = []
        for frame_index, (_, raw_row) in enumerate(df.iterrows()):
            state32, action32, action_mask = _canonical_state_action(
                raw_row,
                state_gripper_unit=state_gripper_unit,
                action_gripper_unit=action_gripper_unit,
                gripper_output_unit=gripper_output_unit,
                gripper_open_width_m=gripper_open_width_m,
            )
            left_index = int(raw_row["tactile.left.zarr_index"])
            right_index = int(raw_row["tactile.right.zarr_index"])
            left_meta = left_source.metadata(left_index)
            right_meta = right_source.metadata(right_index)
            _check_tactile_sync(left_meta, max_tactile_sync_sec, "left", left_index)
            _check_tactile_sync(right_meta, max_tactile_sync_sec, "right", right_index)
            left_frame_ids.append(int(left_meta["frame_idx"]))
            right_frame_ids.append(int(right_meta["frame_idx"]))
            front_path = _resolve_sidecar_path(raw_root, raw_row["image.path"])
            wrist_path = _resolve_sidecar_path(raw_root, raw_row["wrist_image.path"])
            if not front_path.is_file():
                raise FileNotFoundError(front_path)
            if not wrist_path.is_file():
                raise FileNotFoundError(wrist_path)

            rows.append(
                {
                    "index": int(global_index),
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_index),
                    "timestamp": float(frame_index / fps_value),
                    "task_index": 0,
                    STATE_KEY: state32,
                    ACTION_KEY: action32,
                    ACTION_MASK_KEY: action_mask,
                }
            )
            frames.append(
                FrameRef(
                    raw_episode=str(raw_episode_path.relative_to(raw_root)),
                    episode_index=episode_index,
                    frame_index=frame_index,
                    front_path=front_path,
                    wrist_path=wrist_path,
                    tactile_left_index=left_index,
                    tactile_right_index=right_index,
                )
            )
            global_index += 1

        episode_end = global_index
        episode_rows.append(
            {
                "episode_index": int(episode_index),
                "tasks": [task_text],
                "length": int(episode_end - episode_start),
                "from": int(episode_start),
                "to": int(episode_end),
                "dataset_from_index": int(episode_start),
                "dataset_to_index": int(episode_end),
                "data/chunk_index": 0,
                "data/file_index": 0,
                "task_index": 0,
                "raw_episode": str(raw_episode_path.relative_to(raw_root)),
                "tactile_left_unique_ratio": _unique_ratio(left_frame_ids),
                "tactile_right_unique_ratio": _unique_ratio(right_frame_ids),
            }
        )

    if not rows:
        raise RuntimeError(f"no frames exported from {raw_root}")
    if timing_policy == "strict" and source_episode_metadata["duplicate_episode_indices"]:
        raise ValueError(
            "source meta/episodes.jsonl contains duplicate episode_index values: "
            f"{source_episode_metadata['duplicate_episode_indices']}; use a new dataset root "
            "for every recording or repair the overwritten episode metadata before conversion"
        )

    scales = _resolve_tactile_scales(
        tactile_scale=tactile_scale,
        percentile=tactile_percentile,
        sample_stride=scale_sample_stride,
        frames=frames,
        left_source=left_source,
        right_source=right_source,
    )
    baseline_report = _baseline_report(frames, left_source, right_source, scales)

    _prepare_output_root(out_root, overwrite=overwrite)
    data_path = out_root / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_path = out_root / "meta" / "tasks.parquet"
    stats_path = out_root / "meta" / "stats.json"
    info_path = out_root / "meta" / "info.json"
    tactile_encoding_path = out_root / "meta" / "tactile_encoding.json"
    conversion_report_path = out_root / "meta" / "conversion_report.json"
    for path in (data_path, episodes_path, tasks_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    video_paths = {
        key: out_root / "videos" / key / "chunk-000" / "file-000.mp4"
        for key in VIDEO_KEYS
    }
    for path in video_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    _write_data_parquet(rows, data_path)
    _write_tasks_parquet(tasks_path, task_text)

    front_shape = _image_shape(frames[0].front_path)
    wrist_shape = _image_shape(frames[0].wrist_path)
    tactile_left_shape = list(left_source.field(frames[0].tactile_left_index).shape)
    tactile_right_shape = list(right_source.field(frames[0].tactile_right_index).shape)
    _write_path_video((frame.front_path for frame in frames), video_paths[FRONT_KEY], fps_value, video_codec)
    _write_path_video((frame.wrist_path for frame in frames), video_paths[WRIST_KEY], fps_value, video_codec)
    _write_tactile_video(
        frames,
        side="left",
        source=left_source,
        scales=scales,
        out_path=video_paths[TACTILE_LEFT_KEY],
        fps=fps_value,
        codec=video_codec,
    )
    _write_tactile_video(
        frames,
        side="right",
        source=right_source,
        scales=scales,
        out_path=video_paths[TACTILE_RIGHT_KEY],
        fps=fps_value,
        codec=video_codec,
    )

    for episode in episode_rows:
        start = int(episode["from"])
        end = int(episode["to"])
        for key in VIDEO_KEYS:
            episode[f"videos/{key}/chunk_index"] = 0
            episode[f"videos/{key}/file_index"] = 0
            episode[f"videos/{key}/from_timestamp"] = float(start / fps_value)
            episode[f"videos/{key}/to_timestamp"] = float(end / fps_value)
    pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)

    stats = {
        STATE_KEY: _quantile_stats(np.stack([row[STATE_KEY] for row in rows])),
        ACTION_KEY: _quantile_stats(np.stack([row[ACTION_KEY] for row in rows])),
        ACTION_MASK_KEY: _quantile_stats(np.stack([row[ACTION_MASK_KEY] for row in rows])),
    }
    _write_json(stats_path, stats)

    tactile_encoding = {
        "version": 1,
        "source": "DM-Tac W SDK 0.1.4 packed schema 1",
        "source_channels": list(TACTILE_CHANNELS),
        "video_rgb_channels": list(TACTILE_CHANNELS),
        "scales": {name: float(scale) for name, scale in zip(TACTILE_CHANNELS, scales, strict=True)},
        "scale_estimator": (
            "manual" if tactile_scale is not None else f"p{tactile_percentile:g}(abs(field-episode_frame0))"
        ),
        "encoding": "uint8 = round(127.5 + 127.5 * clip(value / scale, -1, 1))",
        "decoding": "value ~= ((uint8 / 127.5) - 1) * scale",
        "n0vtla_baseline": "N0-VTLA loads episode frame 0 and computes current - baseline",
        "shared_between_sensors": True,
        "lossy_video_warning": f"Encoded with {video_codec}; keep tactile_encoding.json for deployment preprocessing",
    }
    _write_json(tactile_encoding_path, tactile_encoding)

    info = _build_info(
        raw_root=raw_root,
        raw_info=raw_info,
        fps=fps_value,
        total_frames=len(rows),
        total_episodes=len(episode_rows),
        front_shape=front_shape,
        wrist_shape=wrist_shape,
        tactile_left_shape=tactile_left_shape,
        tactile_right_shape=tactile_right_shape,
        video_codec=video_codec,
        task=task_text,
        gripper_output_unit=gripper_output_unit,
    )
    _write_json(info_path, info)

    conversion_report = {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "timing_policy": timing_policy,
        "timing_reports": timing_reports,
        "timing_compacted": timing_policy == "compact",
        "fps": fps_value,
        "total_frames": len(rows),
        "total_episodes": len(episode_rows),
        "source_episode_metadata": source_episode_metadata,
        "orphan_tactile_rows": {
            "left": int(left_source.array.shape[0] - len({f.tactile_left_index for f in frames})),
            "right": int(right_source.array.shape[0] - len({f.tactile_right_index for f in frames})),
        },
        "episode_tactile_unique_ratios": [
            {
                "episode_index": int(row["episode_index"]),
                "left": float(row["tactile_left_unique_ratio"]),
                "right": float(row["tactile_right_unique_ratio"]),
            }
            for row in episode_rows
        ],
        "baseline": baseline_report,
        "tactile_sync_sec": {
            "left": _referenced_sync_report(frames, left_source, "left"),
            "right": _referenced_sync_report(frames, right_source, "right"),
        },
        "n0vtla_reference": {
            "action_horizon": 50,
            "full_unpadded_windows_per_episode": [
                {
                    "episode_index": int(row["episode_index"]),
                    "windows": max(0, int(row["length"]) - 50 + 1),
                }
                for row in episode_rows
            ],
        },
        "gripper": {
            "state_source_unit": state_gripper_unit,
            "action_source_unit": action_gripper_unit,
            "output_unit": gripper_output_unit,
            "open_width_m": gripper_open_width_m,
        },
    }
    _write_json(conversion_report_path, conversion_report)

    validation = _validate_export(out_root, len(rows), len(episode_rows))
    return {
        "ok": True,
        "out_root": str(out_root),
        "fps": fps_value,
        "total_frames": len(rows),
        "total_episodes": len(episode_rows),
        "timing_policy": timing_policy,
        "tactile_scales": tactile_encoding["scales"],
        "validation": validation,
        "conversion_report": str(conversion_report_path),
        "tactile_encoding": str(tactile_encoding_path),
    }


def _find_episode_parquets(raw_root: Path) -> list[Path]:
    episode_dir = raw_root / "data" / "episodes"
    paths = sorted(episode_dir.glob("episode_*.parquet")) if episode_dir.is_dir() else []
    if paths:
        return paths
    master = raw_root / "data" / "chunk-000" / "file-000.parquet"
    return [master] if master.is_file() else []


def _source_episode_metadata_report(raw_root: Path) -> dict[str, Any]:
    path = raw_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return {"path": str(path.relative_to(raw_root)), "rows": 0, "duplicate_episode_indices": []}
    indices: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "episode_index" not in row:
            raise ValueError(f"{path}:{line_number}: missing episode_index")
        indices.append(int(row["episode_index"]))
    counts = {index: indices.count(index) for index in sorted(set(indices))}
    return {
        "path": str(path.relative_to(raw_root)),
        "rows": len(indices),
        "duplicate_episode_indices": [index for index, count in counts.items() if count > 1],
    }


def _referenced_sync_report(
    frames: list[FrameRef], source: DMTacSource, side: str
) -> dict[str, float]:
    indices = (
        [frame.tactile_left_index for frame in frames]
        if side == "left"
        else [frame.tactile_right_index for frame in frames]
    )
    values = np.asarray([float(source.metadata(index)["sync_dt"]) for index in indices], dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "abs_max": float(np.max(np.abs(values))),
        "abs_p99": float(np.percentile(np.abs(values), 99)),
    }


def _timing_report(df: pd.DataFrame, fps: float, tolerance: float) -> dict[str, Any]:
    timestamp = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(timestamp)) or np.any(np.diff(timestamp) <= 0):
        raise ValueError("source timestamps must be finite and strictly increasing")
    dt = np.diff(timestamp)
    expected_dt = 1.0 / fps
    timestamp_ok = bool(np.all(np.abs(dt - expected_dt) <= tolerance))
    if "candidate_index" in df:
        candidates = pd.to_numeric(df["candidate_index"], errors="coerce").to_numpy(dtype=np.int64)
        candidate_steps = np.diff(candidates)
        candidate_ok = bool(np.all(candidate_steps == 1))
        missing_candidate_steps = int(np.sum(np.maximum(candidate_steps - 1, 0)))
    else:
        candidate_ok = timestamp_ok
        missing_candidate_steps = int(np.sum(np.maximum(np.rint(dt * fps).astype(int) - 1, 0)))
    source_duration = float(timestamp[-1] - timestamp[0]) if len(timestamp) > 1 else 0.0
    compact_duration = float((len(timestamp) - 1) / fps) if len(timestamp) > 1 else 0.0
    return {
        "frames": int(len(df)),
        "strict_ok": bool(timestamp_ok and candidate_ok),
        "missing_candidate_steps": missing_candidate_steps,
        "source_duration_sec": source_duration,
        "compacted_duration_sec": compact_duration,
        "duration_scale": float(compact_duration / source_duration) if source_duration > 0 else 1.0,
        "median_timestamp_step_sec": float(np.median(dt)) if dt.size else None,
        "max_timestamp_step_sec": float(np.max(dt)) if dt.size else None,
        "effective_retained_hz": float((len(timestamp) - 1) / source_duration) if source_duration > 0 else fps,
    }


def _canonical_state_action(
    row: pd.Series,
    *,
    state_gripper_unit: str,
    action_gripper_unit: str,
    gripper_output_unit: str,
    gripper_open_width_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state7 = _vector(row[STATE_KEY], 7, STATE_KEY)
    action7 = _vector(row[ACTION_KEY], 7, ACTION_KEY)
    target_pose7 = _vector(row["action.pose7"], 7, "action.pose7")
    state_rotation = _rotvec_to_matrix(state7[3:6])
    target_rotation = _quat_xyzw_to_matrix(target_pose7[3:7])
    state_gripper = _convert_gripper(
        float(state7[6]), state_gripper_unit, gripper_output_unit, gripper_open_width_m
    )
    target_gripper_raw = row.get("action.target_gripper_width", action7[6])
    target_gripper = _convert_gripper(
        float(target_gripper_raw), action_gripper_unit, gripper_output_unit, gripper_open_width_m
    )

    state10 = np.concatenate(
        [state7[:3], _matrix_to_rot6d(state_rotation), np.asarray([state_gripper], dtype=np.float32)]
    ).astype(np.float32)
    action10 = np.concatenate(
        [target_pose7[:3], _matrix_to_rot6d(target_rotation), np.asarray([target_gripper], dtype=np.float32)]
    ).astype(np.float32)
    state32 = np.zeros(CANONICAL_DIM, dtype=np.float32)
    action32 = np.zeros(CANONICAL_DIM, dtype=np.float32)
    action_mask = np.zeros(CANONICAL_DIM, dtype=bool)
    state32[:FR3_DIM] = state10
    action32[:FR3_DIM] = action10
    action_mask[:FR3_DIM] = True
    if not np.all(np.isfinite(state32)) or not np.all(np.isfinite(action32)):
        raise ValueError("canonical state/action contains non-finite values")
    return state32, action32, action_mask


def _convert_gripper(value: float, source_unit: str, output_unit: str, open_width_m: float) -> float:
    if not np.isfinite(value):
        raise ValueError("gripper contains a non-finite value")
    width_m = value * open_width_m if source_unit == "normalized" else value
    return float(width_m / open_width_m) if output_unit == "normalized" else float(width_m)


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        k = _skew(rotvec)
        return np.eye(3, dtype=np.float64) + k
    axis = rotvec / theta
    k = _skew(axis)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"invalid target quaternion: {q.tolist()}")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return np.concatenate([rotation[:, 0], rotation[:, 1]]).astype(np.float32)


def _resolve_tactile_scales(
    *,
    tactile_scale: Iterable[float] | None,
    percentile: float,
    sample_stride: int,
    frames: list[FrameRef],
    left_source: DMTacSource,
    right_source: DMTacSource,
) -> np.ndarray:
    if tactile_scale is not None:
        scale = np.asarray(list(tactile_scale), dtype=np.float32).reshape(-1)
        if scale.size != 3 or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("tactile_scale must contain three finite positive values")
        return scale

    baselines: dict[tuple[int, str], np.ndarray] = {}
    samples: list[list[np.ndarray]] = [[], [], []]
    for frame in frames:
        for side, source, index in (
            ("left", left_source, frame.tactile_left_index),
            ("right", right_source, frame.tactile_right_index),
        ):
            field = source.field(index)
            baseline_key = (frame.episode_index, side)
            if baseline_key not in baselines:
                baselines[baseline_key] = field.copy()
            diff = field - baselines[baseline_key]
            sample = np.abs(diff[::sample_stride, ::sample_stride])
            for channel in range(3):
                samples[channel].append(sample[..., channel].reshape(-1))
    scale = np.asarray(
        [np.percentile(np.concatenate(channel_samples), percentile) for channel_samples in samples],
        dtype=np.float32,
    )
    # A zero/no-contact smoke dataset still needs a deterministic, non-zero codec scale.
    scale = np.maximum(scale, np.float32(1e-3))
    return scale


def _encode_tactile(field: np.ndarray, scales: np.ndarray) -> np.ndarray:
    normalized = np.clip(np.asarray(field, dtype=np.float32) / scales.reshape(1, 1, 3), -1.0, 1.0)
    return np.rint(127.5 + 127.5 * normalized).astype(np.uint8)


def _baseline_report(
    frames: list[FrameRef],
    left_source: DMTacSource,
    right_source: DMTacSource,
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    first_by_episode: dict[int, FrameRef] = {}
    for frame in frames:
        first_by_episode.setdefault(frame.episode_index, frame)
    for episode_index, frame in sorted(first_by_episode.items()):
        item: dict[str, Any] = {"episode_index": int(episode_index)}
        for side, source, index in (
            ("left", left_source, frame.tactile_left_index),
            ("right", right_source, frame.tactile_right_index),
        ):
            field = source.field(index)
            abs_p99 = np.percentile(np.abs(field), 99, axis=(0, 1)).astype(float)
            item[side] = {
                "zarr_index": int(index),
                "abs_p99": abs_p99.tolist(),
                "relative_to_scale": (abs_p99 / scales).tolist(),
            }
        report.append(item)
    return report


def _check_tactile_sync(meta: dict[str, Any], limit: float, side: str, index: int) -> None:
    sync_dt = float(meta.get("sync_dt", np.nan))
    if not np.isfinite(sync_dt) or abs(sync_dt) > limit:
        raise ValueError(
            f"{side} tactile zarr_index={index}: abs(sync_dt)={abs(sync_dt):.6f} > {limit:.6f} sec"
        )


def _write_data_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.Table.from_arrays(
        [
            _fixed_list(rows, STATE_KEY, CANONICAL_DIM, pa.float32()),
            _fixed_list(rows, ACTION_KEY, CANONICAL_DIM, pa.float32()),
            _fixed_list(rows, ACTION_MASK_KEY, CANONICAL_DIM, pa.bool_()),
            pa.array([float(row["timestamp"]) for row in rows], type=pa.float32()),
            pa.array([int(row["frame_index"]) for row in rows], type=pa.int64()),
            pa.array([int(row["episode_index"]) for row in rows], type=pa.int64()),
            pa.array([int(row["index"]) for row in rows], type=pa.int64()),
            pa.array([int(row["task_index"]) for row in rows], type=pa.int64()),
        ],
        names=[
            STATE_KEY,
            ACTION_KEY,
            ACTION_MASK_KEY,
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    ).replace_schema_metadata(_hf_schema_metadata())
    pq.write_table(table, path, compression="zstd")


def _fixed_list(rows: list[dict[str, Any]], key: str, width: int, value_type: pa.DataType) -> pa.Array:
    values = np.stack([np.asarray(row[key]).reshape(width) for row in rows])
    flat = pa.array(values.reshape(-1).tolist(), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _hf_schema_metadata() -> dict[bytes, bytes]:
    features = {
        STATE_KEY: {"feature": {"dtype": "float32", "_type": "Value"}, "length": 32, "_type": "Sequence"},
        ACTION_KEY: {"feature": {"dtype": "float32", "_type": "Value"}, "length": 32, "_type": "Sequence"},
        ACTION_MASK_KEY: {"feature": {"dtype": "bool", "_type": "Value"}, "length": 32, "_type": "Sequence"},
        "timestamp": {"dtype": "float32", "_type": "Value"},
        "frame_index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "index": {"dtype": "int64", "_type": "Value"},
        "task_index": {"dtype": "int64", "_type": "Value"},
    }
    return {b"huggingface": json.dumps({"info": {"features": features}}).encode("utf-8")}


def _write_tasks_parquet(path: Path, task: str) -> None:
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array([task], type=pa.string()),
            }
        ),
        path,
        compression="zstd",
    )


def _write_path_video(paths: Iterable[Path], out_path: Path, fps: float, codec: str) -> None:
    iterator = iter(paths)
    try:
        first_path = next(iterator)
    except StopIteration as exc:
        raise ValueError(f"no frames for {out_path}") from exc
    first = np.asarray(Image.open(first_path).convert("RGB"))
    writer = _open_video_writer(out_path, first.shape, fps, codec)
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for path in iterator:
            rgb = np.asarray(Image.open(path).convert("RGB"))
            if rgb.shape != first.shape:
                raise ValueError(f"{path} shape {rgb.shape} != first frame {first.shape}")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_tactile_video(
    frames: list[FrameRef],
    *,
    side: str,
    source: DMTacSource,
    scales: np.ndarray,
    out_path: Path,
    fps: float,
    codec: str,
) -> None:
    if not frames:
        raise ValueError(f"no tactile frames for {out_path}")
    first_index = frames[0].tactile_left_index if side == "left" else frames[0].tactile_right_index
    first = _encode_tactile(source.field(first_index), scales)
    writer = _open_video_writer(out_path, first.shape, fps, codec)
    try:
        for frame in frames:
            index = frame.tactile_left_index if side == "left" else frame.tactile_right_index
            encoded = _encode_tactile(source.field(index), scales)
            if encoded.shape != first.shape:
                raise ValueError(
                    f"{side} zarr_index={index}: encoded shape {encoded.shape} != first {first.shape}"
                )
            writer.write(cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _open_video_writer(out_path: Path, shape: tuple[int, ...], fps: float, codec: str) -> cv2.VideoWriter:
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"video frame must be HWC RGB, got {shape}")
    height, width = int(shape[0]), int(shape[1])
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {out_path} with codec={codec!r}")
    return writer


def _build_info(
    *,
    raw_root: Path,
    raw_info: dict[str, Any],
    fps: float,
    total_frames: int,
    total_episodes: int,
    front_shape: list[int],
    wrist_shape: list[int],
    tactile_left_shape: list[int],
    tactile_right_shape: list[int],
    video_codec: str,
    task: str,
    gripper_output_unit: str,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        STATE_KEY: {"dtype": "float32", "shape": [32], "names": None},
        ACTION_KEY: {"dtype": "float32", "shape": [32], "names": None},
        ACTION_MASK_KEY: {"dtype": "bool", "shape": [32], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key, shape in (
        (FRONT_KEY, front_shape),
        (WRIST_KEY, wrist_shape),
        (TACTILE_LEFT_KEY, tactile_left_shape),
        (TACTILE_RIGHT_KEY, tactile_right_shape),
    ):
        features[key] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": int(shape[0]),
                "video.width": int(shape[1]),
                "video.codec": video_codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    return {
        "codebase_version": "v3.0",
        "dataset_type": "lerobot_v3_n0vtla_canonical_dmtac_w",
        "source_dataset_type": raw_info.get("dataset_type"),
        "source_raw_root": str(raw_root),
        "robot_type": "fr3_single_arm_tactile",
        "canonical_schema": True,
        "canonical_action_dim": 32,
        "active_action_dims": [0, 10],
        "gripper_unit": gripper_output_unit,
        "tactile_msg_package": EXPECTED_TACTILE_PACKAGE,
        "tactile_encoding_path": "meta/tactile_encoding.json",
        "fps": fps,
        "task": task,
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }


def _validate_export(root: Path, expected_frames: int, expected_episodes: int) -> dict[str, Any]:
    info = _load_json(root / "meta" / "info.json")
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path)
    episodes = pq.read_table(episodes_path)
    if table.num_rows != expected_frames or int(info["total_frames"]) != expected_frames:
        raise ValueError("export validation failed: frame count mismatch")
    if episodes.num_rows != expected_episodes or int(info["total_episodes"]) != expected_episodes:
        raise ValueError("export validation failed: episode count mismatch")
    for key in (STATE_KEY, ACTION_KEY, ACTION_MASK_KEY):
        column_type = table.schema.field(key).type
        if not pa.types.is_fixed_size_list(column_type) or column_type.list_size != CANONICAL_DIM:
            raise ValueError(f"export validation failed: {key} type is {column_type}")
    masks = np.asarray(table[ACTION_MASK_KEY].to_pylist(), dtype=bool)
    if not np.all(masks[:, :FR3_DIM]) or np.any(masks[:, FR3_DIM:]):
        raise ValueError("export validation failed: action_mask is not true[0:10]/false[10:32]")

    videos: dict[str, Any] = {}
    for key in VIDEO_KEYS:
        path = root / "videos" / key / "chunk-000" / "file-000.mp4"
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"export validation failed: cannot open {path}")
        count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ok, frame = cap.read()
        cap.release()
        if count != expected_frames or not ok or frame is None:
            raise ValueError(f"export validation failed: invalid video {path}, frames={count}")
        expected_shape = info["features"][key]["shape"]
        if [height, width, int(frame.shape[2])] != [int(x) for x in expected_shape]:
            raise ValueError(
                f"export validation failed: {key} shape {[height, width, frame.shape[2]]} "
                f"!= {expected_shape}"
            )
        videos[key] = {"frames": count, "shape": [height, width, int(frame.shape[2])]}
    return {"frames": expected_frames, "episodes": expected_episodes, "videos": videos}


def _quantile_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values)
    arr64 = arr.astype(np.float64)
    return {
        "min": np.min(arr64, axis=0).tolist(),
        "max": np.max(arr64, axis=0).tolist(),
        "mean": np.mean(arr64, axis=0).tolist(),
        "std": np.std(arr64, axis=0).tolist(),
        "q01": np.quantile(arr64, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr64, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr64, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr64, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr64, 0.99, axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _unique_ratio(frame_ids: list[int]) -> float:
    return float(len(set(frame_ids)) / len(frame_ids)) if frame_ids else 0.0


def _vector(value: Any, expected_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != expected_dim:
        raise ValueError(f"{name} expected dim {expected_dim}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _resolve_sidecar_path(raw_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else raw_root / path


def _image_shape(path: Path) -> list[int]:
    with Image.open(path) as image:
        width, height = image.convert("RGB").size
    return [int(height), int(width), 3]


def _prepare_output_root(out_root: Path, *, overwrite: bool) -> None:
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
