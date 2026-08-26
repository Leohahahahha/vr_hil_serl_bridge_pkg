#!/usr/bin/env python3
"""Convert raw DM-Tac W recordings to the Tabero-style LeRobot v2.1 layout.

This is the action-only first-reproduction contract:

* ``image`` and ``wrist_image`` are RGB videos.
* ``state`` is 7D xyz + axis-angle + measured single-finger position in meters.
* ``actions`` is the 7D absolute target xyz + axis-angle + target
  single-finger position in meters.
* ``tactile_marker_motion`` is [9, 198, 2]: a reference marker grid followed
  by eight current-position history frames.  Dense DM-Tac shear is sampled on
  a 9x11 grid per finger and converted to current positions as grid + shear.
* ``wrist_wrench`` and ``tactile_depth`` are retained for future experiments,
  but are not part of the action target and need not be read by the first
  Tabero loader.

The existing N0-VTLA exporter is intentionally separate and unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .raw_dmtac_w_to_n0vtla_lerobot import (
    DMTacSource,
    EXPECTED_TACTILE_PACKAGE,
    _check_tactile_sync,
    _find_episode_parquets,
    _image_shape,
    _load_json,
    _open_video_writer,
    _prepare_output_root,
    _require_columns,
    _resolve_sidecar_path,
    _source_episode_metadata_report,
    _timing_report,
    _vector,
    _write_json,
)


STATE_KEY = "state"
ACTION_KEY = "actions"
MARKER_KEY = "tactile_marker_motion"
WRENCH_KEY = "wrist_wrench"
DEPTH_KEY = "tactile_depth"
IMAGE_KEY = "image"
WRIST_IMAGE_KEY = "wrist_image"
VIDEO_KEYS = (IMAGE_KEY, WRIST_IMAGE_KEY)

STATE_DIM = 7
ACTION_DIM = 7
WRENCH_DIM = 6
HISTORY_LENGTH = 8
GRID_ROWS = 9
GRID_COLS = 11
POINTS_PER_SIDE = GRID_ROWS * GRID_COLS
TOTAL_POINTS = POINTS_PER_SIDE * 2
MARKER_SHAPE = (1 + HISTORY_LENGTH, TOTAL_POINTS, 2)
DEPTH_SHAPE = (2, 240, 320)

STATE_NAMES = [
    "x",
    "y",
    "z",
    "axis_angle_x",
    "axis_angle_y",
    "axis_angle_z",
    "gripper_finger_position_m",
]
ACTION_NAMES = [
    "target_x",
    "target_y",
    "target_z",
    "target_axis_angle_x",
    "target_axis_angle_y",
    "target_axis_angle_z",
    "target_gripper_finger_position_m",
]
WRENCH_NAMES = ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True, type=Path, help="Raw DM-Tac W dataset root")
    parser.add_argument("--out-root", required=True, type=Path, help="Output LeRobot v2.1 root")
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
        help="Reject a referenced tactile row whose absolute sync_dt exceeds this value",
    )
    parser.add_argument(
        "--shear-scale",
        type=float,
        default=1.0,
        help="Multiply both DM-Tac shear channels before adding them to the marker grid",
    )
    parser.add_argument(
        "--state-gripper-unit",
        choices=("normalized", "meter"),
        default="meter",
        help="Unit of raw observation.state[-1]; current recordings use meter",
    )
    parser.add_argument(
        "--state-gripper-coordinate",
        choices=("finger", "total_width"),
        default="finger",
        help="Meaning of raw observation.state[-1]; current recordings use finger",
    )
    parser.add_argument(
        "--action-gripper-unit",
        choices=("normalized", "meter"),
        default="meter",
        help="Unit of raw action.target_gripper_width",
    )
    parser.add_argument(
        "--gripper-output-unit",
        choices=("meter",),
        default="meter",
        help="Tabero output unit; fixed to physical meters",
    )
    parser.add_argument(
        "--gripper-open-width-m",
        type=float,
        default=0.085,
        help="Fully open width used for normalized/meter conversion",
    )
    parser.add_argument("--video-codec", default="mp4v", help="FourCC codec used for RGB MP4 files")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output root")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = export_raw_dmtac_w_to_tabero_lerobot(
        raw_root=args.raw_root,
        out_root=args.out_root,
        fps=args.fps,
        task=args.task,
        timing_policy=args.timing_policy,
        timing_tolerance_sec=args.timing_tolerance_sec,
        max_tactile_sync_sec=args.max_tactile_sync_sec,
        shear_scale=args.shear_scale,
        state_gripper_unit=args.state_gripper_unit,
        state_gripper_coordinate=args.state_gripper_coordinate,
        action_gripper_unit=args.action_gripper_unit,
        gripper_output_unit=args.gripper_output_unit,
        gripper_open_width_m=args.gripper_open_width_m,
        video_codec=args.video_codec,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def export_raw_dmtac_w_to_tabero_lerobot(
    *,
    raw_root: Path,
    out_root: Path,
    fps: float | None = None,
    task: str | None = None,
    timing_policy: str = "strict",
    timing_tolerance_sec: float | None = None,
    max_tactile_sync_sec: float = 0.15,
    shear_scale: float = 1.0,
    state_gripper_unit: str = "meter",
    state_gripper_coordinate: str = "finger",
    action_gripper_unit: str = "meter",
    gripper_output_unit: str = "meter",
    gripper_open_width_m: float = 0.085,
    video_codec: str = "mp4v",
    overwrite: bool = False,
) -> dict[str, Any]:
    raw_root = raw_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    raw_info = _load_json(raw_root / "meta" / "info.json")
    if str(raw_info.get("tactile_msg_package")) != EXPECTED_TACTILE_PACKAGE:
        raise ValueError(
            f"Tabero DM-Tac W exporter requires tactile_msg_package={EXPECTED_TACTILE_PACKAGE!r}; "
            f"got {raw_info.get('tactile_msg_package')!r}"
        )

    fps_value = float(fps if fps is not None else raw_info.get("fps", 10.0))
    if not np.isfinite(fps_value) or fps_value <= 0:
        raise ValueError(f"invalid fps: {fps_value}")
    task_text = str(task if task is not None else raw_info.get("task_description", "teleoperation"))
    timing_tolerance = float(
        timing_tolerance_sec if timing_tolerance_sec is not None else 0.25 / fps_value
    )
    if timing_policy not in {"strict", "compact"}:
        raise ValueError(f"unsupported timing_policy={timing_policy!r}")
    if timing_tolerance < 0:
        raise ValueError("timing_tolerance_sec must be non-negative")
    if not np.isfinite(max_tactile_sync_sec) or max_tactile_sync_sec <= 0:
        raise ValueError("max_tactile_sync_sec must be finite and positive")
    if not np.isfinite(shear_scale) or shear_scale <= 0:
        raise ValueError("shear_scale must be finite and positive")
    if gripper_open_width_m <= 0:
        raise ValueError("gripper_open_width_m must be positive")
    if state_gripper_coordinate not in {"finger", "total_width"}:
        raise ValueError(f"unsupported state_gripper_coordinate={state_gripper_coordinate!r}")
    if gripper_output_unit != "meter":
        raise ValueError("Tabero state/actions gripper output must use physical meters")
    if len(video_codec) != 4:
        raise ValueError("video_codec must be a four-character FourCC value")

    left_source = DMTacSource(raw_root, "left")
    right_source = DMTacSource(raw_root, "right")
    source_episode_metadata = _source_episode_metadata_report(raw_root)
    episode_paths = _find_episode_parquets(raw_root)
    if not episode_paths:
        raise FileNotFoundError(f"no episode parquet files found below {raw_root / 'data'}")
    if timing_policy == "strict" and source_episode_metadata["duplicate_episode_indices"]:
        raise ValueError(
            "source meta/episodes.jsonl contains duplicate episode_index values: "
            f"{source_episode_metadata['duplicate_episode_indices']}; repair the raw dataset first"
        )

    _prepare_output_root(out_root, overwrite=overwrite)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)

    episode_metadata: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    timing_reports: list[dict[str, Any]] = []
    schema_versions: set[int] = set()
    referenced_left: set[int] = set()
    referenced_right: set[int] = set()
    global_index = 0
    front_shape: list[int] | None = None
    wrist_shape: list[int] | None = None
    grid_reference: np.ndarray | None = None

    for raw_episode_path in episode_paths:
        df = pd.read_parquet(raw_episode_path)
        _require_columns(
            df,
            (
                "timestamp",
                "frame_index",
                "observation.state",
                "action.pose7",
                "action.target_gripper_width",
                "robot.force",
                "robot.torque",
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
            raise ValueError(
                f"{raw_episode_path} is not a continuous {fps_value:g} Hz episode: "
                f"missing_candidate_steps={timing['missing_candidate_steps']}, "
                f"max_timestamp_step_sec={timing['max_timestamp_step_sec']:.6f}. "
                "Re-record or repair the source; --timing-policy compact is only a format smoke test."
            )

        episode_index = len(episode_metadata)
        episode_rows: list[dict[str, Any]] = []
        front_paths: list[Path] = []
        wrist_paths: list[Path] = []
        current_markers: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        left_frame_ids: list[int] = []
        right_frame_ids: list[int] = []

        for frame_index, (_, raw_row) in enumerate(df.iterrows()):
            left_index = int(raw_row["tactile.left.zarr_index"])
            right_index = int(raw_row["tactile.right.zarr_index"])
            left_meta = left_source.metadata(left_index)
            right_meta = right_source.metadata(right_index)
            _check_tactile_sync(left_meta, max_tactile_sync_sec, "left", left_index)
            _check_tactile_sync(right_meta, max_tactile_sync_sec, "right", right_index)
            schema_versions.add(left_source.schema_version(left_index))
            schema_versions.add(right_source.schema_version(right_index))
            referenced_left.add(left_index)
            referenced_right.add(right_index)
            left_frame_ids.append(int(left_meta["frame_idx"]))
            right_frame_ids.append(int(right_meta["frame_idx"]))

            left_field = left_source.field(left_index)
            right_field = right_source.field(right_index)
            if left_field.shape != (240, 320, 3) or right_field.shape != (240, 320, 3):
                raise ValueError(
                    f"DM-Tac fields must be [240,320,3], got {left_field.shape} and {right_field.shape}"
                )
            reference, current = _sample_marker_positions(left_field, right_field, shear_scale)
            if grid_reference is None:
                grid_reference = reference
            elif not np.array_equal(grid_reference, reference):
                raise ValueError("internal error: tactile marker reference grid changed")
            current_markers.append(current)
            depth_frames.append(
                np.stack([left_field[..., 2], right_field[..., 2]], axis=0).astype(np.float32)
            )

            front_path = _resolve_sidecar_path(raw_root, raw_row["image.path"])
            wrist_path = _resolve_sidecar_path(raw_root, raw_row["wrist_image.path"])
            if not front_path.is_file():
                raise FileNotFoundError(front_path)
            if not wrist_path.is_file():
                raise FileNotFoundError(wrist_path)
            front_paths.append(front_path)
            wrist_paths.append(wrist_path)

            state, action = _tabero_state_action(
                raw_row,
                state_gripper_unit=state_gripper_unit,
                state_gripper_coordinate=state_gripper_coordinate,
                action_gripper_unit=action_gripper_unit,
                gripper_open_width_m=gripper_open_width_m,
            )
            wrench = np.concatenate(
                [
                    _vector(raw_row["robot.force"], 3, "robot.force"),
                    _vector(raw_row["robot.torque"], 3, "robot.torque"),
                ]
            ).astype(np.float32)
            episode_rows.append(
                {
                    STATE_KEY: state,
                    ACTION_KEY: action,
                    WRENCH_KEY: wrench,
                    DEPTH_KEY: depth_frames[-1],
                    "timestamp": np.float32(frame_index / fps_value),
                    "frame_index": int(frame_index),
                    "episode_index": int(episode_index),
                    "index": int(global_index),
                    "task_index": 0,
                }
            )
            global_index += 1

        if grid_reference is None:
            raise RuntimeError("internal error: no marker reference was created")
        marker_history = _build_marker_history(np.stack(current_markers), grid_reference)
        for row, marker in zip(episode_rows, marker_history, strict=True):
            row[MARKER_KEY] = marker

        episode_path = _v21_episode_path(out_root, episode_index)
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        _write_episode_parquet(episode_rows, episode_path)

        video_paths = {
            key: _v21_video_path(out_root, key, episode_index)
            for key in VIDEO_KEYS
        }
        for path in video_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        _write_path_video(front_paths, video_paths[IMAGE_KEY], fps_value, video_codec)
        _write_path_video(wrist_paths, video_paths[WRIST_IMAGE_KEY], fps_value, video_codec)

        if front_shape is None:
            front_shape = _image_shape(front_paths[0])
            wrist_shape = _image_shape(wrist_paths[0])
        elif _image_shape(front_paths[0]) != front_shape or _image_shape(wrist_paths[0]) != wrist_shape:
            raise ValueError("all episodes must use consistent RGB image shapes")

        length = len(episode_rows)
        episode_metadata.append(
            {
                "episode_index": int(episode_index),
                "tasks": [task_text],
                "length": int(length),
            }
        )
        episode_stats.append(
            {
                "episode_index": int(episode_index),
                "stats": _episode_numeric_stats(episode_rows),
            }
        )
        timing["left_tactile_unique_ratio"] = _unique_ratio(left_frame_ids)
        timing["right_tactile_unique_ratio"] = _unique_ratio(right_frame_ids)

    if not episode_metadata or front_shape is None or wrist_shape is None or grid_reference is None:
        raise RuntimeError(f"no frames exported from {raw_root}")

    info = _build_info(
        raw_root=raw_root,
        raw_info=raw_info,
        fps=fps_value,
        total_frames=global_index,
        total_episodes=len(episode_metadata),
        front_shape=front_shape,
        wrist_shape=wrist_shape,
        video_codec=video_codec,
        gripper_output_unit=gripper_output_unit,
        gripper_open_width_m=gripper_open_width_m,
        schema_versions=sorted(schema_versions),
    )
    _write_json(out_root / "meta" / "info.json", info)
    _write_jsonlines(out_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task_text}])
    _write_jsonlines(out_root / "meta" / "episodes.jsonl", episode_metadata)
    _write_jsonlines(out_root / "meta" / "episodes_stats.jsonl", episode_stats)

    conversion = {
        "version": 1,
        "source_raw_root": str(raw_root),
        "source_dataset_type": raw_info.get("dataset_type"),
        "output_contract": "tabero_action_only_lerobot_v2.1",
        "action_target": "7D absolute xyz + axis-angle + gripper; no wrench supervision",
        "retained_but_unused_first_loader": [WRENCH_KEY, DEPTH_KEY],
        "timing_policy": timing_policy,
        "timing_compacted": timing_policy == "compact",
        "timing_reports": timing_reports,
        "source_episode_metadata": source_episode_metadata,
        "source_tactile_schema_versions": sorted(schema_versions),
        "marker_field": {
            "shape": list(MARKER_SHAPE),
            "history_length": HISTORY_LENGTH,
            "side_order": ["left", "right"],
            "grid_rows": GRID_ROWS,
            "grid_cols": GRID_COLS,
            "grid_y_indices": _grid_indices(240, GRID_ROWS).tolist(),
            "grid_x_indices": _grid_indices(320, GRID_COLS).tolist(),
            "reference": "sampled sensor pixel coordinates (x,y)",
            "current": "reference + shear_scale * sampled (shear_x,shear_y)",
            "shear_scale": float(shear_scale),
            "early_history_padding": "repeat episode frame 0",
        },
        "depth_field": {
            "shape": list(DEPTH_SHAPE),
            "side_order": ["left", "right"],
            "source_channel": "decoded DM-Tac depth float32",
        },
        "wrench_field": {
            "shape": [WRENCH_DIM],
            "order": WRENCH_NAMES,
            "source": "synchronized robot.force + robot.torque",
        },
        "gripper": {
            "state_source_unit": state_gripper_unit,
            "state_source_coordinate": state_gripper_coordinate,
            "action_source_unit": action_gripper_unit,
            "action_source_coordinate": "total_width (legacy) or explicit finger field",
            "output_unit": "meter",
            "output_coordinate": "single_finger_absolute_position",
            "open_width_m": float(gripper_open_width_m),
            "max_finger_position_m": float(gripper_open_width_m / 2.0),
        },
        "orphan_tactile_rows": {
            "left": int(left_source.array.shape[0] - len(referenced_left)),
            "right": int(right_source.array.shape[0] - len(referenced_right)),
        },
    }
    _write_json(out_root / "meta" / "tabero_conversion.json", conversion)
    validation = validate_tabero_lerobot(out_root)
    return {
        "ok": True,
        "out_root": str(out_root),
        "fps": fps_value,
        "total_frames": global_index,
        "total_episodes": len(episode_metadata),
        "timing_policy": timing_policy,
        "validation": validation,
        "conversion_report": str(out_root / "meta" / "tabero_conversion.json"),
    }


def _tabero_state_action(
    row: pd.Series,
    *,
    state_gripper_unit: str,
    state_gripper_coordinate: str,
    action_gripper_unit: str,
    gripper_open_width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    state = _vector(row["observation.state"], STATE_DIM, "observation.state").copy()
    target_pose = _vector(row["action.pose7"], 7, "action.pose7")
    state[6] = _source_gripper_to_finger_m(
        float(state[6]),
        unit=state_gripper_unit,
        coordinate=state_gripper_coordinate,
        gripper_open_width_m=gripper_open_width_m,
    )
    explicit_finger = row.get("action.target_gripper_finger_position")
    if explicit_finger is not None and not pd.isna(explicit_finger):
        target_gripper = float(explicit_finger)
        legacy_finger = _source_gripper_to_finger_m(
            float(row["action.target_gripper_width"]),
            unit=action_gripper_unit,
            coordinate="total_width",
            gripper_open_width_m=gripper_open_width_m,
        )
        if not np.isclose(target_gripper, legacy_finger, rtol=0.0, atol=1e-7):
            raise ValueError(
                "action.target_gripper_finger_position does not equal "
                "action.target_gripper_width/2"
            )
    else:
        target_gripper = _source_gripper_to_finger_m(
            float(row["action.target_gripper_width"]),
            unit=action_gripper_unit,
            coordinate="total_width",
            gripper_open_width_m=gripper_open_width_m,
        )
    action = np.concatenate(
        [target_pose[:3], _quat_xyzw_to_rotvec(target_pose[3:7]), [target_gripper]]
    ).astype(np.float32)
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError("Tabero state/action contains non-finite values")
    max_finger_position = 0.5 * float(gripper_open_width_m)
    if state[6] < 0.0 or state[6] > max_finger_position + 1e-6:
        raise ValueError(f"state gripper finger position {state[6]} is outside physical range")
    if action[6] < 0.0 or action[6] > max_finger_position + 1e-6:
        raise ValueError(f"action gripper finger position {action[6]} is outside physical range")
    return state.astype(np.float32), action


def _source_gripper_to_finger_m(
    value: float,
    *,
    unit: str,
    coordinate: str,
    gripper_open_width_m: float,
) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid gripper value: {value!r}")
    if unit == "normalized":
        if value > 1.0 + 1e-6:
            raise ValueError(f"normalized gripper value outside [0,1]: {value}")
        return value * float(gripper_open_width_m) / 2.0
    if unit != "meter":
        raise ValueError(f"unsupported gripper unit: {unit!r}")
    if coordinate == "finger":
        return value
    if coordinate == "total_width":
        return value / 2.0
    raise ValueError(f"unsupported gripper coordinate: {coordinate!r}")


def _quat_xyzw_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"invalid target quaternion: {q.tolist()}")
    q /= norm
    if q[3] < 0:
        q = -q
    vector_norm = float(np.linalg.norm(q[:3]))
    if vector_norm < 1e-12:
        return (2.0 * q[:3]).astype(np.float32)
    angle = 2.0 * math.atan2(vector_norm, float(q[3]))
    return (q[:3] * (angle / vector_norm)).astype(np.float32)


def _sample_marker_positions(
    left_field: np.ndarray,
    right_field: np.ndarray,
    shear_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = _grid_indices(left_field.shape[0], GRID_ROWS)
    x = _grid_indices(left_field.shape[1], GRID_COLS)
    grid_x, grid_y = np.meshgrid(x.astype(np.float32), y.astype(np.float32))
    side_reference = np.stack([grid_x, grid_y], axis=-1).reshape(POINTS_PER_SIDE, 2)
    reference = np.concatenate([side_reference, side_reference], axis=0).astype(np.float32)
    left_shear = left_field[np.ix_(y, x)][..., :2].reshape(POINTS_PER_SIDE, 2)
    right_shear = right_field[np.ix_(y, x)][..., :2].reshape(POINTS_PER_SIDE, 2)
    shear = np.concatenate([left_shear, right_shear], axis=0).astype(np.float32)
    current = reference + np.float32(shear_scale) * shear
    if not np.all(np.isfinite(current)):
        raise ValueError("sampled tactile marker positions contain non-finite values")
    return reference, current.astype(np.float32)


def _grid_indices(size: int, count: int) -> np.ndarray:
    indices = np.rint(np.linspace(0, size - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise ValueError(f"cannot sample {count} unique points from size {size}")
    return indices


def _build_marker_history(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    if current.ndim != 3 or current.shape[1:] != (TOTAL_POINTS, 2):
        raise ValueError(f"current marker positions must be [T,{TOTAL_POINTS},2], got {current.shape}")
    output = np.empty((current.shape[0],) + MARKER_SHAPE, dtype=np.float32)
    output[:, 0] = reference
    for frame_index in range(current.shape[0]):
        for history_index in range(HISTORY_LENGTH):
            source_index = max(0, frame_index - (HISTORY_LENGTH - 1 - history_index))
            output[frame_index, 1 + history_index] = current[source_index]
    return output


def _v21_episode_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _v21_video_path(root: Path, video_key: str, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def _write_episode_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    arrays = [
        _fixed_vector_array(rows, STATE_KEY, STATE_DIM),
        _fixed_vector_array(rows, ACTION_KEY, ACTION_DIM),
        _nested_float_array(np.stack([row[MARKER_KEY] for row in rows]), MARKER_SHAPE),
        _fixed_vector_array(rows, WRENCH_KEY, WRENCH_DIM),
        _nested_float_array(np.stack([row[DEPTH_KEY] for row in rows]), DEPTH_SHAPE),
        pa.array([float(row["timestamp"]) for row in rows], type=pa.float32()),
        pa.array([int(row["frame_index"]) for row in rows], type=pa.int64()),
        pa.array([int(row["episode_index"]) for row in rows], type=pa.int64()),
        pa.array([int(row["index"]) for row in rows], type=pa.int64()),
        pa.array([int(row["task_index"]) for row in rows], type=pa.int64()),
    ]
    names = [
        STATE_KEY,
        ACTION_KEY,
        MARKER_KEY,
        WRENCH_KEY,
        DEPTH_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    table = pa.Table.from_arrays(arrays, names=names).replace_schema_metadata(_hf_schema_metadata())
    pq.write_table(table, path, compression="zstd")


def _fixed_vector_array(rows: list[dict[str, Any]], key: str, width: int) -> pa.Array:
    values = np.stack([np.asarray(row[key], dtype=np.float32).reshape(width) for row in rows])
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _nested_float_array(values: np.ndarray, shape: tuple[int, ...]) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[1:] != shape:
        raise ValueError(f"nested array shape {values.shape[1:]} != {shape}")
    array: pa.Array = pa.array(values.reshape(-1), type=pa.float32())
    for width in reversed(shape):
        offsets = pa.array(np.arange(0, len(array) + 1, width, dtype=np.int32))
        array = pa.ListArray.from_arrays(offsets, array)
    # Match Hugging Face datasets.Array3D's Arrow extension exactly. Keeping
    # both the extension metadata and the top-level `huggingface` schema
    # metadata lets a Tabero environment recover ndarray shapes directly.
    class _Array3DExtensionType(pa.ExtensionType):
        def __init__(self, array_shape: tuple[int, ...], dtype: str):
            self.shape = tuple(array_shape)
            self.value_type = dtype
            super().__init__(array.type, "datasets.features.features.Array3DExtensionType")

        def __arrow_ext_serialize__(self) -> bytes:
            return json.dumps((self.shape, self.value_type)).encode("utf-8")

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type: pa.DataType, serialized: bytes):
            array_shape, dtype = json.loads(serialized)
            return cls(tuple(array_shape), dtype)

    extension_type = _Array3DExtensionType(shape, "float32")
    return pa.ExtensionArray.from_storage(extension_type, array)


def _hf_schema_metadata() -> dict[bytes, bytes]:
    value = lambda dtype: {"dtype": dtype, "_type": "Value"}
    sequence = lambda length: {"feature": value("float32"), "length": length, "_type": "Sequence"}
    features: dict[str, Any] = {
        STATE_KEY: sequence(STATE_DIM),
        ACTION_KEY: sequence(ACTION_DIM),
        MARKER_KEY: {"shape": list(MARKER_SHAPE), "dtype": "float32", "_type": "Array3D"},
        WRENCH_KEY: sequence(WRENCH_DIM),
        DEPTH_KEY: {"shape": list(DEPTH_SHAPE), "dtype": "float32", "_type": "Array3D"},
        "timestamp": value("float32"),
        "frame_index": value("int64"),
        "episode_index": value("int64"),
        "index": value("int64"),
        "task_index": value("int64"),
    }
    return {b"huggingface": json.dumps({"info": {"features": features}}).encode("utf-8")}


def _write_path_video(paths: Iterable[Path], out_path: Path, fps: float, codec: str) -> None:
    paths = list(paths)
    if not paths:
        raise ValueError(f"no frames for {out_path}")
    with Image.open(paths[0]) as image:
        first = np.asarray(image.convert("RGB"))
    writer = _open_video_writer(out_path, first.shape, fps, codec)
    try:
        for path in paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            if rgb.shape != first.shape:
                raise ValueError(f"{path} shape {rgb.shape} != first frame {first.shape}")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _episode_numeric_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    # Depth is retained-only and deliberately omitted here: per-pixel JSON
    # statistics for [2,240,320] would dwarf the dataset metadata. The
    # action-only loader normalizes state/actions and consumes marker fields
    # directly; marker stats remain available for future tactile normalization.
    for key in (STATE_KEY, ACTION_KEY, MARKER_KEY, WRENCH_KEY):
        values = np.stack([np.asarray(row[key], dtype=np.float32) for row in rows])
        stats[key] = _basic_stats(values)
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64).reshape(-1, 1)
        stats[key] = _basic_stats(values)
    return stats


def _basic_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _build_info(
    *,
    raw_root: Path,
    raw_info: dict[str, Any],
    fps: float,
    total_frames: int,
    total_episodes: int,
    front_shape: list[int],
    wrist_shape: list[int],
    video_codec: str,
    gripper_output_unit: str,
    gripper_open_width_m: float,
    schema_versions: list[int],
) -> dict[str, Any]:
    features: dict[str, Any] = {
        STATE_KEY: {"dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES},
        ACTION_KEY: {"dtype": "float32", "shape": [ACTION_DIM], "names": ACTION_NAMES},
        MARKER_KEY: {"dtype": "float32", "shape": list(MARKER_SHAPE), "names": None},
        WRENCH_KEY: {"dtype": "float32", "shape": [WRENCH_DIM], "names": WRENCH_NAMES},
        DEPTH_KEY: {
            "dtype": "float32",
            "shape": list(DEPTH_SHAPE),
            "names": ["finger", "height", "width"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key, shape in ((IMAGE_KEY, front_shape), (WRIST_IMAGE_KEY, wrist_shape)):
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
                "video.fps": float(fps),
                "video.channels": 3,
                "has_audio": False,
            },
        }
    return {
        "codebase_version": "v2.1",
        "dataset_type": "tabero_action_only_lerobot_v2.1_dmtac_w",
        "source_dataset_type": raw_info.get("dataset_type"),
        "source_raw_root": str(raw_root),
        "robot_type": "franka",
        "fps": float(fps),
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": int(total_episodes * len(VIDEO_KEYS)),
        "total_chunks": int(math.ceil(total_episodes / 1000)),
        "chunks_size": 1000,
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "action_supervision": "actions[7] only",
        "wrench_supervision": False,
        "gripper_unit": gripper_output_unit,
        "gripper_coordinate": "single_finger_absolute_position",
        "gripper_open_width_m": float(gripper_open_width_m),
        "gripper_max_finger_position_m": float(gripper_open_width_m / 2.0),
        "tactile_msg_package": EXPECTED_TACTILE_PACKAGE,
        "source_tactile_schema_versions": schema_versions,
        "conversion_metadata_path": "meta/tabero_conversion.json",
    }


def validate_tabero_lerobot(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    info = _load_json(root / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Tabero export must use LeRobot codebase_version v2.1")
    if info.get("gripper_unit") != "meter":
        raise ValueError("Tabero gripper unit must be meter")
    if info.get("gripper_coordinate") != "single_finger_absolute_position":
        raise ValueError("Tabero gripper coordinate must be single_finger_absolute_position")
    expected_features = {
        STATE_KEY: [STATE_DIM],
        ACTION_KEY: [ACTION_DIM],
        MARKER_KEY: list(MARKER_SHAPE),
        WRENCH_KEY: [WRENCH_DIM],
        DEPTH_KEY: list(DEPTH_SHAPE),
    }
    for key, shape in expected_features.items():
        actual = info.get("features", {}).get(key, {}).get("shape")
        if actual != shape:
            raise ValueError(f"feature {key} shape {actual} != {shape}")

    episodes = _read_jsonlines(root / "meta" / "episodes.jsonl")
    tasks = _read_jsonlines(root / "meta" / "tasks.jsonl")
    stats = _read_jsonlines(root / "meta" / "episodes_stats.jsonl")
    if len(episodes) != int(info["total_episodes"]) or len(stats) != len(episodes):
        raise ValueError("episode metadata/stat count mismatch")
    if len(tasks) != int(info["total_tasks"]):
        raise ValueError("task metadata count mismatch")

    total_rows = 0
    for expected_episode, episode in enumerate(episodes):
        if int(episode["episode_index"]) != expected_episode:
            raise ValueError("episode indices must be contiguous from zero")
        path = _v21_episode_path(root, expected_episode)
        table = pq.read_table(path)
        length = int(episode["length"])
        if table.num_rows != length:
            raise ValueError(f"episode {expected_episode}: parquet rows {table.num_rows} != {length}")
        total_rows += length
        marker = np.asarray(table[MARKER_KEY].to_pylist(), dtype=np.float32)
        depth = np.asarray(table[DEPTH_KEY].to_pylist(), dtype=np.float32)
        states = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
        actions = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
        if marker.shape != (length,) + MARKER_SHAPE:
            raise ValueError(f"episode {expected_episode}: marker shape {marker.shape}")
        if depth.shape != (length,) + DEPTH_SHAPE:
            raise ValueError(f"episode {expected_episode}: depth shape {depth.shape}")
        if actions.shape != (length, ACTION_DIM):
            raise ValueError(f"episode {expected_episode}: action shape {actions.shape}")
        if states.shape != (length, STATE_DIM):
            raise ValueError(f"episode {expected_episode}: state shape {states.shape}")
        if not np.all(np.isfinite(marker)) or not np.all(np.isfinite(depth)):
            raise ValueError(f"episode {expected_episode}: non-finite tactile data")
        max_finger_position = float(info["gripper_max_finger_position_m"])
        for label, values in (("state", states[:, 6]), ("actions", actions[:, 6])):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"episode {expected_episode}: non-finite {label} gripper values")
            if np.any(values < -1e-6) or np.any(values > max_finger_position + 1e-6):
                raise ValueError(
                    f"episode {expected_episode}: {label}[6] outside physical single-finger range"
                )
        if length > 0 and not np.allclose(marker[0, 1:], marker[0, 1], rtol=0, atol=0):
            raise ValueError(f"episode {expected_episode}: initial marker history is not frame-0 padded")
        for video_key in VIDEO_KEYS:
            video_path = _v21_video_path(root, video_key, expected_episode)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video {video_path}")
            frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            if frame_count != length:
                raise ValueError(
                    f"episode {expected_episode}: {video_key} frames {frame_count} != {length}"
                )
    if total_rows != int(info["total_frames"]):
        raise ValueError(f"total frame count {total_rows} != {info['total_frames']}")
    return {
        "ok": True,
        "codebase_version": "v2.1",
        "episodes": len(episodes),
        "frames": total_rows,
        "action_dim": ACTION_DIM,
        "marker_shape": list(MARKER_SHAPE),
        "depth_shape": list(DEPTH_SHAPE),
    }


def _write_jsonlines(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _unique_ratio(values: list[int]) -> float:
    return float(len(set(values)) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
