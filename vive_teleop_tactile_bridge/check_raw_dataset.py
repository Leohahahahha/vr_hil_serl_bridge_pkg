#!/usr/bin/env python3
"""Sanity checker for raw tactile HIL-SERL ZED/D405 datasets."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from PIL import Image


REQUIRED_COLUMNS = [
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "observation.state",
    "action",
    "action.sent_action8",
    "image.path",
    "wrist_image.path",
    "tactile.left.zarr_index",
    "tactile.right.zarr_index",
    "sync.dt_image",
    "sync.dt_wrist_image",
    "sync.dt_action",
    "sync.dt_state",
    "sync.dt_tactile_left",
    "sync.dt_tactile_right",
]

VECTOR_COLUMNS = {
    "observation.state": 7,
    "action": 7,
    "action.sent_action8": 8,
    "action.pose7": 7,
    "robot.ee_pose": 7,
}

SYNC_COLUMNS = [
    "sync.dt_image",
    "sync.dt_wrist_image",
    "sync.dt_action",
    "sync.dt_state",
    "sync.dt_vr",
    "sync.dt_tactile_left",
    "sync.dt_tactile_right",
    "action.http_latency_ms",
    "robot.latency_ms",
]


@dataclass
class Reporter:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.error(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a raw ZED/D405 + tactile dataset before export/training.")
    parser.add_argument("--root", required=True, help="Raw dataset root")
    parser.add_argument(
        "--sample-images",
        type=int,
        default=5,
        help="Number of image files per camera to decode. Existence is checked for all rows.",
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as a non-zero exit.")
    args = parser.parse_args()

    report = Reporter()
    root = Path(args.root).expanduser().resolve()
    check_raw_dataset(root, max(0, int(args.sample_images)), report)

    print()
    print(f"warnings: {len(report.warnings)}")
    for message in report.warnings:
        print(f"  WARN: {message}")
    print(f"errors: {len(report.errors)}")
    for message in report.errors:
        print(f"  ERROR: {message}")

    if report.errors or (args.strict and report.warnings):
        raise SystemExit(1)
    print("OK")


def check_raw_dataset(root: Path, sample_images: int, report: Reporter) -> None:
    print(f"dataset: {root}")
    info = _read_json(root / "meta" / "info.json", report)
    if info:
        print(
            "meta: "
            f"dataset_type={info.get('dataset_type')}, "
            f"fps={info.get('fps')}, "
            f"tactile_msg_package={info.get('tactile_msg_package')}, "
            f"tactile_output_mode={info.get('tactile_output_mode')}, "
            f"tactile_schema_version={info.get('tactile_schema_version')}, "
            f"state_dim={info.get('state_dim')}, "
            f"action_dim={info.get('action_dim')}"
        )
        report.require(int(info.get("state_dim", -1)) == 7, "meta/info.json state_dim is not 7")
        report.require(int(info.get("action_dim", -1)) == 7, "meta/info.json action_dim is not 7")

    parquet_path = _storage_path(root, info, "master_parquet", "data/chunk-000/file-000.parquet")
    if not parquet_path.exists():
        report.error(f"missing master parquet: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    print(f"master_parquet: {parquet_path.relative_to(root)}")
    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")
    if len(df) == 0:
        report.error("master parquet has zero rows")
        return

    _check_columns(df, report)
    _check_indices(df, report)
    _check_vectors(df, report)
    _print_sync_stats(df)
    _print_http_stats(df)
    _check_episode_files(root, df, report)
    _check_images(root, df, "image", "image.path", sample_images, report)
    _check_images(root, df, "wrist_image", "wrist_image.path", sample_images, report)
    _check_tactile(root, df, "left", info, report)
    _check_tactile(root, df, "right", info, report)
    _check_dmtac_cross_side(df, info, report)


def _read_json(path: Path, report: Reporter) -> dict[str, Any]:
    if not path.exists():
        report.error(f"missing json file: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.error(f"failed to read {path}: {exc}")
        return {}


def _storage_path(root: Path, info: dict[str, Any], key: str, default: str) -> Path:
    storage = info.get("storage", {}) if isinstance(info, dict) else {}
    return root / str(storage.get(key, default))


def _check_columns(df: pd.DataFrame, report: Reporter) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    report.require(not missing, f"missing required columns: {missing}")

    optional = ["sync.valid", "action.http_ok", "robot.state_ok", "image.encoding", "wrist_image.encoding"]
    missing_optional = [col for col in optional if col not in df.columns]
    if missing_optional:
        report.warn(f"missing optional columns: {missing_optional}")


def _check_indices(df: pd.DataFrame, report: Reporter) -> None:
    if "index" in df.columns:
        index = pd.to_numeric(df["index"], errors="coerce").to_numpy(dtype=float)
        expected = np.arange(len(df), dtype=float)
        if not np.array_equal(index, expected):
            report.error("index column is not contiguous from 0 to rows-1")

    if "episode_index" not in df.columns or "frame_index" not in df.columns:
        return

    episode_counts: dict[int, int] = {}
    for episode_index, sub in df.groupby("episode_index", sort=True):
        ep = int(episode_index)
        frame_index = pd.to_numeric(sub["frame_index"], errors="coerce").to_numpy(dtype=int)
        expected = np.arange(len(sub), dtype=int)
        if not np.array_equal(frame_index, expected):
            report.error(f"frame_index is not contiguous for episode {ep}")
        episode_counts[ep] = int(len(sub))
    print(f"episodes_from_master: {episode_counts}")


def _check_vectors(df: pd.DataFrame, report: Reporter) -> None:
    for column, expected_dim in VECTOR_COLUMNS.items():
        if column not in df.columns:
            if column in ["observation.state", "action", "action.sent_action8"]:
                report.error(f"missing vector column: {column}")
            continue
        for row_index, value in enumerate(df[column]):
            if value is None:
                report.error(f"{column} row {row_index} is None")
                continue
            arr = np.asarray(value).reshape(-1)
            if arr.size != expected_dim:
                report.error(f"{column} row {row_index} dim {arr.size} != {expected_dim}")
                break
            if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr.astype(np.float64))):
                report.error(f"{column} row {row_index} contains non-finite values")
                break


def _print_sync_stats(df: pd.DataFrame) -> None:
    print("sync/latency stats:")
    printed = False
    for col in SYNC_COLUMNS:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if x.size == 0:
            continue
        printed = True
        print(
            f"  {col}: mean={np.mean(x):+.4f}, "
            f"p50={np.percentile(x, 50):+.4f}, "
            f"p95_abs={np.percentile(np.abs(x), 95):.4f}, "
            f"max_abs={np.max(np.abs(x)):.4f}"
        )
    if not printed:
        print("  N/A")


def _print_http_stats(df: pd.DataFrame) -> None:
    if "action.http_ok" in df.columns:
        ok_rate = float(df["action.http_ok"].astype(bool).mean())
        print(f"action.http_ok rate: {ok_rate:.3f}")
    if "sync.valid" in df.columns:
        valid_rate = float(df["sync.valid"].astype(bool).mean())
        print(f"sync.valid rate: {valid_rate:.3f}")


def _check_episode_files(root: Path, df: pd.DataFrame, report: Reporter) -> None:
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        report.error(f"missing episodes jsonl: {episodes_path}")
        return

    episodes = []
    for line_no, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            episodes.append(json.loads(line))
        except json.JSONDecodeError as exc:
            report.error(f"episodes.jsonl line {line_no} is invalid JSON: {exc}")
    print(f"episodes_jsonl: {len(episodes)}")

    master_counts = df.groupby("episode_index").size().to_dict() if "episode_index" in df.columns else {}
    episode_row_sum = 0
    for ep in episodes:
        episode_index = int(ep.get("episode_index", -1))
        num_frames = int(ep.get("num_frames", -1))
        data_path = root / str(ep.get("data_path", f"data/episodes/episode_{episode_index:06d}.parquet"))
        if not data_path.exists():
            report.error(f"missing per-episode parquet: {data_path}")
            continue
        ep_df = pd.read_parquet(data_path)
        episode_row_sum += len(ep_df)
        if len(ep_df) != num_frames:
            report.error(f"episode {episode_index} parquet rows {len(ep_df)} != episodes.jsonl num_frames {num_frames}")
        if episode_index in master_counts and int(master_counts[episode_index]) != len(ep_df):
            report.error(
                f"episode {episode_index} master rows {int(master_counts[episode_index])} "
                f"!= per-episode rows {len(ep_df)}"
            )
        if "episode_index" in ep_df.columns and not (ep_df["episode_index"].astype(int) == episode_index).all():
            report.error(f"episode {episode_index} parquet contains another episode_index")
        if "frame_index" in ep_df.columns:
            frame_index = ep_df["frame_index"].astype(int).to_numpy()
            if not np.array_equal(frame_index, np.arange(len(ep_df), dtype=int)):
                report.error(f"episode {episode_index} parquet frame_index is not contiguous")
    if episode_row_sum and episode_row_sum != len(df):
        report.error(f"sum of per-episode rows {episode_row_sum} != master rows {len(df)}")


def _check_images(
    root: Path,
    df: pd.DataFrame,
    label: str,
    path_column: str,
    sample_images: int,
    report: Reporter,
) -> None:
    if path_column not in df.columns:
        report.error(f"missing image path column: {path_column}")
        return

    paths = [root / str(p) for p in df[path_column].tolist()]
    missing = [path for path in paths if not path.exists()]
    if missing:
        report.error(f"{label}: {len(missing)} missing image files, first={missing[0]}")
        return
    print(f"{label}: {len(paths)} files exist")

    sample_rows = _sample_row_indices(len(df), sample_images)
    for row_index in sample_rows:
        path = paths[row_index]
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
        except Exception as exc:
            report.error(f"{label}: failed to open row {row_index} image {path}: {exc}")
            continue

        if mode != "RGB":
            report.error(f"{label}: row {row_index} image mode {mode} != RGB")
        _check_image_shape_columns(df, label, row_index, height, width, report)


def _sample_row_indices(row_count: int, sample_count: int) -> list[int]:
    if row_count <= 0 or sample_count <= 0:
        return []
    base = {0, row_count - 1}
    if sample_count > 2:
        for idx in np.linspace(0, row_count - 1, sample_count, dtype=int).tolist():
            base.add(int(idx))
    return sorted(base)


def _check_image_shape_columns(
    df: pd.DataFrame,
    label: str,
    row_index: int,
    height: int,
    width: int,
    report: Reporter,
) -> None:
    height_col = f"{label}.height"
    width_col = f"{label}.width"
    if height_col in df.columns:
        expected_height = int(df[height_col].iloc[row_index])
        if expected_height != height:
            report.error(f"{label}: row {row_index} image height {height} != column {expected_height}")
    if width_col in df.columns:
        expected_width = int(df[width_col].iloc[row_index])
        if expected_width != width:
            report.error(f"{label}: row {row_index} image width {width} != column {expected_width}")


def _check_tactile(root: Path, df: pd.DataFrame, side: str, info: dict[str, Any], report: Reporter) -> None:
    prefix = f"tactile.{side}"
    storage_key = f"tactile_{side}_zarr"
    default = f"tactile/tactile_{side}/data.zarr"
    zarr_path = _storage_path(root, info, storage_key, default)
    if not zarr_path.exists():
        report.error(f"{prefix}: missing zarr store {zarr_path}")
        return

    try:
        arr = zarr.open(str(zarr_path), mode="r")
    except Exception as exc:
        report.error(f"{prefix}: failed to open zarr {zarr_path}: {exc}")
        return

    print(f"{prefix}: zarr_shape={tuple(arr.shape)}, dtype={arr.dtype}, chunks={getattr(arr, 'chunks', None)}")
    index_col = f"{prefix}.zarr_index"
    if index_col not in df.columns:
        report.error(f"{prefix}: missing {index_col}")
        return
    indices = pd.to_numeric(df[index_col], errors="coerce")
    if indices.isna().any():
        report.error(f"{prefix}: {int(indices.isna().sum())} null/non-numeric zarr indices")
        return
    idx = indices.to_numpy(dtype=int)
    if idx.size:
        if idx.min() < 0 or idx.max() >= arr.shape[0]:
            report.error(f"{prefix}: zarr indices out of range [0, {arr.shape[0]})")
        if len(set(idx.tolist())) != len(idx):
            report.warn(f"{prefix}: zarr indices are not unique")

    data_len_col = f"{prefix}.data_len"
    if data_len_col in df.columns and len(arr.shape) >= 2:
        expected_dim = int(arr.shape[1])
        data_lens = pd.to_numeric(df[data_len_col], errors="coerce").dropna().astype(int).unique().tolist()
        if data_lens != [expected_dim]:
            report.error(f"{prefix}: data_len values {data_lens} != zarr dim {expected_dim}")

    dtype_col = f"{prefix}.data_dtype"
    if dtype_col in df.columns:
        dtypes = sorted({str(v) for v in df[dtype_col].dropna().unique().tolist()})
        if dtypes and dtypes != [str(arr.dtype)]:
            report.error(f"{prefix}: data_dtype values {dtypes} != zarr dtype {arr.dtype}")

    _check_tactile_package_expectation(prefix, info, arr, df, report)


def _check_tactile_package_expectation(
    prefix: str,
    info: dict[str, Any],
    arr: Any,
    df: pd.DataFrame,
    report: Reporter,
) -> None:
    package = info.get("tactile_msg_package")
    if not package:
        return
    package = str(package)
    if package == "tashan_tactile":
        if tuple(arr.shape[1:]) != (25,):
            report.warn(f"{prefix}: tashan expected frame dim 25, got {tuple(arr.shape[1:])}")
        if str(arr.dtype) != "float32":
            report.warn(f"{prefix}: tashan expected float32 zarr, got {arr.dtype}")
    elif package == "paxini_tactile":
        if tuple(arr.shape[1:]) != (234,):
            report.warn(f"{prefix}: paxini expected frame dim 234, got {tuple(arr.shape[1:])}")
        if str(arr.dtype) != "uint8":
            report.warn(f"{prefix}: paxini expected uint8 zarr, got {arr.dtype}")
    elif package == "dmtac_tactile":
        if str(arr.dtype) != "uint8":
            report.warn(f"{prefix}: DM-Tac expected uint8 packed zarr, got {arr.dtype}")
        _check_dmtac_layout(prefix, arr, df, info, report)


def _check_dmtac_layout(
    prefix: str,
    arr: Any,
    df: pd.DataFrame,
    info: dict[str, Any],
    report: Reporter,
) -> None:
    schema_column = f"{prefix}.schema_version"
    packed_column = f"{prefix}.packed_frame_bytes"
    if schema_column not in df.columns or packed_column not in df.columns:
        report.error(
            f"{prefix}: missing DM-Tac schema columns: "
            f"{[c for c in (schema_column, packed_column) if c not in df.columns]}"
        )
        return

    schema_values = (
        pd.to_numeric(df[schema_column], errors="coerce").dropna().astype(int).unique().tolist()
    )
    if len(schema_values) != 1:
        report.error(f"{prefix}: schema_version is not constant: {schema_values}")
        return
    schema_version = int(schema_values[0])
    if schema_version == 1:
        segments = ("raw_image", "deformation2d", "normal", "shear", "depth")
        expected_layout = {
            "raw_image": (240, 320, 1, 1),
            "deformation2d": (240, 320, 2, 4),
            "normal": (240, 320, 1, 4),
            "shear": (240, 320, 2, 4),
            "depth": (240, 320, 1, 4),
        }
        expected_packed_bytes = 1_920_000
    elif schema_version == 2:
        segments = (
            "raw_image",
            "infer_image",
            "deformation2d",
            "depth",
            "shear",
            "distributed_force",
            "contact_area",
            "wrench",
        )
        expected_layout = None
        expected_packed_bytes = None
        expected_output_mode = None
    elif schema_version == 3:
        segments = ("shear", "depth")
        expected_layout = {
            "shear": (240, 320, 2, 4),
            "depth": (240, 320, 1, 4),
        }
        expected_packed_bytes = 921_600
        expected_output_mode = "shear_depth"
    else:
        report.error(f"{prefix}: unsupported DM-Tac schema version {schema_version}")
        return

    if schema_version == 1:
        expected_output_mode = "full"

    layout_fields = ("start", "len", "height", "width", "channels", "itemsize")
    required = [schema_column, packed_column]
    required.extend(
        f"{prefix}.{segment}_{field}"
        for segment in segments
        for field in layout_fields
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        report.error(
            f"{prefix}: missing DM-Tac schema-v{schema_version} layout columns: {missing}"
        )
        return

    values: dict[str, int] = {}
    for column in required:
        unique = pd.to_numeric(df[column], errors="coerce").dropna().astype(int).unique().tolist()
        if len(unique) != 1:
            report.error(f"{prefix}: layout column {column} is not constant: {unique}")
            return
        values[column] = int(unique[0])

    expected_start = 0
    for segment in segments:
        start = values[f"{prefix}.{segment}_start"]
        length = values[f"{prefix}.{segment}_len"]
        height = values[f"{prefix}.{segment}_height"]
        width = values[f"{prefix}.{segment}_width"]
        channels = values[f"{prefix}.{segment}_channels"]
        itemsize = values[f"{prefix}.{segment}_itemsize"]
        if start != expected_start:
            report.error(
                f"{prefix}: {segment} starts at {start}, expected contiguous offset {expected_start}"
            )
        expected_length = height * width * channels * itemsize
        if length != expected_length:
            report.error(
                f"{prefix}: {segment} byte length {length} != shape-derived {expected_length}"
            )
        expected_start = start + length
        if expected_layout is not None:
            actual_layout = (height, width, channels, itemsize)
            if actual_layout != expected_layout[segment]:
                report.error(
                    f"{prefix}: {segment} layout {actual_layout} != verified "
                    f"DM-Tac W layout {expected_layout[segment]}"
                )

    packed_bytes = values[f"{prefix}.packed_frame_bytes"]
    if expected_start != packed_bytes:
        report.error(
            f"{prefix}: final layout byte {expected_start} != packed_frame_bytes {packed_bytes}"
        )
    if len(arr.shape) != 2 or int(arr.shape[1]) != packed_bytes:
        report.error(f"{prefix}: zarr frame size {tuple(arr.shape[1:])} != {packed_bytes}")
    if expected_packed_bytes is not None and packed_bytes != expected_packed_bytes:
        report.error(
            f"{prefix}: packed_frame_bytes {packed_bytes} != verified "
            f"DM-Tac W size {expected_packed_bytes}"
        )

    # The explicit format metadata was introduced with schema 3. Keep legacy
    # schema-1/2 datasets valid when these keys are absent, but validate them
    # whenever present.
    expected_info = {
        "tactile_schema_version": schema_version,
        "tactile_packed_frame_bytes": packed_bytes,
    }
    if expected_output_mode is not None:
        expected_info["tactile_output_mode"] = expected_output_mode
    for key, expected_value in expected_info.items():
        if key not in info:
            if schema_version == 3:
                report.error(f"{prefix}: schema 3 requires meta/info.json {key}")
            continue
        if info[key] != expected_value:
            report.error(
                f"{prefix}: meta/info.json {key}={info[key]!r} != {expected_value!r}"
            )


def _check_dmtac_cross_side(
    df: pd.DataFrame,
    info: dict[str, Any],
    report: Reporter,
) -> None:
    """Reject a dataset whose two tactile sides use different packed schemas."""
    if str(info.get("tactile_msg_package", "")) != "dmtac_tactile":
        return
    for field in ("schema_version", "packed_frame_bytes"):
        side_values: dict[str, list[int]] = {}
        for side in ("left", "right"):
            column = f"tactile.{side}.{field}"
            if column not in df.columns:
                continue
            side_values[side] = sorted(
                pd.to_numeric(df[column], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
        if len(side_values) == 2 and side_values["left"] != side_values["right"]:
            report.error(
                f"DM-Tac left/right {field} mismatch: "
                f"left={side_values['left']}, right={side_values['right']}"
            )


if __name__ == "__main__":
    main()
