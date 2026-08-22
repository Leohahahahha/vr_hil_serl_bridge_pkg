from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .common import (
    ACTION_KEY,
    FRONT_VIDEO_KEY,
    STATE_KEY,
    TACTILE_LEFT_KEY,
    TACTILE_RIGHT_KEY,
    WRIST_VIDEO_KEY,
)


TIMESTAMP_ABS_TOL_SEC = 1e-5


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a LeRobot v3-style tactile dataset export.")
    parser.add_argument("--dataset-root", required=True, help="Exported LeRobot v3-style dataset root")
    parser.add_argument("--raw-root", default=None, help="Optional source raw dataset root for end-to-end alignment checks")
    parser.add_argument(
        "--video-samples",
        type=int,
        default=5,
        help="Number of decoded MP4 frames per camera to compare against raw PNG when --raw-root is set",
    )
    parser.add_argument(
        "--max-video-mae",
        type=float,
        default=12.0,
        help="Maximum allowed mean absolute pixel error for MP4-vs-PNG sample checks",
    )
    args = parser.parse_args()
    result = validate_dataset(
        Path(args.dataset_root),
        raw_root=None if args.raw_root is None else Path(args.raw_root),
        video_samples=max(0, int(args.video_samples)),
        max_video_mae=float(args.max_video_mae),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def validate_dataset(
    dataset_root: Path,
    *,
    raw_root: Path | None = None,
    video_samples: int = 5,
    max_video_mae: float = 12.0,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    stats_path = dataset_root / "meta" / "stats.json"

    required_paths = [info_path, data_path, episodes_path, tasks_path, stats_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    arrow_schema = pq.read_schema(data_path)
    df = pd.read_parquet(data_path)
    episodes = pd.read_parquet(episodes_path)
    tasks = pd.read_parquet(tasks_path)

    total_frames = int(info.get("total_frames", -1))
    total_episodes = int(info.get("total_episodes", -1))
    if len(df) != total_frames:
        raise ValueError(f"data rows {len(df)} != info.total_frames {total_frames}")
    if len(episodes) != total_episodes:
        raise ValueError(f"episode rows {len(episodes)} != info.total_episodes {total_episodes}")
    if tasks.empty:
        raise ValueError("tasks.parquet is empty")

    _require_columns(
        df,
        [
            "index",
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
            STATE_KEY,
            ACTION_KEY,
            TACTILE_LEFT_KEY,
            TACTILE_RIGHT_KEY,
        ],
    )
    _require_columns(
        episodes,
        [
            "episode_index",
            "length",
            "from",
            "to",
            "dataset_from_index",
            "dataset_to_index",
            "task_index",
            "tasks",
        ],
    )

    _check_vector_column(df, STATE_KEY, 7)
    _check_vector_column(df, ACTION_KEY, 7)
    tactile_left_dim = _feature_dim(info, TACTILE_LEFT_KEY)
    tactile_right_dim = _feature_dim(info, TACTILE_RIGHT_KEY)
    _check_low_dim_schema(arrow_schema, STATE_KEY, 7)
    _check_low_dim_schema(arrow_schema, ACTION_KEY, 7)
    _check_low_dim_schema(arrow_schema, TACTILE_LEFT_KEY, tactile_left_dim)
    _check_low_dim_schema(arrow_schema, TACTILE_RIGHT_KEY, tactile_right_dim)
    _check_vector_column(df, TACTILE_LEFT_KEY, tactile_left_dim)
    _check_vector_column(df, TACTILE_RIGHT_KEY, tactile_right_dim)
    _check_episode_ranges(df, episodes)
    _check_index_and_timestamps(df, float(info.get("fps", 10.0)))

    video_results = {}
    for key in [FRONT_VIDEO_KEY, WRIST_VIDEO_KEY]:
        video_path = dataset_root / "videos" / key / "chunk-000" / "file-000.mp4"
        expected_shape = info["features"][key]["shape"]
        video_results[key] = _check_video(video_path, len(df), expected_shape)

    raw_result = None
    if raw_root is not None:
        raw_result = _check_against_raw(
            dataset_root=dataset_root,
            raw_root=raw_root.expanduser().resolve(),
            df=df,
            episodes=episodes,
            video_samples=video_samples,
            max_video_mae=max_video_mae,
        )

    result: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "ok": True,
        "total_frames": len(df),
        "total_episodes": len(episodes),
        "task_count": len(tasks),
        "state_dim": 7,
        "action_dim": 7,
        "tactile_left_dim": tactile_left_dim,
        "tactile_right_dim": tactile_right_dim,
        "videos": video_results,
    }
    if raw_result is not None:
        result["raw_alignment"] = raw_result
    return result


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _feature_dim(info: dict[str, Any], key: str) -> int:
    shape = info["features"][key]["shape"]
    dtype = str(info["features"][key].get("dtype"))
    if dtype != "float32":
        raise ValueError(f"{key} metadata dtype must be float32, got {dtype!r}")
    if len(shape) != 1:
        raise ValueError(f"{key} expected 1D shape, got {shape}")
    return int(shape[0])


def _check_low_dim_schema(schema: pa.Schema, column: str, expected_dim: int) -> None:
    field_type = schema.field(column).type
    if not pa.types.is_fixed_size_list(field_type):
        raise ValueError(f"{column} parquet type must be fixed_size_list<float32>[{expected_dim}], got {field_type}")
    if int(field_type.list_size) != int(expected_dim):
        raise ValueError(f"{column} parquet list size {field_type.list_size} != {expected_dim}")
    if field_type.value_type != pa.float32():
        raise ValueError(f"{column} parquet value type must be float32, got {field_type.value_type}")


def _check_vector_column(df: pd.DataFrame, column: str, expected_dim: int) -> None:
    for row_index, value in enumerate(df[column]):
        arr = np.asarray(value).reshape(-1)
        if arr.size != expected_dim:
            raise ValueError(f"{column} row {row_index} dim {arr.size} != {expected_dim}")
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr.astype(np.float64))):
            raise ValueError(f"{column} row {row_index} contains non-finite values")


def _check_episode_ranges(df: pd.DataFrame, episodes: pd.DataFrame) -> None:
    total_rows = len(df)
    for _, ep in episodes.iterrows():
        start = int(ep["from"])
        end = int(ep["to"])
        dataset_start = int(ep["dataset_from_index"])
        dataset_end = int(ep["dataset_to_index"])
        length = int(ep["length"])
        episode_index = int(ep["episode_index"])
        if (start, end) != (dataset_start, dataset_end):
            raise ValueError(
                f"episode {episode_index} range mismatch: from/to={start}:{end}, "
                f"dataset_from/to={dataset_start}:{dataset_end}"
            )
        if start < 0 or end > total_rows or start >= end:
            raise ValueError(f"invalid episode range for episode {episode_index}: {start}:{end}")
        if end - start != length:
            raise ValueError(f"episode {episode_index} length mismatch: {length} vs {end - start}")
        sub = df.iloc[start:end]
        if not (sub["episode_index"].astype(int).to_numpy() == episode_index).all():
            raise ValueError(f"episode_index mismatch inside range {start}:{end}")
        frame_index = sub["frame_index"].astype(int).to_numpy()
        expected = np.arange(length, dtype=int)
        if not np.array_equal(frame_index, expected):
            raise ValueError(f"frame_index is not contiguous for episode {episode_index}")


def _check_index_and_timestamps(df: pd.DataFrame, fps: float) -> None:
    index = df["index"].astype(int).to_numpy()
    if not np.array_equal(index, np.arange(len(df), dtype=int)):
        raise ValueError("index column is not contiguous from 0 to total_frames-1")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid fps in info.json: {fps}")

    for episode_index, sub in df.groupby("episode_index", sort=True):
        frame_index = sub["frame_index"].astype(int).to_numpy()
        expected_timestamp = frame_index.astype(np.float64) / float(fps)
        timestamp = pd.to_numeric(sub["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.allclose(timestamp, expected_timestamp, atol=TIMESTAMP_ABS_TOL_SEC, rtol=0):
            raise ValueError(f"timestamp != frame_index/fps for episode {int(episode_index)}")


def _check_video(video_path: Path, expected_frames: int, expected_shape: list[int]) -> dict[str, Any]:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to decode first frame: {video_path}")
    if frame_count != expected_frames:
        raise ValueError(f"{video_path} frame count {frame_count} != expected {expected_frames}")
    expected_height, expected_width, expected_channels = [int(x) for x in expected_shape]
    if (height, width) != (expected_height, expected_width):
        raise ValueError(f"{video_path} size {(height, width)} != expected {(expected_height, expected_width)}")
    if frame.shape[2] != expected_channels:
        raise ValueError(f"{video_path} channel count {frame.shape[2]} != expected {expected_channels}")
    return {
        "path": str(video_path),
        "frames": frame_count,
        "height": height,
        "width": width,
        "channels": int(frame.shape[2]),
    }


def _check_against_raw(
    *,
    dataset_root: Path,
    raw_root: Path,
    df: pd.DataFrame,
    episodes: pd.DataFrame,
    video_samples: int,
    max_video_mae: float,
) -> dict[str, Any]:
    raw_rows = _raw_rows_in_export_order(raw_root, episodes)
    if len(raw_rows) != len(df):
        raise ValueError(f"raw rows {len(raw_rows)} != exported rows {len(df)}")

    raw_df = pd.DataFrame(raw_rows).reset_index(drop=True)
    _compare_vector_frames(raw_df, df, STATE_KEY, 7, atol=1e-6)
    _compare_vector_frames(raw_df, df, ACTION_KEY, 7, atol=1e-6)

    tactile_left = _open_zarr(raw_root / "tactile" / "tactile_left" / "data.zarr")
    tactile_right = _open_zarr(raw_root / "tactile" / "tactile_right" / "data.zarr")
    _compare_tactile(raw_df, df, tactile_left, "tactile.left.zarr_index", TACTILE_LEFT_KEY)
    _compare_tactile(raw_df, df, tactile_right, "tactile.right.zarr_index", TACTILE_RIGHT_KEY)

    front_paths = [_resolve_sidecar_path(raw_root, value) for value in raw_df["image.path"].tolist()]
    wrist_paths = [_resolve_sidecar_path(raw_root, value) for value in raw_df["wrist_image.path"].tolist()]
    front_video = dataset_root / "videos" / FRONT_VIDEO_KEY / "chunk-000" / "file-000.mp4"
    wrist_video = dataset_root / "videos" / WRIST_VIDEO_KEY / "chunk-000" / "file-000.mp4"

    front_cmp = _compare_video_to_png_samples(
        front_video,
        front_paths,
        video_key=FRONT_VIDEO_KEY,
        sample_count=video_samples,
        max_video_mae=max_video_mae,
    )
    wrist_cmp = _compare_video_to_png_samples(
        wrist_video,
        wrist_paths,
        video_key=WRIST_VIDEO_KEY,
        sample_count=video_samples,
        max_video_mae=max_video_mae,
    )

    return {
        "raw_root": str(raw_root),
        "frames_checked": int(len(df)),
        "episodes_checked": int(len(episodes)),
        "state": "match",
        "action": "match",
        "tactile_left": {
            "shape": list(tactile_left.shape),
            "dtype": str(tactile_left.dtype),
            "status": "match",
        },
        "tactile_right": {
            "shape": list(tactile_right.shape),
            "dtype": str(tactile_right.dtype),
            "status": "match",
        },
        "videos": {
            FRONT_VIDEO_KEY: front_cmp,
            WRIST_VIDEO_KEY: wrist_cmp,
        },
    }


def _raw_rows_in_export_order(raw_root: Path, episodes: pd.DataFrame) -> list[dict[str, Any]]:
    if "raw_episode" not in episodes.columns:
        raise ValueError("episodes parquet has no raw_episode column; cannot compare with --raw-root")

    rows: list[dict[str, Any]] = []
    cursors: dict[str, int] = {}
    for _, ep in episodes.sort_values("episode_index").iterrows():
        raw_episode = str(ep["raw_episode"])
        length = int(ep["length"])
        raw_episode_path = raw_root / raw_episode
        if not raw_episode_path.exists():
            raise FileNotFoundError(raw_episode_path)
        raw_df = pd.read_parquet(raw_episode_path)
        start = cursors.get(raw_episode, 0)
        end = start + length
        if end > len(raw_df):
            raise ValueError(f"{raw_episode} does not have enough rows for exported segment {start}:{end}")
        segment = raw_df.iloc[start:end].reset_index(drop=True)
        cursors[raw_episode] = end

        episode_index = int(ep["episode_index"])
        if len(segment) != length:
            raise ValueError(f"raw segment for episode {episode_index} has {len(segment)} rows, expected {length}")
        rows.extend(segment.to_dict("records"))
    return rows


def _compare_vector_frames(
    raw_df: pd.DataFrame,
    exported_df: pd.DataFrame,
    key: str,
    expected_dim: int,
    *,
    atol: float,
) -> None:
    raw_arr = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in raw_df[key]])
    exported_arr = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in exported_df[key]])
    if raw_arr.shape != (len(raw_df), expected_dim):
        raise ValueError(f"raw {key} shape {raw_arr.shape} != {(len(raw_df), expected_dim)}")
    if exported_arr.shape != (len(exported_df), expected_dim):
        raise ValueError(f"exported {key} shape {exported_arr.shape} != {(len(exported_df), expected_dim)}")
    if not np.allclose(raw_arr, exported_arr, atol=atol, rtol=0):
        diff = np.abs(raw_arr - exported_arr)
        row, dim = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise ValueError(f"{key} mismatch at row {row}, dim {dim}: max_abs={float(diff[row, dim])}")


def _open_zarr(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    import zarr

    return zarr.open(str(path), mode="r")


def _compare_tactile(
    raw_df: pd.DataFrame,
    exported_df: pd.DataFrame,
    zarr_array: Any,
    index_column: str,
    exported_key: str,
) -> None:
    expected = []
    for row_index, value in enumerate(raw_df[index_column]):
        idx = int(value)
        if idx < 0 or idx >= int(zarr_array.shape[0]):
            raise ValueError(f"{index_column} row {row_index} zarr index {idx} out of range")
        expected.append(np.asarray(zarr_array[idx]).reshape(-1))
    expected_arr = np.stack(expected)
    exported_arr = np.stack([np.asarray(value).reshape(-1) for value in exported_df[exported_key]])
    if expected_arr.shape != exported_arr.shape:
        raise ValueError(f"{exported_key} shape {exported_arr.shape} != raw zarr shape {expected_arr.shape}")
    if not np.allclose(expected_arr.astype(np.float32), exported_arr.astype(np.float32), atol=1e-6, rtol=0):
        diff = np.abs(expected_arr.astype(np.float32) - exported_arr.astype(np.float32))
        row, dim = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise ValueError(f"{exported_key} mismatch at row {row}, dim {dim}: max_abs={float(diff[row, dim])}")


def _resolve_sidecar_path(raw_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return raw_root / path


def _compare_video_to_png_samples(
    video_path: Path,
    raw_paths: list[Path],
    *,
    video_key: str,
    sample_count: int,
    max_video_mae: float,
) -> dict[str, Any]:
    sample_indices = _sample_indices(len(raw_paths), sample_count)
    if not sample_indices:
        return {"samples_checked": 0, "max_mae": None, "max_abs": None}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    maes: list[float] = []
    max_abs_values: list[int] = []
    try:
        for frame_index in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"{video_key}: failed to decode video frame {frame_index}")
            decoded_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_rgb = np.asarray(Image.open(raw_paths[frame_index]).convert("RGB"))
            if decoded_rgb.shape != raw_rgb.shape:
                raise ValueError(f"{video_key}: frame {frame_index} video shape {decoded_rgb.shape} != raw {raw_rgb.shape}")

            diff = decoded_rgb.astype(np.int16) - raw_rgb.astype(np.int16)
            mae = float(np.mean(np.abs(diff)))
            max_abs = int(np.max(np.abs(diff)))
            swapped_mae = float(np.mean(np.abs(decoded_rgb[..., ::-1].astype(np.int16) - raw_rgb.astype(np.int16))))
            if swapped_mae + 1.0 < mae:
                raise ValueError(
                    f"{video_key}: frame {frame_index} looks channel-swapped "
                    f"(rgb_mae={mae:.3f}, swapped_mae={swapped_mae:.3f})"
                )
            if mae > max_video_mae:
                raise ValueError(f"{video_key}: frame {frame_index} MAE {mae:.3f} > {max_video_mae:.3f}")
            maes.append(mae)
            max_abs_values.append(max_abs)
    finally:
        cap.release()

    return {
        "samples_checked": int(len(sample_indices)),
        "sample_indices": [int(i) for i in sample_indices],
        "max_mae": float(max(maes)),
        "mean_mae": float(np.mean(maes)),
        "max_abs": int(max(max_abs_values)),
    }


def _sample_indices(row_count: int, sample_count: int) -> list[int]:
    if row_count <= 0 or sample_count <= 0:
        return []
    if sample_count == 1:
        return [0]
    return sorted({int(i) for i in np.linspace(0, row_count - 1, sample_count, dtype=int).tolist()})


if __name__ == "__main__":
    main()
