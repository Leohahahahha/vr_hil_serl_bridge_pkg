from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from ..action_space import ACTION_NAMES, STATE_NAMES


FRONT_VIDEO_KEY = "observation.images.front"
WRIST_VIDEO_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TACTILE_LEFT_KEY = "observation.tactile_left"
TACTILE_RIGHT_KEY = "observation.tactile_right"
STATE_DIM = 7
ACTION_DIM = 7


@dataclass(frozen=True)
class ExportProfile:
    name: str
    tactile_msg_package: str
    tactile_description: str


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--raw-root", required=True, help="Raw dataset root")
    parser.add_argument("--out-root", required=True, help="Output LeRobot v3-style dataset root")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS. Defaults to raw meta/info.json fps")
    parser.add_argument("--task", default=None, help="Override task text. Defaults to raw meta/info.json task_description")
    parser.add_argument("--split-on-gap", action="store_true", help="Optionally split raw episodes when timestamp gaps are large")
    parser.add_argument("--gap-factor", type=float, default=1.5, help="With --split-on-gap, split when timestamp gap exceeds gap_factor / fps")
    parser.add_argument("--overwrite", action="store_true", help="Delete output root before writing")
    return parser


def export_main(profile: ExportProfile, description: str) -> None:
    parser = build_arg_parser(description)
    args = parser.parse_args()
    result = export_raw_to_lerobot_v3(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        profile=profile,
        fps=args.fps,
        task=args.task,
        split_on_gap=args.split_on_gap,
        gap_factor=args.gap_factor,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def export_raw_to_lerobot_v3(
    *,
    raw_root: Path,
    out_root: Path,
    profile: ExportProfile,
    fps: float | None = None,
    task: str | None = None,
    split_on_gap: bool = False,
    gap_factor: float = 1.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    raw_root = raw_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    raw_info = _load_json(raw_root / "meta" / "info.json")

    raw_tactile_package = raw_info.get("tactile_msg_package")
    if raw_tactile_package != profile.tactile_msg_package:
        raise ValueError(
            f"{profile.name} exporter requires tactile_msg_package={profile.tactile_msg_package}, "
            f"got {raw_tactile_package!r} from {raw_root / 'meta' / 'info.json'}"
        )

    fps_value = float(fps if fps is not None else raw_info.get("fps", 10.0))
    if not np.isfinite(fps_value) or fps_value <= 0:
        raise ValueError(f"invalid fps: {fps_value}")
    task_text = str(task if task is not None else raw_info.get("task_description", "teleoperation"))

    _prepare_output_root(out_root, overwrite=overwrite)
    data_path = out_root / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_path = out_root / "meta" / "tasks.parquet"
    stats_path = out_root / "meta" / "stats.json"
    info_path = out_root / "meta" / "info.json"
    front_video_path = out_root / "videos" / FRONT_VIDEO_KEY / "chunk-000" / "file-000.mp4"
    wrist_video_path = out_root / "videos" / WRIST_VIDEO_KEY / "chunk-000" / "file-000.mp4"

    for path in [
        data_path.parent,
        episodes_path.parent,
        tasks_path.parent,
        front_video_path.parent,
        wrist_video_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    tactile_left = _open_zarr_array(raw_root / "tactile" / "tactile_left" / "data.zarr")
    tactile_right = _open_zarr_array(raw_root / "tactile" / "tactile_right" / "data.zarr")
    tactile_left_shape = _flat_shape(tactile_left.shape)
    tactile_right_shape = _flat_shape(tactile_right.shape)

    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    front_paths: list[Path] = []
    wrist_paths: list[Path] = []
    raw_episode_count = 0
    split_count = 0
    global_index = 0
    export_episode_index = 0

    episode_paths = _find_episode_parquets(raw_root)
    if not episode_paths:
        raise FileNotFoundError(f"no raw episode parquet files under {raw_root / 'data' / 'episodes'}")

    for raw_episode_path in episode_paths:
        raw_episode_count += 1
        df = pd.read_parquet(raw_episode_path)
        _require_columns(
            df,
            [
                "timestamp",
                STATE_KEY,
                ACTION_KEY,
                "image.path",
                "wrist_image.path",
                "tactile.left.zarr_index",
                "tactile.right.zarr_index",
            ],
            raw_episode_path,
        )
        segments = _episode_segments(df, fps_value, gap_factor, split_on_gap=split_on_gap)
        for segment_index, segment in enumerate(segments):
            if segment.empty:
                continue
            if segment_index > 0:
                split_count += 1
            episode_start = global_index
            frame_index = 0
            for _, raw_row in segment.iterrows():
                try:
                    state = _vector(raw_row[STATE_KEY], STATE_DIM, STATE_KEY)
                    action = _vector(raw_row[ACTION_KEY], ACTION_DIM, ACTION_KEY)
                    left_idx = int(raw_row["tactile.left.zarr_index"])
                    right_idx = int(raw_row["tactile.right.zarr_index"])
                    tactile_l = _zarr_vector(tactile_left, left_idx, TACTILE_LEFT_KEY)
                    tactile_r = _zarr_vector(tactile_right, right_idx, TACTILE_RIGHT_KEY)
                    front_path = _resolve_sidecar_path(raw_root, raw_row["image.path"])
                    wrist_path = _resolve_sidecar_path(raw_root, raw_row["wrist_image.path"])
                    if not front_path.exists():
                        raise FileNotFoundError(front_path)
                    if not wrist_path.exists():
                        raise FileNotFoundError(wrist_path)
                except Exception as e:
                    raise RuntimeError(f"failed to convert raw row from {raw_episode_path}: {e}") from e

                rows.append(
                    {
                        "index": int(global_index),
                        "episode_index": int(export_episode_index),
                        "frame_index": int(frame_index),
                        "timestamp": float(frame_index / fps_value),
                        "task_index": 0,
                        STATE_KEY: _float32_list(state),
                        ACTION_KEY: _float32_list(action),
                        TACTILE_LEFT_KEY: _float32_list(tactile_l),
                        TACTILE_RIGHT_KEY: _float32_list(tactile_r),
                    }
                )
                front_paths.append(front_path)
                wrist_paths.append(wrist_path)
                global_index += 1
                frame_index += 1

            episode_end = global_index
            episode_rows.append(
                {
                    "episode_index": int(export_episode_index),
                    "tasks": [task_text],
                    "length": int(episode_end - episode_start),
                    "from": int(episode_start),
                    "to": int(episode_end),
                    "dataset_from_index": int(episode_start),
                    "dataset_to_index": int(episode_end),
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    f"videos/{FRONT_VIDEO_KEY}/chunk_index": 0,
                    f"videos/{FRONT_VIDEO_KEY}/file_index": 0,
                    f"videos/{FRONT_VIDEO_KEY}/from_timestamp": float(episode_start / fps_value),
                    f"videos/{FRONT_VIDEO_KEY}/to_timestamp": float(episode_end / fps_value),
                    f"videos/{WRIST_VIDEO_KEY}/chunk_index": 0,
                    f"videos/{WRIST_VIDEO_KEY}/file_index": 0,
                    f"videos/{WRIST_VIDEO_KEY}/from_timestamp": float(episode_start / fps_value),
                    f"videos/{WRIST_VIDEO_KEY}/to_timestamp": float(episode_end / fps_value),
                    "task_index": 0,
                    "raw_episode": str(raw_episode_path.relative_to(raw_root)),
                }
            )
            export_episode_index += 1

    if not rows:
        raise RuntimeError(f"no rows exported from {raw_root}")

    front_shape = _image_shape(front_paths[0])
    wrist_shape = _image_shape(wrist_paths[0])
    _write_data_parquet(rows, data_path, tactile_left_shape[0], tactile_right_shape[0])
    pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task_text], name="task")).to_parquet(tasks_path)

    _write_video(front_paths, front_video_path, fps_value)
    _write_video(wrist_paths, wrist_video_path, fps_value)

    stats = _compute_stats(rows)
    _write_json(stats_path, stats)

    info = _build_info(
        profile=profile,
        raw_root=raw_root,
        raw_info=raw_info,
        fps=fps_value,
        task=task_text,
        total_frames=len(rows),
        total_episodes=len(episode_rows),
        front_shape=front_shape,
        wrist_shape=wrist_shape,
        tactile_left_shape=tactile_left_shape,
        tactile_right_shape=tactile_right_shape,
    )
    _write_json(info_path, info)

    return {
        "out_root": str(out_root),
        "profile": profile.name,
        "fps": fps_value,
        "total_frames": len(rows),
        "total_episodes": len(episode_rows),
        "raw_episode_count": raw_episode_count,
        "split_on_gap": split_on_gap,
        "split_count": split_count,
        "data": str(data_path),
        "front_video": str(front_video_path),
        "wrist_video": str(wrist_video_path),
    }


def _prepare_output_root(out_root: Path, *, overwrite: bool) -> None:
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_episode_parquets(raw_root: Path) -> list[Path]:
    episode_dir = raw_root / "data" / "episodes"
    if episode_dir.exists():
        paths = sorted(episode_dir.glob("episode_*.parquet"))
        if paths:
            return paths
    master = raw_root / "data" / "chunk-000" / "file-000.parquet"
    return [master] if master.exists() else []


def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _episode_segments(
    df: pd.DataFrame,
    fps: float,
    gap_factor: float,
    *,
    split_on_gap: bool,
) -> list[pd.DataFrame]:
    if df.empty:
        return []
    if not split_on_gap:
        return [df.reset_index(drop=True)]
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    max_gap = float(gap_factor / fps)
    starts = [0]
    for idx in range(1, len(df)):
        gap = timestamps[idx] - timestamps[idx - 1]
        if np.isfinite(gap) and gap > max_gap:
            starts.append(idx)
    starts.append(len(df))
    return [df.iloc[starts[i] : starts[i + 1]].reset_index(drop=True) for i in range(len(starts) - 1)]


def _open_zarr_array(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    import zarr

    return zarr.open(str(path), mode="r")


def _flat_shape(shape: tuple[int, ...]) -> list[int]:
    if len(shape) < 2:
        return [1]
    return [int(np.prod(shape[1:], dtype=np.int64))]


def _vector(value: Any, expected_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != expected_dim:
        raise ValueError(f"{name} expected dim {expected_dim}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _zarr_vector(zarr_array: Any, index: int, name: str) -> np.ndarray:
    if index < 0 or index >= int(zarr_array.shape[0]):
        raise IndexError(f"{name} zarr index {index} out of range 0..{int(zarr_array.shape[0]) - 1}")
    arr = np.asarray(zarr_array[index]).reshape(-1)
    return arr


def _float32_list(value: Any) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("float32 vector contains non-finite values")
    return arr.tolist()


def _write_data_parquet(
    rows: list[dict[str, Any]],
    data_path: Path,
    tactile_left_dim: int,
    tactile_right_dim: int,
) -> None:
    table = pa.table(
        {
            "index": pa.array([int(row["index"]) for row in rows], type=pa.int64()),
            "episode_index": pa.array([int(row["episode_index"]) for row in rows], type=pa.int64()),
            "frame_index": pa.array([int(row["frame_index"]) for row in rows], type=pa.int64()),
            "timestamp": pa.array([float(row["timestamp"]) for row in rows], type=pa.float32()),
            "task_index": pa.array([int(row["task_index"]) for row in rows], type=pa.int64()),
            STATE_KEY: _fixed_float32_list_array(rows, STATE_KEY, STATE_DIM),
            ACTION_KEY: _fixed_float32_list_array(rows, ACTION_KEY, ACTION_DIM),
            TACTILE_LEFT_KEY: _fixed_float32_list_array(rows, TACTILE_LEFT_KEY, tactile_left_dim),
            TACTILE_RIGHT_KEY: _fixed_float32_list_array(rows, TACTILE_RIGHT_KEY, tactile_right_dim),
        }
    )
    pq.write_table(table, data_path)


def _fixed_float32_list_array(rows: list[dict[str, Any]], key: str, dim: int) -> pa.Array:
    values = []
    for row_index, row in enumerate(rows):
        arr = np.asarray(row[key], dtype=np.float32).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"{key} row {row_index} expected dim {dim}, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} row {row_index} contains non-finite values")
        values.append(arr.tolist())
    return pa.array(values, type=pa.list_(pa.float32(), list_size=dim))


def _resolve_sidecar_path(raw_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return raw_root / path


def _image_shape(path: Path) -> list[int]:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
    return [int(height), int(width), 3]


def _write_video(image_paths: list[Path], out_path: Path, fps: float) -> None:
    if not image_paths:
        raise ValueError(f"no image paths for {out_path}")
    first = np.asarray(Image.open(image_paths[0]).convert("RGB"))
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {out_path}")
    try:
        for path in image_paths:
            rgb = np.asarray(Image.open(path).convert("RGB"))
            if rgb.shape[:2] != (height, width):
                raise ValueError(f"{path} shape {rgb.shape[:2]} differs from first frame {(height, width)}")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def _compute_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key in [STATE_KEY, ACTION_KEY, TACTILE_LEFT_KEY, TACTILE_RIGHT_KEY]:
        arr = np.asarray([row[key] for row in rows], dtype=np.float64)
        stats[key] = {
            "mean": np.mean(arr, axis=0).astype(float).tolist(),
            "std": np.std(arr, axis=0).astype(float).tolist(),
            "min": np.min(arr, axis=0).astype(float).tolist(),
            "max": np.max(arr, axis=0).astype(float).tolist(),
            "count": int(arr.shape[0]),
        }
    return stats


def _build_info(
    *,
    profile: ExportProfile,
    raw_root: Path,
    raw_info: dict[str, Any],
    fps: float,
    task: str,
    total_frames: int,
    total_episodes: int,
    front_shape: list[int],
    wrist_shape: list[int],
    tactile_left_shape: list[int],
    tactile_right_shape: list[int],
) -> dict[str, Any]:
    return {
        "codebase_version": "v3.0",
        "dataset_type": "lerobot_v3_style_tactile",
        "source_dataset_type": raw_info.get("dataset_type"),
        "source_raw_root": str(raw_root),
        "robot_type": "fr3",
        "tactile_profile": profile.name,
        "tactile_msg_package": profile.tactile_msg_package,
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
        "features": {
            FRONT_VIDEO_KEY: {
                "dtype": "video",
                "shape": front_shape,
                "names": ["height", "width", "channel"],
                "info": {"video.fps": fps, "video.codec": "mp4v", "video.pix_fmt": "yuv420p"},
            },
            WRIST_VIDEO_KEY: {
                "dtype": "video",
                "shape": wrist_shape,
                "names": ["height", "width", "channel"],
                "info": {"video.fps": fps, "video.codec": "mp4v", "video.pix_fmt": "yuv420p"},
            },
            STATE_KEY: {"dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES},
            ACTION_KEY: {"dtype": "float32", "shape": [ACTION_DIM], "names": ACTION_NAMES},
            TACTILE_LEFT_KEY: {
                "dtype": "float32",
                "shape": tactile_left_shape,
                "names": None,
                "description": profile.tactile_description,
            },
            TACTILE_RIGHT_KEY: {
                "dtype": "float32",
                "shape": tactile_right_shape,
                "names": None,
                "description": profile.tactile_description,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
