from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .common import (
    ACTION_KEY,
    FRONT_VIDEO_KEY,
    STATE_KEY,
    TACTILE_LEFT_KEY,
    TACTILE_RIGHT_KEY,
    WRIST_VIDEO_KEY,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one canonical sample from a LeRobot v3-style tactile dataset.")
    parser.add_argument("--dataset-root", required=True, help="Exported LeRobot v3-style dataset root")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to inspect")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index inside the selected episode")
    parser.add_argument("--global-index", type=int, default=None, help="Global dataset row index. Overrides episode/frame.")
    parser.add_argument("--save-preview", default=None, help="Optional path to save front+wrist RGB preview PNG")
    args = parser.parse_args()

    result = inspect_sample(
        dataset_root=Path(args.dataset_root),
        episode_index=int(args.episode_index),
        frame_index=int(args.frame_index),
        global_index=None if args.global_index is None else int(args.global_index),
        save_preview=None if args.save_preview is None else Path(args.save_preview),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def inspect_sample(
    *,
    dataset_root: Path,
    episode_index: int,
    frame_index: int,
    global_index: int | None,
    save_preview: Path | None,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    info = _read_json(dataset_root / "meta" / "info.json")
    df = pd.read_parquet(dataset_root / "data" / "chunk-000" / "file-000.parquet")
    episodes = pd.read_parquet(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    row_index, resolved_episode_index, resolved_frame_index = _resolve_row_index(
        df=df,
        episodes=episodes,
        episode_index=episode_index,
        frame_index=frame_index,
        global_index=global_index,
    )
    row = df.iloc[row_index]

    front = _read_video_frame(dataset_root / "videos" / FRONT_VIDEO_KEY / "chunk-000" / "file-000.mp4", row_index)
    wrist = _read_video_frame(dataset_root / "videos" / WRIST_VIDEO_KEY / "chunk-000" / "file-000.mp4", row_index)
    state = _vector(row[STATE_KEY], 7, np.float32, STATE_KEY)
    action = _vector(row[ACTION_KEY], 7, np.float32, ACTION_KEY)
    tactile_left = _feature_vector(row[TACTILE_LEFT_KEY], info, TACTILE_LEFT_KEY)
    tactile_right = _feature_vector(row[TACTILE_RIGHT_KEY], info, TACTILE_RIGHT_KEY)

    _assert_rgb_image(FRONT_VIDEO_KEY, front)
    _assert_rgb_image(WRIST_VIDEO_KEY, wrist)

    preview_path = None
    if save_preview is not None:
        preview_path = _save_preview(save_preview.expanduser().resolve(), front, wrist)

    return {
        "dataset_root": str(dataset_root),
        "episode_index": int(resolved_episode_index),
        "frame_index": int(resolved_frame_index),
        "global_index": int(row_index),
        "timestamp": float(row["timestamp"]),
        "canonical_sample": {
            FRONT_VIDEO_KEY: _image_summary(front),
            WRIST_VIDEO_KEY: _image_summary(wrist),
            STATE_KEY: _array_summary(state),
            TACTILE_LEFT_KEY: _array_summary(tactile_left),
            TACTILE_RIGHT_KEY: _array_summary(tactile_right),
            ACTION_KEY: _array_summary(action),
        },
        "preview_path": preview_path,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_row_index(
    *,
    df: pd.DataFrame,
    episodes: pd.DataFrame,
    episode_index: int,
    frame_index: int,
    global_index: int | None,
) -> tuple[int, int, int]:
    if global_index is not None:
        if global_index < 0 or global_index >= len(df):
            raise IndexError(f"global-index {global_index} out of range [0, {len(df)})")
        row = df.iloc[global_index]
        return int(global_index), int(row["episode_index"]), int(row["frame_index"])

    matches = episodes[episodes["episode_index"].astype(int) == int(episode_index)]
    if matches.empty:
        available = sorted(int(x) for x in episodes["episode_index"].tolist())
        raise ValueError(f"episode-index {episode_index} not found; available={available}")
    ep = matches.iloc[0]
    length = int(ep["length"])
    if frame_index < 0 or frame_index >= length:
        raise IndexError(f"frame-index {frame_index} out of range [0, {length}) for episode {episode_index}")
    row_index = int(ep["from"]) + int(frame_index)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"resolved row index {row_index} out of range [0, {len(df)})")
    row = df.iloc[row_index]
    if int(row["episode_index"]) != int(episode_index) or int(row["frame_index"]) != int(frame_index):
        raise ValueError(
            f"resolved row mismatch: requested episode/frame={episode_index}/{frame_index}, "
            f"got {int(row['episode_index'])}/{int(row['frame_index'])}"
        )
    return row_index, int(episode_index), int(frame_index)


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = cap.read()
    finally:
        cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"failed to decode frame {frame_index} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _vector(value: Any, expected_dim: int, dtype: np.dtype[Any] | type, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.shape != (expected_dim,):
        raise ValueError(f"{name} expected shape {(expected_dim,)}, got {arr.shape}")
    if not np.all(np.isfinite(arr.astype(np.float64))):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _feature_vector(value: Any, info: dict[str, Any], key: str) -> np.ndarray:
    feature = info["features"][key]
    expected_shape = tuple(int(x) for x in feature["shape"])
    dtype = np.dtype(str(feature["dtype"]))
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.shape != expected_shape:
        raise ValueError(f"{key} expected shape {expected_shape}, got {arr.shape}")
    return arr


def _assert_rgb_image(name: str, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} expected HWC RGB image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} expected uint8 image, got {image.dtype}")


def _image_summary(image: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(x) for x in image.shape],
        "dtype": str(image.dtype),
        "min": int(np.min(image)),
        "max": int(np.max(image)),
        "mean": float(np.mean(image)),
        "channel_mean_rgb": [float(x) for x in np.mean(image, axis=(0, 1)).tolist()],
    }


def _array_summary(arr: np.ndarray) -> dict[str, Any]:
    flat = arr.reshape(-1)
    if flat.size == 0:
        return {"shape": [int(x) for x in arr.shape], "dtype": str(arr.dtype), "size": 0}
    numeric = flat.astype(np.float64)
    return {
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
        "mean": float(np.mean(numeric)),
        "first_values": [float(x) for x in numeric[: min(8, numeric.size)].tolist()],
    }


def _save_preview(path: Path, front: np.ndarray, wrist: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    front_img = Image.fromarray(front, mode="RGB")
    wrist_img = Image.fromarray(wrist, mode="RGB")
    canvas = Image.new("RGB", (front_img.width + wrist_img.width, max(front_img.height, wrist_img.height)))
    canvas.paste(front_img, (0, 0))
    canvas.paste(wrist_img, (front_img.width, 0))
    canvas.save(path)
    return str(path)


if __name__ == "__main__":
    main()
